import os
import sys
import csv
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

# Add project root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from p2rank_utils import parse_p2rank_output
from esm_extractor import ESMFeatureExtractor
from preprocessing import (
    DEFAULT_ESM_MODEL, DEFAULT_MIN_PROB, DEFAULT_POCKET_EMBEDDING, POCKET_EMBEDDING_MODES,
    make_preprocessing_config, config_from_records, describe,
)

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
NAME_TO_LABEL = {name: i for i, name in enumerate(TARGET_NAMES)}


def load_metadata_targets(metadata_path):
    """
    Loads dataset_metadata.tsv and returns pdb_file -> metadata dictionary mapping.
    """
    targets = {}
    if not os.path.exists(metadata_path):
        return targets

    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            if not row or len(row) < 3:
                continue
            acc = row[0].strip()
            pdb_file = row[1].strip()
            cofactors_str = row[2].strip()

            if pdb_file == "NONE" or not pdb_file:
                continue

            cofactors_list = [c.strip() for c in cofactors_str.split(';') if c.strip()]
            assigned_label = None
            primary_cofactor = None
            for cof in cofactors_list:
                if cof in NAME_TO_LABEL:
                    assigned_label = NAME_TO_LABEL[cof]
                    primary_cofactor = cof
                    break

            if assigned_label is not None:
                targets[pdb_file] = {
                    'uniprot_id': acc,
                    'pdb_file': pdb_file,
                    'stem': pdb_file.replace('.pdb', '').strip(),
                    'cofactors': cofactors_str,
                    'primary_cofactor': primary_cofactor,
                    'label': assigned_label
                }
    return targets


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_pdb_dir = os.path.join(base_dir, 'structures', 'all_pdbs')
    default_prank_dir = os.path.join(base_dir, 'structures', 'p2rank_outputs')
    default_metadata = os.path.join(base_dir, 'structures', 'dataset_metadata.tsv')
    default_out = os.path.join(base_dir, 'data_prep', 'esm_dataset.pt')

    parser = argparse.ArgumentParser(description="Extract ESM-2 pocket embeddings from standalone P2Rank outputs")
    parser.add_argument('--prank-dir', type=str, default=default_prank_dir, help='Directory containing P2Rank outputs (*_predictions.csv)')
    parser.add_argument('--pdb-dir', type=str, default=default_pdb_dir, help='Folder containing PDB structures (structures/all_pdbs)')
    parser.add_argument('--metadata', type=str, default=default_metadata, help='Path to dataset_metadata.tsv')
    parser.add_argument('--out-path', type=str, default=default_out, help='Output file path for esm_dataset.pt')
    parser.add_argument('--min-prob', type=float, default=DEFAULT_MIN_PROB, help=f'Minimum P2Rank pocket probability threshold (default: {DEFAULT_MIN_PROB})')
    parser.add_argument('--pocket-embedding', type=str, default=DEFAULT_POCKET_EMBEDDING, choices=POCKET_EMBEDDING_MODES,
                        help="'slice' = pool pocket residues from the full-protein ESM pass (default); 'concat' = legacy residue-string embedding")
    parser.add_argument('--esm-model', type=str, default=DEFAULT_ESM_MODEL, help='HuggingFace ESM-2 model identifier')
    parser.add_argument('--device', type=str, default=None, help='Compute device for ESM (cuda, mps, cpu)')
    parser.add_argument('--save-interval', type=int, default=100, help='Checkpoint saving interval')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of processed proteins for testing')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)

    prep_cfg = make_preprocessing_config(
        min_prob=args.min_prob,
        pocket_embedding=args.pocket_embedding,
        long_sequences='chunk',
        esm_model=args.esm_model,
    )

    print("=" * 65)
    print("AMICO: POCKET EMBEDDINGS GENERATOR (FROM P2RANK OUTPUTS)")
    print("=" * 65)
    print(f"P2Rank outputs:  {args.prank_dir}")
    print(f"PDB directory:   {args.pdb_dir}")
    print(f"Metadata file:   {args.metadata}")
    print(f"Output path:     {args.out_path}")
    print(f"Preprocessing:   {describe(prep_cfg)}")
    print("=" * 65)

    # 1. Load metadata
    metadata_targets = load_metadata_targets(args.metadata)
    print(f"-> Loaded {len(metadata_targets)} valid target structures from metadata.")

    # 2. Load existing pocket dataset for resume capability
    existing_pockets = []
    processed_pids = set()
    if os.path.exists(args.out_path):
        print(f"-> Found existing pocket dataset in {args.out_path}, loading for resume...")
        existing_pockets = torch.load(args.out_path, weights_only=False)
        if existing_pockets:
            existing_cfg = config_from_records(existing_pockets)
            if existing_cfg != prep_cfg:
                print("❌ Existing pocket dataset was built with different preprocessing settings:")
                print(f"   existing:  {describe(existing_cfg)}")
                print(f"   requested: {describe(prep_cfg)}")
                print("   Delete/rename the file or pass a new --out-path.")
                return
        for item in existing_pockets:
            raw_pid = item['protein_id']
            base_name = os.path.basename(raw_pid)
            pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
            processed_pids.add(pid)
        print(f"-> Already processed: {len(processed_pids)} unique proteins in {args.out_path}.")

    # 3. Index PDB files on disk
    pdb_dir_path = Path(args.pdb_dir)
    if not pdb_dir_path.exists():
        print(f"❌ Error: PDB structure directory {args.pdb_dir} does not exist.")
        return

    all_pdb_files = [f for f in pdb_dir_path.glob("*.pdb") if "_pocket_" not in f.name]
    pdb_by_stem = {f.stem: f for f in all_pdb_files}

    # Assemble queue to process
    queue = []
    if metadata_targets:
        for pdb_fname, meta in metadata_targets.items():
            stem = meta['stem']
            if stem in processed_pids:
                continue
            if stem in pdb_by_stem:
                queue.append((pdb_by_stem[stem], meta))
    else:
        for f in all_pdb_files:
            stem = f.stem
            if stem in processed_pids:
                continue
            meta = {
                'uniprot_id': stem.split('_')[0],
                'pdb_file': f.name,
                'stem': stem,
                'cofactors': 'unknown',
                'primary_cofactor': 'unknown',
                'label': 0
            }
            queue.append((f, meta))

    if args.limit:
        queue = queue[:args.limit]
        print(f"-> Limit active: Processing {len(queue)} structures.")
    else:
        print(f"-> Remaining to process: {len(queue)} structures.")

    if not queue:
        print("✅ All pockets have already been extracted.")
        return

    # 4. Initialize ESM Feature Extractor
    print(f"\n-> Initializing ESM Feature Extractor ({args.esm_model})...")
    extractor = ESMFeatureExtractor(model_name=args.esm_model, device=args.device,
                                    long_sequences=prep_cfg['long_sequences'], verbose=False)

    new_pockets_list = list(existing_pockets)
    new_counter = 0
    missing_prank = 0
    no_pocket_count = 0

    pbar = tqdm(queue, desc="ESM-2 Pocket Extraction")
    for pdb_path, meta in pbar:
        stem = meta['stem']
        label = meta['label']

        try:
            parsed = parse_p2rank_output(args.prank_dir, pdb_path, min_prob=args.min_prob)
            feats = extractor.extract_features(parsed, pocket_embedding=args.pocket_embedding)
            if not feats['pockets']:
                no_pocket_count += 1

            for p_info, p_feat in zip(feats['pockets'], feats['pocket_features']):
                new_pockets_list.append({
                    'protein_id': f"{stem}_pocket_{p_info['pocket_id']}.pdb",
                    'features': p_feat,
                    'label': label,
                    'probability': p_info.get('probability', 0.0),
                    'score': p_info.get('score', 0.0),
                    'center': p_info.get('center', [0.0, 0.0, 0.0]),
                    'residue_count': p_info.get('residue_count', len(p_info.get('sequence', ''))),
                    'preprocessing': prep_cfg,
                })

            new_counter += 1

            if new_counter % args.save_interval == 0:
                torch.save(new_pockets_list, args.out_path)

        except (FileNotFoundError, ValueError):
            missing_prank += 1
        except Exception as e:
            print(f"\n❌ Error extracting pockets for {stem}: {e}")

    # Final save
    torch.save(new_pockets_list, args.out_path)
    print("\n" + "=" * 65)
    print("✅ COMPLETED!")
    print(f"Total pockets saved to {args.out_path}: {len(new_pockets_list)}")
    if extractor.num_long_sequences:
        print(f"ℹ️  {extractor.num_long_sequences} sequence(s) exceeded 1022 aa and were embedded in overlapping windows.")
    if no_pocket_count:
        print(f"⚠️  {no_pocket_count} protein(s) had no pocket with probability >= {args.min_prob} and are excluded from training.")
    if missing_prank > 0:
        print(f"⚠️ Warning: For {missing_prank} proteins, P2Rank outputs were not found in {args.prank_dir}.")
    print("=" * 65)


if __name__ == "__main__":
    main()
