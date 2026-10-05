import os
import sys
import csv
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

# Add project root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from p2rank_utils import run_p2rank, run_p2rank_batch, parse_p2rank_output, find_p2rank_executable, get_full_sequence_from_pdb
from esm_extractor import ESMFeatureExtractor
from preprocessing import (
    DEFAULT_ESM_MODEL, DEFAULT_MIN_PROB, DEFAULT_POCKET_EMBEDDING, POCKET_EMBEDDING_MODES,
    make_preprocessing_config, config_from_records, describe,
)

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
NAME_TO_LABEL = {name: i for i, name in enumerate(TARGET_NAMES)}


def load_metadata_targets(metadata_path):
    """
    Loads dataset_metadata.tsv and returns mapping pdb_filename -> {uniprot_id, cofactors, label, ec, length, source}.
    """
    targets = {}
    if not os.path.exists(metadata_path):
        return targets

    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        headers = next(reader, None)
        for row in reader:
            if not row or len(row) < 3:
                continue
            acc = row[0].strip()
            pdb_file = row[1].strip()
            cofactors_str = row[2].strip()
            ec = row[3].strip() if len(row) > 3 else "unassigned"
            length = int(row[4].strip()) if len(row) > 4 and row[4].strip().isdigit() else 0
            source = row[5].strip() if len(row) > 5 else "unknown"

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
                    'stem': pdb_file.replace('.pdb', ''),
                    'cofactors': cofactors_str,
                    'primary_cofactor': primary_cofactor,
                    'label': assigned_label,
                    'ec': ec,
                    'length': length,
                    'source': source
                }
    return targets


def main():
    default_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_pdb_dir = os.path.join(default_base, 'structures', 'all_pdbs')
    default_metadata = os.path.join(default_base, 'structures', 'dataset_metadata.tsv')
    default_pockets_out = os.path.join(default_base, 'data_prep', 'esm_dataset.pt')
    default_full_out = os.path.join(default_base, 'data_prep', 'esm_full_proteins.pt')
    default_prank_dir = os.path.join(default_base, 'structures', 'p2rank_outputs')

    parser = argparse.ArgumentParser(description="Process PDB structures (P2Rank + ESM-2) into esm_dataset.pt and esm_full_proteins.pt")
    parser.add_argument('--pdb-dir', type=str, default=default_pdb_dir, help='Folder containing PDB structures (structures/all_pdbs)')
    parser.add_argument('--metadata', type=str, default=default_metadata, help='Path to dataset_metadata.tsv')
    parser.add_argument('--pockets-out', type=str, default=default_pockets_out, help='Output file for pocket dataset (.pt)')
    parser.add_argument('--full-proteins-out', type=str, default=default_full_out, help='Output file for full protein embeddings (.pt)')
    parser.add_argument('--prank-out-dir', type=str, default=default_prank_dir, help='Directory for storing P2Rank predictions')
    parser.add_argument('--prank-exec', type=str, default=None, help='Custom path to prank executable binary')
    parser.add_argument('--config', type=str, default='alphafold', help='P2Rank configuration (alphafold / default)')
    parser.add_argument('--threads', type=int, default=8, help='Number of CPU threads for P2Rank')
    parser.add_argument('--chunk-size', type=int, default=500, help='Batch size for batch P2Rank processing')
    parser.add_argument('--esm-model', type=str, default=DEFAULT_ESM_MODEL, help='HuggingFace ESM-2 model identifier')
    parser.add_argument('--device', type=str, default=None, help='Compute device for ESM (cuda, mps, cpu)')
    parser.add_argument('--min-prob', type=float, default=DEFAULT_MIN_PROB, help=f'Minimum P2Rank pocket probability threshold (default: {DEFAULT_MIN_PROB})')
    parser.add_argument('--pocket-embedding', type=str, default=DEFAULT_POCKET_EMBEDDING, choices=POCKET_EMBEDDING_MODES,
                        help="'slice' = pool pocket residues from the full-protein ESM pass (default); 'concat' = legacy residue-string embedding")
    parser.add_argument('--skip-p2rank', action='store_true', help='Skip running P2Rank (reuse existing prediction CSV files)')
    parser.add_argument('--pockets-only', action='store_true', help='Generate only esm_dataset.pt (skip whole-protein embeddings)')
    parser.add_argument('--full-only', action='store_true', help='Generate only esm_full_proteins.pt (skip pockets)')
    parser.add_argument('--save-interval', type=int, default=100, help='Checkpoint saving interval')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of processed proteins for testing')
    args = parser.parse_args()

    os.makedirs(args.prank_out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.pockets_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.full_proteins_out)), exist_ok=True)

    # Settings stamped on every record and later copied into training checkpoints.
    prep_cfg = make_preprocessing_config(
        min_prob=args.min_prob,
        pocket_embedding=args.pocket_embedding,
        long_sequences='chunk',
        esm_model=args.esm_model,
    )

    print("=" * 65)
    print("AMICO: DATA PREPARATION PIPELINE (P2Rank + ESM-2)")
    print("=" * 65)
    print(f"PDB directory:    {args.pdb_dir}")
    print(f"Metadata file:    {args.metadata}")
    print(f"Pockets output:   {args.pockets_out}")
    print(f"Full output:      {args.full_proteins_out}")
    print(f"P2Rank threads:   {args.threads} | Config: {args.config} | Chunk: {args.chunk_size}")
    print(f"Preprocessing:    {describe(prep_cfg)}")
    if args.skip_p2rank:
        print("⚡ P2Rank run:     SKIPPED (reusing existing CSV outputs)")
    print("=" * 65)

    # 1. Load metadata
    metadata_targets = load_metadata_targets(args.metadata)
    print(f"-> Loaded {len(metadata_targets)} valid target structures from metadata.")

    # 2. Load existing results for resume capability
    existing_pockets = []
    processed_pids = set()
    if os.path.exists(args.pockets_out):
        print(f"-> Found existing pocket dataset in {args.pockets_out}, loading...")
        existing_pockets = torch.load(args.pockets_out, weights_only=False)
        if existing_pockets:
            existing_cfg = config_from_records(existing_pockets)
            if existing_cfg != prep_cfg:
                print("❌ Existing pocket dataset was built with different preprocessing settings:")
                print(f"   existing:  {describe(existing_cfg)}")
                print(f"   requested: {describe(prep_cfg)}")
                print("   Resuming would mix representations. Delete/rename both output files "
                      "(esm_dataset.pt and esm_full_proteins.pt) or pass new --pockets-out/--full-proteins-out paths.")
                return
        for item in existing_pockets:
            raw_pid = item['protein_id']
            base_name = os.path.basename(raw_pid)
            pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
            processed_pids.add(pid)
        print(f"-> Already processed: {len(processed_pids)} proteins in esm_dataset.pt.")

    full_proteins_dict = {}
    if os.path.exists(args.full_proteins_out):
        print(f"-> Found existing whole-protein embeddings in {args.full_proteins_out}, loading...")
        full_proteins_dict = torch.load(args.full_proteins_out, weights_only=False)
        print(f"-> Already processed: {len(full_proteins_dict)} proteins in esm_full_proteins.pt.")

    # 3. Locate PDB files on disk
    pdb_dir_path = Path(args.pdb_dir)
    if not pdb_dir_path.exists():
        print(f"❌ Error: Folder {args.pdb_dir} does not exist.")
        return

    all_pdb_files = [f for f in pdb_dir_path.glob("*.pdb") if "_pocket_" not in f.name]
    pdb_by_stem = {f.stem: f for f in all_pdb_files}
    print(f"-> Found {len(all_pdb_files)} PDB structures on disk.")

    # Assemble queue to process
    queue = []
    if metadata_targets:
        for pdb_fname, meta in metadata_targets.items():
            stem = meta['stem']
            need_pockets = (not args.full_only) and (stem not in processed_pids)
            need_full = (not args.pockets_only) and (stem not in full_proteins_dict)
            if not need_pockets and not need_full:
                continue
            if stem in pdb_by_stem:
                queue.append((pdb_by_stem[stem], meta))
    else:
        for f in all_pdb_files:
            stem = f.stem
            need_pockets = (not args.full_only) and (stem not in processed_pids)
            need_full = (not args.pockets_only) and (stem not in full_proteins_dict)
            if not need_pockets and not need_full:
                continue
            meta = {
                'uniprot_id': stem.split('_')[0],
                'pdb_file': f.name,
                'stem': stem,
                'cofactors': 'unknown',
                'primary_cofactor': 'unknown',
                'label': 0,
                'ec': 'unassigned',
                'length': 0,
                'source': 'disk'
            }
            queue.append((f, meta))

    if args.limit:
        queue = queue[:args.limit]
        print(f"-> Limit active: Processing {len(queue)} structures.")
    else:
        print(f"-> Remaining to process: {len(queue)} structures.")

    if not queue:
        print("✅ All requested structures are already processed.")
        return

    # 4. Run P2Rank batch mode if not skipped
    if not args.skip_p2rank and not args.full_only:
        prank_bin = find_p2rank_executable(args.prank_exec)
        print(f"-> Using P2Rank binary: {prank_bin}")
        queue_pdb_paths = [pdb_path for pdb_path, _ in queue]
        print(f"\n🚀 Running batch P2Rank prediction for {len(queue_pdb_paths)} structures...")
        run_p2rank_batch(
            queue_pdb_paths,
            prank_exec=prank_bin,
            output_dir=args.prank_out_dir,
            config=args.config,
            threads=args.threads,
            chunk_size=args.chunk_size
        )
    elif args.skip_p2rank and not args.full_only:
        print(f"\n⚡ Skipping P2Rank execution (--skip-p2rank). Reusing CSV predictions from {args.prank_out_dir}")

    # 5. Initialize ESM feature extractor (long sequences are windowed, not truncated)
    print("\n-> Initializing ESM Feature Extractor...")
    extractor = ESMFeatureExtractor(model_name=args.esm_model, device=args.device,
                                    long_sequences=prep_cfg['long_sequences'], verbose=False)

    new_pockets_list = list(existing_pockets)
    new_counter = 0
    no_pocket_pids = []

    def store_full(stem, full_emb):
        full_proteins_dict[stem] = full_emb
        clean_acc = stem.split('_')[0]
        if clean_acc not in full_proteins_dict:
            full_proteins_dict[clean_acc] = full_emb

    pbar = tqdm(queue, desc="ESM-2 Extraction (pockets & proteins)")
    for pdb_path, meta in pbar:
        stem = meta['stem']
        label = meta['label']
        need_pockets = (not args.full_only) and (stem not in processed_pids)
        need_full = (not args.pockets_only) and (stem not in full_proteins_dict)

        try:
            if need_pockets:
                # One ESM pass yields both the global embedding and the pocket embeddings.
                parsed = parse_p2rank_output(args.prank_out_dir, pdb_path, min_prob=args.min_prob)
                feats = extractor.extract_features(parsed, pocket_embedding=args.pocket_embedding)

                if not args.pockets_only:
                    store_full(stem, feats['full_protein_feature'])

                if not feats['pockets']:
                    no_pocket_pids.append(stem)

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
            elif need_full:
                full_seq, _, _ = get_full_sequence_from_pdb(pdb_path)
                if full_seq:
                    store_full(stem, extractor.extract_sequence_embedding(full_seq))

            new_counter += 1

            # Save checkpoint
            if new_counter % args.save_interval == 0:
                torch.save(new_pockets_list, args.pockets_out)
                torch.save(full_proteins_dict, args.full_proteins_out)

        except Exception as e:
            print(f"\n❌ Error processing {stem}: {e}")

    # Final save
    torch.save(new_pockets_list, args.pockets_out)
    torch.save(full_proteins_dict, args.full_proteins_out)

    print("\n" + "=" * 65)
    print("✅ COMPLETED!")
    print(f"Saved pockets to {args.pockets_out}: {len(new_pockets_list)}")
    print(f"Saved whole proteins to {args.full_proteins_out}: {len(full_proteins_dict)}")
    if extractor.num_long_sequences:
        print(f"ℹ️  {extractor.num_long_sequences} sequence(s) exceeded 1022 aa and were embedded in overlapping windows.")
    if no_pocket_pids:
        print(f"⚠️  {len(no_pocket_pids)} protein(s) had no pocket with probability >= {args.min_prob}. "
              f"They produce no MIL bag and are excluded from training; predict.py flags such proteins as NO_POCKETS.")
    print("=" * 65)


if __name__ == "__main__":
    main()
