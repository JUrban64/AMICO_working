#!/usr/bin/env python3
"""
AMICO Residue Dataset Generator (build_residue_dataset.py)
==========================================================
Processes 3D PDB protein structures with P2Rank and ESM-2 into a unified
residue-level dataset (residues_dataset.pt) for ResidueSelfAttentionMIL
and ResidueLigandCrossAttentionMIL models.

For each protein, extracts:
1. Whole-protein ESM-2 sequence embedding (context anchor on Token 0).
2. Per-residue ESM-2 embeddings for candidate binding pocket residues.
3. 3D C-alpha Cartesian coordinates for pairwise geometric distance bias.
4. P2Rank residue-level metadata (residue name, chain, sequence index, score).
5. Cofactor ground truth label (0: acetyl-CoA, 1: ATP, 2: B12, 3: FAD, 4: NAD).
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

# Ensure project root directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.p2rank_utils import run_p2rank_batch, find_p2rank_executable
from dataset import load_split_ids, match_id
from residue_attention.feature_extractor_residues import ResidueFeatureExtractor
from residue_attention.dataset_residues import TARGET_NAMES

NAME_TO_LABEL = {name: i for i, name in enumerate(TARGET_NAMES)}


def load_metadata_targets(metadata_path):
    """
    Loads dataset_metadata.tsv and returns mapping pdb_filename -> metadata dict.
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
                stem = pdb_file.replace('.pdb', '')
                meta_item = {
                    'uniprot_id': acc,
                    'pdb_file': pdb_file,
                    'stem': stem,
                    'cofactors': cofactors_str,
                    'primary_cofactor': primary_cofactor,
                    'label': assigned_label,
                    'ec': ec,
                    'length': length,
                    'source': source
                }
                targets[pdb_file] = meta_item
                targets[stem] = meta_item

    return targets


def build_residue_dataset(
    pdb_dir='structures/all_pdbs',
    metadata_path='structures/dataset_metadata.tsv',
    prank_out_dir='structures/p2rank_outputs',
    output_path='data_prep/residues_dataset.pt',
    prank_exec=None,
    esm_model='facebook/esm2_t33_650M_UR50D',
    device=None,
    min_prob=0.10,
    max_pockets=5,
    skip_p2rank=False,
    threads=8,
    chunk_size=500,
    save_interval=50,
    limit=None,
    single_pdb=None,
    split_suffix=None
):
    # Resolve absolute paths
    base_dir = PROJECT_ROOT
    pdb_dir = os.path.join(base_dir, pdb_dir) if not os.path.isabs(pdb_dir) else pdb_dir
    metadata_path = os.path.join(base_dir, metadata_path) if not os.path.isabs(metadata_path) else metadata_path
    prank_out_dir = os.path.join(base_dir, prank_out_dir) if not os.path.isabs(prank_out_dir) else prank_out_dir
    output_path = os.path.join(base_dir, output_path) if not os.path.isabs(output_path) else output_path

    os.makedirs(prank_out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print("=" * 65)
    print("AMICO: RESIDUE DATASET GENERATION PIPELINE")
    print("=" * 65)
    print(f"PDB directory:    {pdb_dir}")
    print(f"Metadata file:    {metadata_path}")
    print(f"P2Rank directory: {prank_out_dir}")
    print(f"Output file:      {output_path}")
    print(f"Min pocket prob:  {min_prob} | Max pockets: {max_pockets}")
    print(f"ESM-2 model:      {esm_model}")
    print("=" * 65)

    # 1. Load metadata
    metadata_targets = load_metadata_targets(metadata_path)
    if metadata_targets:
        print(f"-> Loaded {len(metadata_targets)} structure annotations from {metadata_path}")
    else:
        print(f"-> Notice: Metadata file {metadata_path} not found. Generating default labels (label=0).")

    # 2. Check existing results for resume capability
    existing_items = []
    processed_pids = set()
    if os.path.exists(output_path):
        print(f"-> Loading existing dataset from {output_path} to resume...")
        raw_loaded = torch.load(output_path, map_location='cpu', weights_only=False)
        if isinstance(raw_loaded, list):
            existing_items = raw_loaded
        elif isinstance(raw_loaded, dict) and 'all' in raw_loaded:
            existing_items = raw_loaded['all']
        elif isinstance(raw_loaded, dict) and 'train' in raw_loaded:
            existing_items = raw_loaded['train'] + raw_loaded.get('val', []) + raw_loaded.get('test', [])

        for it in existing_items:
            processed_pids.add(it['protein_id'])
        print(f"-> Found {len(processed_pids)} already processed proteins.")

    # 3. Locate target PDB structures
    queue = []
    if single_pdb:
        single_path = Path(single_pdb)
        if not single_path.is_absolute():
            single_path = Path(base_dir) / single_path
        if not single_path.exists():
            raise FileNotFoundError(f"Single PDB file not found: {single_path}")
        stem = single_path.stem
        meta = metadata_targets.get(single_path.name, metadata_targets.get(stem, {
            'stem': stem,
            'label': 1,  # Default ATP
            'primary_cofactor': 'ATP'
        }))
        queue.append((single_path, meta))
    else:
        pdb_dir_path = Path(pdb_dir)
        if not pdb_dir_path.exists():
            print(f"❌ Error: Folder {pdb_dir} does not exist.")
            return

        all_pdb_files = [f for f in pdb_dir_path.glob("*.pdb") if "_pocket_" not in f.name]
        print(f"-> Found {len(all_pdb_files)} PDB structures in {pdb_dir}")

        for f in all_pdb_files:
            stem = f.stem
            if stem in processed_pids:
                continue
            meta = metadata_targets.get(f.name, metadata_targets.get(stem, {
                'stem': stem,
                'label': 0,
                'primary_cofactor': 'unknown'
            }))
            queue.append((f, meta))

    if limit:
        queue = queue[:limit]
        print(f"-> Limit active: Processing {len(queue)} structures.")
    else:
        print(f"-> Remaining to process: {len(queue)} structures.")

    if not queue:
        print("✅ All requested structures are already processed!")
        return

    # 4. Run P2Rank batch if needed
    if not skip_p2rank:
        prank_bin = find_p2rank_executable(prank_exec)
        print(f"-> Using P2Rank binary: {prank_bin}")
        queue_pdb_paths = [pdb_path for pdb_path, _ in queue]
        print(f"\n🚀 Running batch P2Rank prediction for {len(queue_pdb_paths)} structures...")
        run_p2rank_batch(
            queue_pdb_paths,
            prank_exec=prank_bin,
            output_dir=prank_out_dir,
            config='alphafold',
            threads=threads,
            chunk_size=chunk_size
        )
    else:
        print(f"\n⚡ Skipping P2Rank execution (--skip-p2rank). Reusing predictions from {prank_out_dir}")

    # 5. Initialize Feature Extractor
    print(f"\n-> Initializing ResidueFeatureExtractor with {esm_model}...")
    extractor = ResidueFeatureExtractor(model_name=esm_model, device=device)

    # 6. Extract residue features
    dataset_records = list(existing_items)
    saved_count = len(dataset_records)
    errors = []

    pbar = tqdm(queue, desc="Extracting residue embeddings & 3D coords")
    for pdb_path, meta in pbar:
        stem = meta['stem']
        label = meta.get('label', 0)
        cofactor_name = meta.get('primary_cofactor', 'unknown')

        try:
            sample = extractor.extract_from_pdb_and_p2rank(
                pdb_path=pdb_path,
                prank_dir=prank_out_dir,
                min_prob=min_prob,
                max_pockets=max_pockets
            )

            record = {
                'protein_id': stem,
                'label': torch.tensor(label, dtype=torch.long),
                'residue_features': sample['residue_features'],       # [M, 1280]
                'residue_coords': sample['residue_coords'],           # [M, 3]
                'full_protein_feature': sample['full_protein_feature'], # [1280]
                'residue_metadata': sample['residue_metadata'],       # list of dicts
                'primary_cofactor': cofactor_name,
                'num_residues': sample['residue_features'].size(0)
            }
            dataset_records.append(record)
            saved_count += 1

            if saved_count % save_interval == 0:
                torch.save(dataset_records, output_path)

        except Exception as e:
            errors.append((stem, str(e)))

    # 7. Partition into splits if requested
    final_output = dataset_records
    if split_suffix:
        print(f"\n-> Partitioning dataset into train/val/test splits using suffix '{split_suffix}'...")
        train_ids, val_ids, test_ids = load_split_ids(base_dir, split_suffix=split_suffix)
        train_list, val_list, test_list, unassigned_list = [], [], [], []

        for item in dataset_records:
            pid = item['protein_id']
            if match_id(pid, train_ids):
                train_list.append(item)
            elif match_id(pid, val_ids):
                val_list.append(item)
            elif match_id(pid, test_ids):
                test_list.append(item)
            else:
                unassigned_list.append(item)

        print(f"   Split counts -> Train: {len(train_list)}, Val: {len(val_list)}, Test: {len(test_list)}")
        if unassigned_list:
            print(f"   Note: {len(unassigned_list)} proteins were not in split lists (unassigned).")

        final_output = {
            'train': train_list,
            'val': val_list,
            'test': test_list,
            'all': dataset_records
        }

    # Final save
    torch.save(final_output, output_path)
    print("\n" + "=" * 65)
    print(f"✅ SUCCESS: Saved {len(dataset_records)} proteins to {output_path}")
    if errors:
        print(f"⚠️ Encountered {len(errors)} errors during processing (e.g. no pockets found).")
    print("=" * 65)


def resplit_residue_dataset(dataset_path, split_suffix, output_path=None, base_dir=None):
    """
    Re-partitions an already generated residues_dataset.pt without re-extracting ESM features.
    """
    if base_dir is None:
        base_dir = PROJECT_ROOT
    if output_path is None:
        output_path = dataset_path

    print(f"Loading existing residue dataset from {dataset_path}...")
    raw_data = torch.load(dataset_path, map_location='cpu', weights_only=False)

    all_items = []
    if isinstance(raw_data, dict):
        if 'all' in raw_data and len(raw_data['all']) > 0:
            all_items = raw_data['all']
        elif 'train' in raw_data:
            all_items = list(raw_data['train']) + list(raw_data.get('val', [])) + list(raw_data.get('test', []))
        else:
            for v in raw_data.values():
                if isinstance(v, list):
                    all_items.extend(v)
    elif isinstance(raw_data, list):
        all_items = raw_data
    else:
        raise ValueError(f"Unrecognized dataset format in {dataset_path}")

    print(f"-> Found {len(all_items)} proteins to partition.")
    print(f"-> Loading split IDs using suffix '{split_suffix}'...")
    train_ids, val_ids, test_ids = load_split_ids(base_dir, split_suffix=split_suffix)

    if len(train_ids) == 0 and len(val_ids) == 0 and len(test_ids) == 0:
        raise ValueError(f"Could not load any IDs for split suffix '{split_suffix}'. Check your data_prep directory.")

    train_list, val_list, test_list, unassigned_list = [], [], [], []
    for item in all_items:
        pid = item['protein_id']
        if match_id(pid, train_ids):
            train_list.append(item)
        elif match_id(pid, val_ids):
            val_list.append(item)
        elif match_id(pid, test_ids):
            test_list.append(item)
        else:
            unassigned_list.append(item)

    print(f"\n✅ Partition completed successfully:")
    print(f"   Train:      {len(train_list)}")
    print(f"   Validation: {len(val_list)}")
    print(f"   Test:       {len(test_list)}")
    if unassigned_list:
        print(f"   Unassigned: {len(unassigned_list)} (proteins not present in split files)")

    final_output = {
        'train': train_list,
        'val': val_list,
        'test': test_list,
        'all': all_items
    }
    torch.save(final_output, output_path)
    print(f"\n💾 Saved re-partitioned dataset to {output_path}")
    return final_output


def main():
    parser = argparse.ArgumentParser(description="Generate or Re-partition AMICO Residue Dataset (.pt)")
    parser.add_argument('--pdb-dir', type=str, default='structures/all_pdbs',
                        help='Folder containing PDB structures (default: structures/all_pdbs)')
    parser.add_argument('--metadata', type=str, default='structures/dataset_metadata.tsv',
                        help='Path to dataset_metadata.tsv annotations')
    parser.add_argument('--prank-out-dir', type=str, default='structures/p2rank_outputs',
                        help='Directory storing P2Rank prediction outputs')
    parser.add_argument('--prank-exec', type=str, default=None,
                        help='Path to prank executable binary')
    parser.add_argument('--output-path', type=str, default='data_prep/residues_dataset.pt',
                        help='Destination path for residue dataset (.pt)')
    parser.add_argument('--esm-model', type=str, default='facebook/esm2_t33_650M_UR50D',
                        help='HuggingFace ESM-2 model name')
    parser.add_argument('--device', type=str, default=None,
                        help='Compute device for ESM (cuda, mps, cpu)')
    parser.add_argument('--min-prob', type=float, default=0.30,
                        help='Minimum pocket probability threshold (default: 0.30)')
    parser.add_argument('--max-pockets', type=int, default=5,
                        help='Maximum number of top pockets to pool residues from (default: 5)')
    parser.add_argument('--skip-p2rank', action='store_true',
                        help='Skip running P2Rank (reuse existing CSV outputs)')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of CPU threads for P2Rank')
    parser.add_argument('--chunk-size', type=int, default=500,
                        help='Batch size for batch P2Rank processing')
    parser.add_argument('--save-interval', type=int, default=50,
                        help='Checkpoint save interval')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of structures to process')
    parser.add_argument('--single-pdb', type=str, default=None,
                        help='Process a single PDB file (useful for testing)')
    parser.add_argument('--split-suffix', type=str, default=None,
                        help='Optional split suffix (e.g. mil_0.5_nr0.9 or mil0.5_nr0.9) to partition into train/val/test')
    parser.add_argument('--resplit-only', action='store_true',
                        help='Fast mode: re-partition an existing .pt file into train/val/test without re-extracting ESM')

    args = parser.parse_args()

    if args.resplit_only:
        if not args.split_suffix:
            parser.error("--resplit-only requires --split-suffix (e.g. --split-suffix mil_0.5_nr0.9)")
        resplit_residue_dataset(
            dataset_path=args.output_path,
            split_suffix=args.split_suffix,
            output_path=args.output_path
        )
        return

    build_residue_dataset(
        pdb_dir=args.pdb_dir,
        metadata_path=args.metadata,
        prank_out_dir=args.prank_out_dir,
        output_path=args.output_path,
        prank_exec=args.prank_exec,
        esm_model=args.esm_model,
        device=args.device,
        min_prob=args.min_prob,
        max_pockets=args.max_pockets,
        skip_p2rank=args.skip_p2rank,
        threads=args.threads,
        chunk_size=args.chunk_size,
        save_interval=args.save_interval,
        limit=args.limit,
        single_pdb=args.single_pdb,
        split_suffix=args.split_suffix
    )


if __name__ == '__main__':
    main()
