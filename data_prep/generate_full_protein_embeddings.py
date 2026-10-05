import os
import csv
import argparse
import torch
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser
import glob
import sys

# Add project root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from esm_extractor import ESMFeatureExtractor


def is_aa(residue):
    return residue.get_id()[0] == ' '


def get_full_sequence_from_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
    except Exception as e:
        print(f"Error parsing {pdb_path}: {e}")
        return None

    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    sequence = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue):
                    resname = residue.get_resname()
                    if resname in three_to_one:
                        sequence.append(three_to_one[resname])
                    else:
                        sequence.append('X')
                        
    seq_str = ''.join(sequence)
    if len(seq_str) == 0:
        return None
    return seq_str


def find_metadata_tsv(candidate_paths):
    for p in candidate_paths:
        if p and os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_pdb_dir = os.path.join(base_dir, 'structures', 'all_pdbs')
    if not os.path.exists(default_pdb_dir):
        default_pdb_dir = os.path.join(base_dir, 'structures')

    default_metadata = os.path.join(base_dir, 'structures', 'dataset_metadata.tsv')

    parser = argparse.ArgumentParser(description="Generate whole-protein ESM-2 embeddings from PDB structures")
    parser.add_argument('--pdb-dir', type=str, default=default_pdb_dir, help='Folder containing PDB structures (e.g. structures/all_pdbs)')
    parser.add_argument('--metadata', type=str, default=default_metadata, help='Path to dataset_metadata.tsv (if available)')
    parser.add_argument('--dataset-path', type=str, default=os.path.join(base_dir, 'data_prep', 'esm_dataset.pt'), help='Path to esm_dataset.pt (optional)')
    parser.add_argument('--out-path', type=str, default=os.path.join(base_dir, 'data_prep', 'esm_full_proteins.pt'), help='Output embeddings file path')
    parser.add_argument('--model-name', type=str, default='facebook/esm2_t33_650M_UR50D', help='HuggingFace ESM-2 model identifier')
    parser.add_argument('--device', type=str, default=None, help='Compute device (cuda, mps, cpu)')
    parser.add_argument('--save-interval', type=int, default=250, help='Checkpoint saving interval')
    args = parser.parse_args()

    out_path = args.out_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # 1. Determine proteins to process
    unique_pids = set()
    metadata_file = find_metadata_tsv([args.metadata, os.path.join(args.pdb_dir, '..', 'dataset_metadata.tsv'), os.path.join(args.pdb_dir, 'dataset_metadata.tsv')])
    
    if metadata_file:
        print(f"-> Loading protein IDs from metadata: {metadata_file}")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)  # skip header
            for row in reader:
                if not row or len(row) < 2:
                    continue
                pdb_fname = row[1].strip()
                if pdb_fname and pdb_fname != 'NONE':
                    pid = pdb_fname.replace('.pdb', '').strip()
                    unique_pids.add(pid)
        print(f"-> Found {len(unique_pids)} proteins in dataset_metadata.tsv.")
    elif os.path.exists(args.dataset_path):
        print(f"-> Loading protein IDs from {args.dataset_path}...")
        raw_data = torch.load(args.dataset_path, weights_only=False)
        for item in raw_data:
            raw_pid = item['protein_id']
            base_name = os.path.basename(raw_pid)
            pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
            unique_pids.add(pid)
        print(f"-> Found {len(unique_pids)} unique proteins in esm_dataset.pt.")
    else:
        print(f"-> Metadata or esm_dataset.pt not found. Processing all PDB files in {args.pdb_dir}.")

    # 2. Resume from existing output
    if os.path.exists(out_path):
        print(f"-> Found existing run in {out_path}, loading for resume...")
        full_embeddings_dict = torch.load(out_path, weights_only=False)
        print(f"-> Already processed: {len(full_embeddings_dict)} records.")
    else:
        full_embeddings_dict = {}

    # 3. Index files on disk
    print(f"-> Searching for PDB files in {args.pdb_dir}...")
    all_pdb_files = glob.glob(os.path.join(args.pdb_dir, '**', '*.pdb'), recursive=True)
    print(f"-> Total PDB files found: {len(all_pdb_files)}")
    if len(all_pdb_files) == 0:
        print("❌ ERROR: Specified directory contains no .pdb files or does not exist.")
        return

    pdb_map = {}
    for f in all_pdb_files:
        if '_pocket_' in f:
            continue
        basename = os.path.basename(f)
        pid = basename.replace('.pdb', '')
        pdb_map[pid] = f

    if not unique_pids:
        unique_pids = set(pdb_map.keys())

    print(f"-> Total proteins to verify / process: {len(unique_pids)}.")

    # 4. Initialize ESM model
    extractor = ESMFeatureExtractor(model_name=args.model_name, device=args.device)

    missing_pdbs = 0
    newly_processed = 0
    pbar = tqdm(sorted(list(unique_pids)), desc="Extracting ESM Protein Embeddings")

    for pid in pbar:
        if pid in full_embeddings_dict:
            continue

        pdb_path = pdb_map.get(pid)
        if not pdb_path:
            clean_pid = pid.split('_')[0]
            if clean_pid in pdb_map:
                pdb_path = pdb_map[clean_pid]
            else:
                matches = [f for f in all_pdb_files if clean_pid in os.path.basename(f) and '_pocket_' not in f]
                if matches:
                    pdb_path = matches[0]

        if not pdb_path:
            missing_pdbs += 1
            continue

        seq = get_full_sequence_from_pdb(pdb_path)
        if not seq:
            print(f"Warning: Could not extract sequence from {pdb_path}.")
            continue

        try:
            mean_pooled_emb = extractor.extract_sequence_embedding(seq)  # torch.FloatTensor [1280]
            full_embeddings_dict[pid] = mean_pooled_emb
            
            clean_pid = pid.split('_')[0]
            if clean_pid not in full_embeddings_dict:
                full_embeddings_dict[clean_pid] = mean_pooled_emb
                
            newly_processed += 1
        except Exception as e:
            print(f"Error extracting ESM embedding for {pid}: {e}")

        if newly_processed > 0 and newly_processed % args.save_interval == 0:
            torch.save(full_embeddings_dict, out_path)

    torch.save(full_embeddings_dict, out_path)
    print("\n" + "=" * 50)
    print(f"✅ Completed! Saved {len(full_embeddings_dict)} protein embeddings to {out_path}")
    if missing_pdbs > 0:
        print(f"⚠️ Warning: For {missing_pdbs} IDs, PDB files were not found on disk.")
    print("=" * 50)


if __name__ == "__main__":
    main()
