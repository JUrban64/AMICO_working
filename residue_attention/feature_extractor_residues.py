import os
import sys
import re
import csv
from pathlib import Path
import torch
import numpy as np

# Ensure root directory is accessible
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.p2rank_utils import get_full_sequence_from_pdb, run_p2rank
from utils.esm_extractor import ESMFeatureExtractor

THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


def extract_ca_coords_from_pdb(pdb_path):
    """
    Extracts C-alpha (CA) 3D coordinates for all standard amino acid residues.
    Returns: dict mapping (chain_id, resseq) -> np.ndarray [3]
    """
    ca_coords = {}
    with open(pdb_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith(('ATOM  ', 'HETATM')):
                atom_name = line[12:16].strip()
                if atom_name == 'CA':
                    resname = line[17:20].strip()
                    chain_id = line[21].strip()
                    resseq = line[22:26].strip()
                    if resname in THREE_TO_ONE:
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            ca_coords[(chain_id, resseq)] = np.array([x, y, z], dtype=np.float32)
                            if resseq not in ca_coords:
                                ca_coords[resseq] = np.array([x, y, z], dtype=np.float32)
                        except (ValueError, IndexError):
                            continue
    return ca_coords


def parse_p2rank_residues_for_protein(prank_output_dir, pdb_path, min_prob=0.10, max_pockets=5):
    """
    Parses P2Rank prediction outputs and collects unique residues belonging to candidate pockets.
    
    Args:
        prank_output_dir: directory containing P2Rank outputs
        pdb_path: path to PDB file
        min_prob: minimum pocket probability threshold
        max_pockets: maximum number of top pockets to consider
        
    Returns:
        dict:
            'full_sequence': str,
            'pdb_residues': dict,
            'selected_residues': list of dicts (deduplicated residues with coords and metadata)
    """
    prank_dir = Path(prank_output_dir)
    pdb_path = Path(pdb_path)
    stem = pdb_path.stem
    fname = pdb_path.name

    # 1. Sequence & PDB Residues
    full_seq, pdb_residues, structure = get_full_sequence_from_pdb(pdb_path)
    ca_coords = extract_ca_coords_from_pdb(pdb_path)

    # 2. Locate P2Rank CSV files
    sub_dir = prank_dir / f"{stem}_prank_output"
    search_dirs = [sub_dir, prank_dir]

    pred_csv, res_csv = None, None
    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for pat in [f"{fname}_predictions.csv", f"{stem}.pdb_predictions.csv", f"{stem}_predictions.csv"]:
            cand = s_dir / pat
            if cand.exists() and cand.stat().st_size > 0:
                pred_csv = cand
                break
        for pat in [f"{fname}_residues.csv", f"{stem}.pdb_residues.csv", f"{stem}_residues.csv"]:
            cand = s_dir / pat
            if cand.exists() and cand.stat().st_size > 0:
                res_csv = cand
                break
        if pred_csv and res_csv:
            break

    if not pred_csv:
        cands = list(prank_dir.glob(f"{stem}*_predictions.csv"))
        if cands:
            pred_csv = cands[0]
    if not res_csv:
        cands = list(prank_dir.glob(f"{stem}*_residues.csv"))
        if cands:
            res_csv = cands[0]

    # Collect pocket metadata from predictions.csv
    valid_pocket_ids = set()
    pocket_info = {}
    if pred_csv and pred_csv.exists():
        with open(pred_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                name = clean_row.get('name', '')
                prob = float(clean_row.get('probability', clean_row.get('prob', 0.0)))
                rank = int(clean_row.get('rank', 1))
                m = re.search(r'(\d+)', name)
                p_id = int(m.group(1)) if m else rank

                if prob >= min_prob and len(valid_pocket_ids) < max_pockets:
                    valid_pocket_ids.add(p_id)
                    pocket_info[p_id] = {
                        'probability': prob,
                        'score': float(clean_row.get('score', 0.0)),
                        'rank': rank,
                        'center': [
                            float(clean_row.get('center_x', clean_row.get('x', 0.0))),
                            float(clean_row.get('center_y', clean_row.get('y', 0.0))),
                            float(clean_row.get('center_z', clean_row.get('z', 0.0)))
                        ]
                    }

    # If no pockets met threshold, take top 1
    if not valid_pocket_ids and pocket_info:
        valid_pocket_ids.add(1)

    # Collect residues from residues.csv
    residues_by_key = {}
    if res_csv and res_csv.exists():
        with open(res_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                p_idx = int(clean_row.get('pocket', 0))
                if p_idx in valid_pocket_ids:
                    c = clean_row.get('chain', 'A').strip()
                    r_num = clean_row.get('residue_label', '').strip()
                    resname = clean_row.get('residue_name', '').strip()
                    prob = float(clean_row.get('probability', 0.0))
                    score = float(clean_row.get('score', 0.0))

                    u_key = (c, r_num)
                    if u_key not in residues_by_key:
                        # Find seq_idx in parsed sequence
                        r_meta = pdb_residues.get(u_key, pdb_residues.get(r_num, {}))
                        seq_idx = r_meta.get('seq_idx', None)

                        # Find C-alpha coordinates
                        coord = ca_coords.get(u_key, ca_coords.get(r_num, np.array([0.0, 0.0, 0.0], dtype=np.float32)))

                        residues_by_key[u_key] = {
                            'chain': c,
                            'resseq': r_num,
                            'resname': resname,
                            'one_letter': THREE_TO_ONE.get(resname, 'X'),
                            'seq_idx': seq_idx,
                            'coords': coord,
                            'p2rank_score': score,
                            'p2rank_prob': prob,
                            'pockets': [p_idx]
                        }
                    else:
                        residues_by_key[u_key]['pockets'].append(p_idx)

    # Sort residues by sequence index
    selected_residues = [
        r for r in residues_by_key.values() if r['seq_idx'] is not None and 0 <= r['seq_idx'] < len(full_seq)
    ]
    selected_residues.sort(key=lambda r: r['seq_idx'])

    return {
        'full_sequence': full_seq,
        'selected_residues': selected_residues,
        'pockets_metadata': pocket_info
    }


class ResidueFeatureExtractor:
    """
    Extracts per-residue ESM-2 embeddings and 3D C-alpha coordinates for candidate pocket residues.
    """
    def __init__(self, esm_extractor=None, model_name=None, device=None):
        if esm_extractor is not None:
            self.esm = esm_extractor
        else:
            self.esm = ESMFeatureExtractor(
                model_name=model_name or "facebook/esm2_t33_650M_UR50D",
                device=device
            )

    def extract_from_pdb_and_p2rank(self, pdb_path, prank_dir, min_prob=0.10, max_pockets=5):
        """
        Extracts all residue features, coordinates, and metadata for a single structure.
        
        Returns:
            dict containing:
                'full_protein_feature': Tensor [1280]
                'residue_features': Tensor [M, 1280]
                'residue_coords': Tensor [M, 3]
                'residue_metadata': list of M dicts
                'full_sequence': str
        """
        parsed = parse_p2rank_residues_for_protein(
            prank_dir, pdb_path, min_prob=min_prob, max_pockets=max_pockets
        )
        full_seq = parsed['full_sequence']
        sel_residues = parsed['selected_residues']

        if not sel_residues:
            raise ValueError(f"No valid pocket residues found for {pdb_path} with min_prob >= {min_prob}")

        # Compute full sequence ESM-2 embeddings
        per_res = self.esm.extract_sequence_embeddings_raw(full_seq)  # [L, 1280]
        full_protein_feat = per_res.mean(dim=0).cpu()                 # [1280]

        # Extract residue slices
        seq_indices = [r['seq_idx'] for r in sel_residues]
        idx_tensor = torch.tensor(seq_indices, dtype=torch.long, device=per_res.device)
        residue_features = per_res.index_select(0, idx_tensor).cpu()  # [M, 1280]

        # Assemble coordinates [M, 3]
        coords_arr = np.stack([r['coords'] for r in sel_residues], axis=0)
        residue_coords = torch.from_numpy(coords_arr).float()

        return {
            'full_protein_feature': full_protein_feat,
            'residue_features': residue_features,
            'residue_coords': residue_coords,
            'residue_metadata': sel_residues,
            'full_sequence': full_seq
        }
