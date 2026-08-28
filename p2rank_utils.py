import os
import csv
import re
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
from Bio.PDB import PDBParser

THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def is_aa(residue):
    """Ověří, zda je reziduum standardní aminokyselina."""
    return residue.get_id()[0] == ' '

def find_p2rank_executable(custom_path=None):
    """
    Vyhledá spustitelný soubor P2Rank (prank).
    Kontroluje zadanou cestu, systémovou proměnnou PATH a běžné relativní cesty.
    """
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))

    # Standardní lokace v projektu a PATH
    candidates.extend([
        Path("p2rank_2.5.1/prank"),
        Path("../p2rank_2.5.1/prank"),
        Path("data_prep/p2rank_2.5.1/prank"),
        Path("../data_prep/p2rank_2.5.1/prank"),
        Path("p2rank/prank"),
        Path("../p2rank/prank")
    ])

    for cand in candidates:
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand.resolve())

    which_prank = shutil.which("prank")
    if which_prank:
        return which_prank
    
    which_p2rank = shutil.which("p2rank")
    if which_p2rank:
        return which_p2rank

    return custom_path if custom_path else "p2rank_2.5.1/prank"

def run_p2rank(pdb_path, prank_exec=None, output_dir=None, config="alphafold"):
    """
    Spustí P2Rank na zadaném PDB souboru a vrátí cestu ke složce s výstupy.
    
    Args:
        pdb_path: Cesta k PDB souboru.
        prank_exec: Cesta k binárce prank (pokud None, zkusí automatickou detekci).
        output_dir: Složka pro uložení výsledků (výchozí: ./temp_p2rank/<pdb_name>_prank_output).
        config: Konfigurace P2Ranku (např. 'alphafold' nebo 'default').
        
    Returns:
        Path: Cesta ke složce s výstupy P2Ranku.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB soubor nebyl nalezen: {pdb_path}")

    executable = find_p2rank_executable(prank_exec)

    if output_dir is None:
        output_dir = Path("./temp_p2rank") / f"{pdb_path.stem}_prank_output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        executable, "predict",
        "-c", config,
        "-f", str(pdb_path.resolve()),
        "-o", str(output_dir.resolve()),
        "-visualizations", "0"
    ]

    print(f"-> Spouštím P2Rank na {pdb_path.name}...")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Chyba při běhu P2Ranku:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        ) from e
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Spustitelný soubor P2Rank nebyl nalezen na '{executable}'. "
            f"Zadejte prosím správnou cestu pomocí parametru --prank."
        )

    return output_dir

def get_full_sequence_from_pdb(pdb_path):
    """
    Extrahuje kompletní aminokyselinovou sekvenci a rezidua z PDB souboru.
    
    Returns:
        tuple: (seq_str, pdb_residues_dict, structure)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(pdb_path))
    
    sequence = []
    pdb_residues = {} # (chain_id, resseq) -> residue
    
    for model in structure:
        for chain in model:
            chain_id = chain.get_id().strip()
            for residue in chain:
                if is_aa(residue):
                    resname = residue.get_resname().strip()
                    resseq = str(residue.get_id()[1]).strip()
                    one_letter = THREE_TO_ONE.get(resname, 'X')
                    sequence.append(one_letter)
                    pdb_residues[(chain_id, resseq)] = {
                        'residue': residue,
                        'resname': resname,
                        'one_letter': one_letter,
                        'chain_id': chain_id,
                        'resseq': resseq
                    }

    seq_str = ''.join(sequence)
    return seq_str, pdb_residues, structure

def parse_p2rank_output(prank_output_dir, pdb_path, min_prob=0.0):
    """
    Načte výsledky P2Ranku (_predictions.csv a _residues.csv) pro daný protein.
    
    Args:
        prank_output_dir: Složka s výstupy P2Ranku.
        pdb_path: Cesta k původnímu PDB souboru pro extrakci sekvencí a souřadnic.
        min_prob: Minimální pravděpodobnost kapsy z P2Ranku (0.0 = vrátit všechny).
        
    Returns:
        dict: {
            'full_sequence': str,
            'pockets': list of dicts: [
                {
                    'pocket_id': int,
                    'name': str,
                    'probability': float,
                    'center': [x, y, z],
                    'score': float,
                    'sequence': str,
                    'residue_count': int
                }, ...
            ]
        }
    """
    prank_dir = Path(prank_output_dir)
    pdb_path = Path(pdb_path)
    
    full_seq, pdb_residues, structure = get_full_sequence_from_pdb(pdb_path)
    if not full_seq:
        raise ValueError(f"Z {pdb_path} se nepodařilo extrahovat žádnou aminokyselinovou sekvenci.")

    # 1. Hledání _predictions.csv
    pred_csv_candidates = list(prank_dir.glob("*_predictions.csv"))
    if not pred_csv_candidates:
        # Zkusíme také přímo v prank_dir
        pred_csv_candidates = list(prank_dir.glob("*.csv"))

    pred_csv = None
    for cand in pred_csv_candidates:
        if "_predictions" in cand.name:
            pred_csv = cand
            break
    if not pred_csv and pred_csv_candidates:
        pred_csv = pred_csv_candidates[0]

    # 2. Hledání _residues.csv
    res_csv_candidates = list(prank_dir.glob("*_residues.csv"))
    res_csv = res_csv_candidates[0] if res_csv_candidates else None

    pockets_dict = {}

    if pred_csv and pred_csv.exists():
        with open(pred_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                name = clean_row.get('name', '')
                prob = float(clean_row.get('probability', clean_row.get('prob', 0.0)))
                score = float(clean_row.get('score', 0.0))
                rank = int(clean_row.get('rank', 1))

                # Extrakce čísla kapsy
                m = re.search(r'(\d+)', name)
                pocket_id = int(m.group(1)) if m else rank

                # Extrakce středu kapsy (center_x, center_y, center_z)
                cx = float(clean_row.get('center_x', clean_row.get('x', 0.0)))
                cy = float(clean_row.get('center_y', clean_row.get('y', 0.0)))
                cz = float(clean_row.get('center_z', clean_row.get('z', 0.0)))

                if prob >= min_prob:
                    pockets_dict[pocket_id] = {
                        'pocket_id': pocket_id,
                        'name': name,
                        'probability': prob,
                        'score': score,
                        'center': [cx, cy, cz],
                        'residues': [],
                        'sequence': ''
                    }

    # Načtení reziduí z _residues.csv
    if res_csv and res_csv.exists():
        with open(res_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                chain_id = clean_row.get('chain', clean_row.get('chain_id', '')).strip()
                resseq = clean_row.get('resseq', clean_row.get('residue_number', '')).strip()
                pname = clean_row.get('pocket', clean_row.get('pocket_name', '')).strip()

                m = re.search(r'(\d+)', pname)
                if m:
                    pid = int(m.group(1))
                    if pid in pockets_dict:
                        key = (chain_id, resseq)
                        if key in pdb_residues:
                            pockets_dict[pid]['residues'].append(pdb_residues[key])

    # Fallback: Pokud nebyly nalezeny kapsy v CSV, zkusíme hledat fyzické *_pocket_*.pdb soubory
    if not pockets_dict:
        pocket_pdbs = sorted(list(prank_dir.glob("*_pocket_*.pdb")))
        parser = PDBParser(QUIET=True)
        for idx, p_pdb in enumerate(pocket_pdbs, start=1):
            p_struct = parser.get_structure(f'pocket_{idx}', str(p_pdb))
            p_seq = []
            p_coords = []
            for r in p_struct.get_residues():
                if is_aa(r):
                    resname = r.get_resname().strip()
                    p_seq.append(THREE_TO_ONE.get(resname, 'X'))
                    if 'CA' in r:
                        p_coords.append(r['CA'].get_coord())
            
            center = np.mean(p_coords, axis=0).tolist() if p_coords else [0.0, 0.0, 0.0]
            pockets_dict[idx] = {
                'pocket_id': idx,
                'name': f"pocket{idx}",
                'probability': 1.0,
                'score': 1.0,
                'center': center,
                'residues': [],
                'sequence': ''.join(p_seq)
            }

    # Sestavení sekvencí pro jednotlivé kapsy
    pocket_list = []
    for pid in sorted(pockets_dict.keys()):
        p_data = pockets_dict[pid]
        if not p_data['sequence'] and p_data['residues']:
            p_data['sequence'] = ''.join([r['one_letter'] for r in p_data['residues']])
        
        p_data['residue_count'] = len(p_data['sequence'])
        if p_data['residue_count'] > 0:
            pocket_list.append(p_data)

    return {
        'full_sequence': full_seq,
        'pockets': pocket_list
    }
