import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser
import glob

# Přidání cesty pro import
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from esm2_feature_ex import ESMFeatureExtractor

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
    # Procházíme atomy a extrahujeme celou sekvenci (pro všechny chainy)
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

def main():
    parser = argparse.ArgumentParser(description="Vygeneruje full protein embeddings z původních PDB souborů")
    parser.add_argument('--pdb-dir', type=str, required=True, help='Cesta ke složce se všemi PDB strukturami')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'data_prep', 'esm_dataset.pt')
    out_path = os.path.join(base_dir, 'data_prep', 'esm_full_proteins.pt')
    
    print(f"Loading pockets dataset from {dataset_path} to find unique proteins...")
    raw_data = torch.load(dataset_path, weights_only=False)
    
    unique_pids = set()
    for item in raw_data:
        raw_pid = item['protein_id']
        base_name = os.path.basename(raw_pid)
        pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
        unique_pids.add(pid)
        
    print(f"Found {len(unique_pids)} unique proteins.")
    
    # Zkusíme načíst dosud zpracované, abychom mohli navázat (resume)
    if os.path.exists(out_path):
        print(f"Nalezen předchozí běh v {out_path}, načítám pro případný resume...")
        full_embeddings_dict = torch.load(out_path, weights_only=False)
    else:
        full_embeddings_dict = {}

    extractor = ESMFeatureExtractor()
    
    # Připravíme si mapování z ID na cestu k PDB souboru
    print(f"Hledám PDB soubory ve složce {args.pdb_dir}...")
    all_pdb_files = glob.glob(os.path.join(args.pdb_dir, '**', '*.pdb'), recursive=True)
    
    print(f"Nalezeno PDB souborů celkem: {len(all_pdb_files)}")
    if len(all_pdb_files) == 0:
        print("CHYBA: Zadaná složka neobsahuje žádné .pdb soubory nebo cesta neexistuje.")
        return
        
    pdb_map = {}
    for f in all_pdb_files:
        if '_pocket_' in f:  # Přeskočíme kapsy, chceme původní plné proteiny
            continue
        basename = os.path.basename(f)
        pid = basename.replace('.pdb', '')
        # Občas je ve jméně protein_id ještě navíc '_out' atd, budeme hledat přesnou shodu:
        pdb_map[pid] = f
        
    # DEBUG ukázky pro snazší pochopení problému s cestami/jmény:
    print("\n--- DEBUG UKÁZKY NÁZVŮ ---")
    print("Ukázka 3 ID z datasetu kapes (unique_pids):", list(unique_pids)[:3])
    print("Ukázka 3 ID nalezených ve složce (pdb_map):", list(pdb_map.keys())[:3])
    print("--------------------------\n")
        
    # Pokud některé proteiny z datasetu mají jiný název (např. P27352_MERGED), pokusíme se je spárovat
    
    missing_pdbs = 0
    for pid in tqdm(unique_pids, desc="Processing proteins"):
        if pid in full_embeddings_dict:
            continue
            
        # Hledání správného PDB souboru
        pdb_path = None
        if pid in pdb_map:
            pdb_path = pdb_map[pid]
        else:
            # Zkusme odstranit _MERGED apod.
            clean_pid = pid.split('_')[0]
            if clean_pid in pdb_map:
                pdb_path = pdb_map[clean_pid]
            else:
                # Zkusme fuzzy match (např. hledat P27352 kdekoliv v názvu)
                matches = [f for f in all_pdb_files if clean_pid in os.path.basename(f) and '_pocket_' not in f]
                if matches:
                    pdb_path = matches[0]

        if not pdb_path:
            missing_pdbs += 1
            # print(f"PDB file not found for {pid}")
            continue
            
        seq = get_full_sequence_from_pdb(pdb_path)
        if not seq:
            print(f"No valid sequence extracted from {pdb_path}")
            continue
            
        try:
            # Extrakce ESM [L, 1280]
            emb = extractor.extract_embeddings(seq)
            
            # Agregace (Mean Pooling) přes celou sekvenci -> získáme 1 vektor [1280] pro celý protein
            mean_pooled_emb = np.mean(emb, axis=0)
            
            # Uložení tenzoru
            full_embeddings_dict[pid] = torch.FloatTensor(mean_pooled_emb)
        except Exception as e:
            print(f"ESM Extraction failed for {pid}: {e}")
            
        # Průběžné ukládání (každých 500)
        if len(full_embeddings_dict) % 500 == 0:
            torch.save(full_embeddings_dict, out_path)
            
    print(f"Successfully processed {len(full_embeddings_dict)} proteins.")
    if missing_pdbs > 0:
        print(f"Upozornění: Pro {missing_pdbs} proteinů nebyl nalezen původní PDB soubor.")
        
    torch.save(full_embeddings_dict, out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
