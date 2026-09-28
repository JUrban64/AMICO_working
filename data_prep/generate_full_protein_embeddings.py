import os
import csv
import argparse
import torch
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser
import glob
import sys

# Přidání cesty pro import z kořenového adresáře AMICO
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from esm_extractor import ESMFeatureExtractor
except ImportError:
    from esm2_feature_ex import ESMFeatureExtractor

def is_aa(residue):
    return residue.get_id()[0] == ' '

def get_full_sequence_from_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
    except Exception as e:
        print(f"Chyba při parsování {pdb_path}: {e}")
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

    parser = argparse.ArgumentParser(description="Vygeneruje full protein embeddings z PDB struktur pomocí ESM-2")
    parser.add_argument('--pdb-dir', type=str, default=default_pdb_dir, help='Cesta ke složce se všemi PDB strukturami (např. structures/all_pdbs)')
    parser.add_argument('--metadata', type=str, default=default_metadata, help='Cesta k dataset_metadata.tsv (pokud existuje)')
    parser.add_argument('--dataset-path', type=str, default=os.path.join(base_dir, 'data_prep', 'esm_dataset.pt'), help='Cesta k esm_dataset.pt (volitelné)')
    parser.add_argument('--out-path', type=str, default=os.path.join(base_dir, 'data_prep', 'esm_full_proteins.pt'), help='Výstupní soubor pro embeddings')
    parser.add_argument('--model-name', type=str, default='facebook/esm2_t33_650M_UR50D', help='ESM-2 model z HuggingFace')
    parser.add_argument('--device', type=str, default=None, help='Zařízení (cuda, mps, cpu)')
    parser.add_argument('--save-interval', type=int, default=250, help='Interval průběžného ukládání')
    args = parser.parse_args()

    out_path = args.out_path
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # 1. Zjištění množiny proteinů ke zpracování
    unique_pids = set()
    metadata_file = find_metadata_tsv([args.metadata, os.path.join(args.pdb_dir, '..', 'dataset_metadata.tsv'), os.path.join(args.pdb_dir, 'dataset_metadata.tsv')])
    
    if metadata_file:
        print(f"-> Načítám seznam proteinů z metadat: {metadata_file}")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None) # přeskočit hlavičku
            for row in reader:
                if not row or len(row) < 2:
                    continue
                pdb_fname = row[1].strip()
                if pdb_fname and pdb_fname != 'NONE':
                    pid = pdb_fname.replace('.pdb', '').strip()
                    unique_pids.add(pid)
        print(f"-> Nalezeno {len(unique_pids)} proteinů v dataset_metadata.tsv.")
    elif os.path.exists(args.dataset_path):
        print(f"-> Načítám seznam proteinů z {args.dataset_path}...")
        raw_data = torch.load(args.dataset_path, weights_only=False)
        for item in raw_data:
            raw_pid = item['protein_id']
            base_name = os.path.basename(raw_pid)
            pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
            unique_pids.add(pid)
        print(f"-> Nalezeno {len(unique_pids)} unikátních proteinů v esm_dataset.pt.")
    else:
        print(f"-> Metadata ani esm_dataset.pt nenalezeny. Budou zpracovány všechny PDB soubory ve složce {args.pdb_dir}.")

    # 2. Načtení případného předchozího běhu (Resume)
    if os.path.exists(out_path):
        print(f"-> Nalezen předchozí běh v {out_path}, načítám pro resume...")
        full_embeddings_dict = torch.load(out_path, weights_only=False)
        print(f"-> Již zpracováno: {len(full_embeddings_dict)} záznamů.")
    else:
        full_embeddings_dict = {}

    # 3. Indexace souborů na disku
    print(f"-> Hledám PDB soubory ve složce {args.pdb_dir}...")
    all_pdb_files = glob.glob(os.path.join(args.pdb_dir, '**', '*.pdb'), recursive=True)
    print(f"-> Nalezeno PDB souborů celkem: {len(all_pdb_files)}")
    if len(all_pdb_files) == 0:
        print("❌ CHYBA: Zadaná složka neobsahuje žádné .pdb soubory nebo cesta neexistuje.")
        return

    pdb_map = {}
    for f in all_pdb_files:
        if '_pocket_' in f:
            continue
        basename = os.path.basename(f)
        pid = basename.replace('.pdb', '')
        pdb_map[pid] = f

    # Pokud jsme neměli metadata ani dataset_path, vezmeme všechny nalezené PDB
    if not unique_pids:
        unique_pids = set(pdb_map.keys())

    print(f"-> Celkem k ověření / zpracování: {len(unique_pids)} proteinů.")

    # 4. Inicializace ESM modelu
    extractor = ESMFeatureExtractor(model_name=args.model_name, device=args.device)

    missing_pdbs = 0
    newly_processed = 0
    pbar = tqdm(sorted(list(unique_pids)), desc="Extrakce ESM proteinových embeddingů")

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
            print(f"Varování: Ze souboru {pdb_path} se nepodařilo extrahovat sekvenci.")
            continue

        try:
            # Extrakce globálního embeddingu celého proteinu (Mean-Pooling přes rezidua -> [1280])
            mean_pooled_emb = extractor.extract_sequence_embedding(seq) # torch.FloatTensor [1280]
            full_embeddings_dict[pid] = mean_pooled_emb
            
            # Pokud se jedná o fragment (např. P12345_F1), uložíme i základní ID pokud ještě není
            clean_pid = pid.split('_')[0]
            if clean_pid not in full_embeddings_dict:
                full_embeddings_dict[clean_pid] = mean_pooled_emb
                
            newly_processed += 1
        except Exception as e:
            print(f"Chyba při extrakci ESM pro {pid}: {e}")

        # Průběžné ukládání pro ochranu proti přerušení
        if newly_processed > 0 and newly_processed % args.save_interval == 0:
            torch.save(full_embeddings_dict, out_path)

    torch.save(full_embeddings_dict, out_path)
    print("\n" + "=" * 50)
    print(f"✅ Hotovo! Celkem uloženo {len(full_embeddings_dict)} proteinových embeddingů do {out_path}")
    if missing_pdbs > 0:
        print(f"⚠️ Upozornění: Pro {missing_pdbs} ID nebyl nalezen PDB soubor na disku.")
    print("=" * 50)

if __name__ == "__main__":
    main()
