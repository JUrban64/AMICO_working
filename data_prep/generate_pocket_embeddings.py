import os
import sys
import csv
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

# Přidání kořenového adresáře AMICO do sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from p2rank_utils import parse_p2rank_output
from esm_extractor import ESMFeatureExtractor

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
NAME_TO_LABEL = {name: i for i, name in enumerate(TARGET_NAMES)}

def load_metadata_targets(metadata_path):
    """
    Načte dataset_metadata.tsv a vrátí mapování pdb_file -> metadata slovník.
    """
    targets = {}
    if not os.path.exists(metadata_path):
        return targets

    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None) # přeskočit hlavičku
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

    parser = argparse.ArgumentParser(description="Extrakce ESM-2 pocket embeddings ze samostatných P2Rank výstupů")
    parser.add_argument('--prank-dir', type=str, default=default_prank_dir, help='Složka s existujícími výstupy P2Ranku (*_predictions.csv)')
    parser.add_argument('--pdb-dir', type=str, default=default_pdb_dir, help='Složka se strukturami PDB (structures/all_pdbs)')
    parser.add_argument('--metadata', type=str, default=default_metadata, help='Cesta k dataset_metadata.tsv')
    parser.add_argument('--out-path', type=str, default=default_out, help='Výstupní soubor pro esm_dataset.pt')
    parser.add_argument('--min-prob', type=float, default=0.30, help='Minimální pravděpodobnost kapsy z P2Ranku (default: 0.30)')
    parser.add_argument('--esm-model', type=str, default='facebook/esm2_t33_650M_UR50D', help='Model ESM-2')
    parser.add_argument('--device', type=str, default=None, help='Zařízení pro ESM (cuda, mps, cpu)')
    parser.add_argument('--save-interval', type=int, default=100, help='Interval ukládání checkpointu')
    parser.add_argument('--limit', type=int, default=None, help='Testovací limit na počet proteinů')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)

    print("=" * 65)
    print("AMICO: GENERÁTOR POCKET EMBEDDINGS (Z HOTOVÉHO P2RANKU)")
    print("=" * 65)
    print(f"P2Rank výstupy:{args.prank_dir}")
    print(f"PDB složka:    {args.pdb_dir}")
    print(f"Metadata:      {args.metadata}")
    print(f"Výstupní soubor:{args.out_path}")
    print(f"Min. prob:     {args.min_prob}")
    print("=" * 65)

    # 1. Načtení metadat
    metadata_targets = load_metadata_targets(args.metadata)
    print(f"-> Načteno {len(metadata_targets)} validních struktur z metadat.")

    # 2. Načtení existujícího souboru kapes (Resume)
    existing_pockets = []
    processed_pids = set()
    if os.path.exists(args.out_path):
        print(f"-> Nalezen existující soubor kapes v {args.out_path}, načítám pro resume...")
        existing_pockets = torch.load(args.out_path, weights_only=False)
        for item in existing_pockets:
            raw_pid = item['protein_id']
            base_name = os.path.basename(raw_pid)
            pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
            processed_pids.add(pid)
        print(f"-> Již zpracováno: {len(processed_pids)} unikátních proteinů v {args.out_path}.")

    # 3. Indexace PDB souborů na disku
    pdb_dir_path = Path(args.pdb_dir)
    if not pdb_dir_path.exists():
        print(f"❌ Chyba: Složka s PDB strukturami {args.pdb_dir} neexistuje.")
        return

    all_pdb_files = [f for f in pdb_dir_path.glob("*.pdb") if "_pocket_" not in f.name]
    pdb_by_stem = {f.stem: f for f in all_pdb_files}

    # Sestavení fronty ke zpracování
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
        print(f"-> Testovací limit: Zpracovávám {len(queue)} struktur.")
    else:
        print(f"-> Zbývá ke zpracování: {len(queue)} struktur.")

    if not queue:
        print("✅ Všechny kapsy jsou již kompletně extrahovány.")
        return

    # 4. Inicializace ESM Feature Extractor
    print(f"\n-> Inicializuji ESM Feature Extractor ({args.esm_model})...")
    extractor = ESMFeatureExtractor(model_name=args.esm_model, device=args.device)

    new_pockets_list = list(existing_pockets)
    new_counter = 0
    missing_prank = 0

    pbar = tqdm(queue, desc="ESM-2 Extrakce kapes")
    for pdb_path, meta in pbar:
        stem = meta['stem']
        label = meta['label']

        try:
            parsed = parse_p2rank_output(args.prank_dir, pdb_path, min_prob=args.min_prob)
            pockets = parsed['pockets']

            pocket_seqs = [p['sequence'] for p in pockets if p.get('sequence')]
            if pocket_seqs:
                # Extrakce embeddingů pro všechny detekované kapsy daného proteinu
                pocket_embs = extractor.extract_pocket_embeddings(pocket_seqs) # [N, 1280]

                valid_idx = 0
                for p_info in pockets:
                    if not p_info.get('sequence'):
                        continue
                    p_feat = pocket_embs[valid_idx] # [1280]
                    valid_idx += 1

                    new_pockets_list.append({
                        'protein_id': f"{stem}_pocket_{p_info['pocket_id']}.pdb",
                        'features': p_feat,
                        'label': label,
                        'probability': p_info.get('probability', 0.0),
                        'score': p_info.get('score', 0.0),
                        'center': p_info.get('center', [0.0, 0.0, 0.0]),
                        'residue_count': p_info.get('residue_count', len(p_info.get('sequence', '')))
                    })

            new_counter += 1

            if new_counter % args.save_interval == 0:
                torch.save(new_pockets_list, args.out_path)

        except (FileNotFoundError, ValueError) as e:
            missing_prank += 1
        except Exception as e:
            print(f"\n❌ Chyba při extrakci kapes u {stem}: {e}")

    # Finální uložení
    torch.save(new_pockets_list, args.out_path)
    print("\n" + "=" * 65)
    print("✅ HOTOVO!")
    print(f"Celkem uloženo kapes do {args.out_path}: {len(new_pockets_list)}")
    if missing_prank > 0:
        print(f"⚠️ Upozornění: Pro {missing_prank} proteinů nebyl nalezen výstup P2Ranku v {args.prank_dir}.")
    print("=" * 65)

if __name__ == "__main__":
    main()
