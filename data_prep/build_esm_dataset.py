import os
import sys
import csv
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

# Přidání kořenového adresáře do sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from p2rank_utils import run_p2rank, run_p2rank_batch, parse_p2rank_output, find_p2rank_executable
from esm_extractor import ESMFeatureExtractor

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
NAME_TO_LABEL = {name: i for i, name in enumerate(TARGET_NAMES)}

def load_metadata_targets(metadata_path):
    """
    Načte dataset_metadata.tsv a vrátí mapování pdb_filename -> {uniprot_id, cofactors, label, ec, length, source}.
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

    parser = argparse.ArgumentParser(description="Zpracování PDB struktur (P2Rank + ESM-2) do esm_dataset.pt a esm_full_proteins.pt")
    parser.add_argument('--pdb-dir', type=str, default=default_pdb_dir, help='Cesta ke složce se všemi PDB strukturami (structures/all_pdbs)')
    parser.add_argument('--metadata', type=str, default=default_metadata, help='Cesta k dataset_metadata.tsv')
    parser.add_argument('--pockets-out', type=str, default=default_pockets_out, help='Výstupní soubor pro pocket dataset (.pt)')
    parser.add_argument('--full-proteins-out', type=str, default=default_full_out, help='Výstupní soubor pro full protein embeddings (.pt)')
    parser.add_argument('--prank-out-dir', type=str, default=default_prank_dir, help='Složka pro ukládání výstupů P2Ranku')
    parser.add_argument('--prank-exec', type=str, default=None, help='Vlastní cesta ke spustitelnému souboru prank')
    parser.add_argument('--config', type=str, default='alphafold', help='P2Rank konfigurace (alphafold / default)')
    parser.add_argument('--threads', type=int, default=8, help='Počet CPU vláken pro P2Rank')
    parser.add_argument('--chunk-size', type=int, default=500, help='Velikost dávky pro dávkový běh P2Ranku')
    parser.add_argument('--esm-model', type=str, default='facebook/esm2_t33_650M_UR50D', help='Model ESM-2')
    parser.add_argument('--device', type=str, default=None, help='Zařízení pro ESM (cuda, mps, cpu)')
    parser.add_argument('--min-prob', type=float, default=0.0, help='Minimální pravděpodobnost kapsy z P2Ranku')
    parser.add_argument('--save-interval', type=int, default=100, help='Interval ukládání checkpointu')
    parser.add_argument('--limit', type=int, default=None, help='Omezit počet zpracovaných proteinů pro testování')
    args = parser.parse_args()

    os.makedirs(args.prank_out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.pockets_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.full_proteins_out)), exist_ok=True)

    print("=" * 65)
    print("AMICO: DATOVÁ PIPELINE (P2Rank + ESM-2)")
    print("=" * 65)
    print(f"PDB složka:    {args.pdb_dir}")
    print(f"Metadata:      {args.metadata}")
    print(f"Pockets výstup:{args.pockets_out}")
    print(f"Full výstup:   {args.full_proteins_out}")
    print(f"P2Rank vláken: {args.threads} | Config: {args.config} | Chunk: {args.chunk_size}")
    print("=" * 65)

    # 1. Načtení metadat
    metadata_targets = load_metadata_targets(args.metadata)
    print(f"-> Načteno {len(metadata_targets)} validních struktur z metadat.")

    # 2. Načtení existujících výsledků (Resume)
    existing_pockets = []
    processed_pids = set()
    if os.path.exists(args.pockets_out):
        print(f"-> Nalezen existující pocket dataset v {args.pockets_out}, načítám...")
        existing_pockets = torch.load(args.pockets_out, weights_only=False)
        for item in existing_pockets:
            raw_pid = item['protein_id']
            base_name = os.path.basename(raw_pid)
            pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
            processed_pids.add(pid)
        print(f"-> Již zpracováno: {len(processed_pids)} proteinů v esm_dataset.pt.")

    full_proteins_dict = {}
    if os.path.exists(args.full_proteins_out):
        print(f"-> Nalezeny existující full embeddings v {args.full_proteins_out}, načítám...")
        full_proteins_dict = torch.load(args.full_proteins_out, weights_only=False)
        print(f"-> Již zpracováno: {len(full_proteins_dict)} proteinů v esm_full_proteins.pt.")

    # 3. Nalezení PDB souborů na disku
    pdb_dir_path = Path(args.pdb_dir)
    if not pdb_dir_path.exists():
        print(f"❌ Chyba: Složka {args.pdb_dir} neexistuje.")
        return

    all_pdb_files = [f for f in pdb_dir_path.glob("*.pdb") if "_pocket_" not in f.name]
    pdb_by_stem = {f.stem: f for f in all_pdb_files}
    print(f"-> Nalezeno {len(all_pdb_files)} PDB struktur na disku.")

    # Sestavení fronty ke zpracování
    queue = []
    if metadata_targets:
        for pdb_fname, meta in metadata_targets.items():
            stem = meta['stem']
            if stem in processed_pids and stem in full_proteins_dict:
                continue
            if stem in pdb_by_stem:
                queue.append((pdb_by_stem[stem], meta))
    else:
        for f in all_pdb_files:
            stem = f.stem
            if stem in processed_pids and stem in full_proteins_dict:
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
        print(f"-> Testovací limit: Zpracovávám {len(queue)} struktur.")
    else:
        print(f"-> Zbývá ke zpracování: {len(queue)} struktur.")

    if not queue:
        print("✅ Všechny struktury jsou již kompletně zpracovány.")
        return

    # Ověření dostupnosti P2Ranku
    prank_bin = find_p2rank_executable(args.prank_exec)
    print(f"-> Používám P2Rank binárku: {prank_bin}")

    # 4. Spuštění P2Rank v dávkovém režimu pro všechny nezpracované struktury
    queue_pdb_paths = [pdb_path for pdb_path, _ in queue]
    print(f"\n🚀 Spouštím hromadnou predikci kapes P2Rank pro {len(queue_pdb_paths)} struktur...")
    run_p2rank_batch(
        queue_pdb_paths,
        prank_exec=prank_bin,
        output_dir=args.prank_out_dir,
        config=args.config,
        threads=args.threads,
        chunk_size=args.chunk_size
    )

    # 5. Inicializace ESM extraktoru pro výpočet embeddingů
    print("\n-> Inicializuji ESM Feature Extractor...")
    extractor = ESMFeatureExtractor(model_name=args.esm_model, device=args.device)

    new_pockets_list = list(existing_pockets)
    new_counter = 0

    pbar = tqdm(queue, desc="ESM-2 Extrakce kapes & proteinů")
    for pdb_path, meta in pbar:
        stem = meta['stem']
        label = meta['label']

        try:
            # Parsování kapes z výstupů P2Ranku
            parsed = parse_p2rank_output(args.prank_out_dir, pdb_path, min_prob=args.min_prob)
            full_seq = parsed['full_sequence']
            pockets = parsed['pockets']

            # Extrakce full protein embeddingu
            if stem not in full_proteins_dict:
                full_emb = extractor.extract_sequence_embedding(full_seq) # [1280]
                full_proteins_dict[stem] = full_emb
                clean_acc = stem.split('_')[0]
                if clean_acc not in full_proteins_dict:
                    full_proteins_dict[clean_acc] = full_emb

            # D. Extrakce pocket embeddingů
            pocket_seqs = [p['sequence'] for p in pockets if p.get('sequence')]
            if pocket_seqs:
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

            # Průběžné ukládání
            if new_counter % args.save_interval == 0:
                torch.save(new_pockets_list, args.pockets_out)
                torch.save(full_proteins_dict, args.full_proteins_out)

        except Exception as e:
            print(f"\n❌ Chyba při zpracování {stem}: {e}")

    # Finální uložení
    torch.save(new_pockets_list, args.pockets_out)
    torch.save(full_proteins_dict, args.full_proteins_out)

    print("\n" + "=" * 65)
    print("✅ HOTOVO!")
    print(f"Uloženo kapes do {args.pockets_out}: {len(new_pockets_list)}")
    print(f"Uloženo full proteinů do {args.full_proteins_out}: {len(full_proteins_dict)}")
    print("=" * 65)

if __name__ == "__main__":
    main()
