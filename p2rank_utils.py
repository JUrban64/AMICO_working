import os
import csv
import re
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
try:
    from Bio.PDB import PDBParser
except ImportError:
    PDBParser = None

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

    if output_dir is None:
        output_dir = Path("./temp_p2rank") / f"{pdb_path.stem}_prank_output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    existing_preds = list(output_dir.glob("*_predictions.csv")) + list(output_dir.glob("*.csv"))
    if existing_preds:
        print(f"-> Nalezeny existující výstupy P2Ranku v {output_dir} (přeskakuji opakovaný běh).")
        return output_dir

    executable = find_p2rank_executable(prank_exec)

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

def run_p2rank_batch(
    pdb_paths,
    prank_exec=None,
    output_dir=None,
    config="alphafold",
    threads=8,
    chunk_size=500
):
    """
    Spustí P2Rank ve vysoce efektivním dávkovém režimu (-l dataset.ds) pro trénovací sadu.
    
    Výhody pro trénování (desítky tisíc struktur):
    1. Spustí JVM a načte model náhodných lesů pouze JEDNOU (50-100x rychlejší než volání po jednom souboru).
    2. Využívá multithreading P2Ranku přes zadaný počet jader CPU (-threads).
    3. Automaticky přeskakuje již zpracované struktury (ochrana proti přerušení / resume).
    4. Rozděluje seznam do chunků (např. po 500 souborech) s průběžným ukládáním.
    
    Args:
        pdb_paths: Seznam cest (str nebo Path) k PDB souborům.
        prank_exec: Cesta k binárce prank (pokud None, zkusí automatickou detekci).
        output_dir: Složka pro uložení výsledků (výchozí: ./structures/p2rank_outputs).
        config: Konfigurace P2Ranku ('alphafold' nebo 'default').
        threads: Počet vláken pro P2Rank.
        chunk_size: Velikost dávky (počet struktur na jeden běh P2Rank procesu).
        
    Returns:
        Path: Cesta ke složce s výstupy.
    """
    if output_dir is None:
        output_dir = Path("./structures/p2rank_outputs")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    executable = find_p2rank_executable(prank_exec)

    # 1. Filtrování již zpracovaných PDB struktur (Resume podpora)
    to_process = []
    skipped_count = 0
    for p in pdb_paths:
        p = Path(p)
        stem = p.stem
        fname = p.name
        
        pred_candidates = [
            output_dir / f"{fname}_predictions.csv",
            output_dir / f"{stem}.pdb_predictions.csv",
            output_dir / f"{stem}_predictions.csv",
            output_dir / f"{stem}_prank_output" / f"{fname}_predictions.csv",
            output_dir / f"{stem}_prank_output" / f"{stem}.pdb_predictions.csv",
            output_dir / f"{stem}_prank_output" / f"{stem}_predictions.csv"
        ]
        
        already_done = any(c.exists() and c.stat().st_size > 0 for c in pred_candidates)
        if already_done:
            skipped_count += 1
        else:
            to_process.append(p.resolve())

    print(f"-> P2Rank Dávkový režim: Celkem {len(pdb_paths)} struktur.")
    if skipped_count > 0:
        print(f"   ⚡ Přeskočeno (již dříve predikováno): {skipped_count} struktur.")
    print(f"   🚀 Zbývá predikovat: {len(to_process)} struktur (vláken: {threads}, chunk size: {chunk_size}).")

    if not to_process:
        print("-> Všechny struktury jsou již kompletně zpracovány.")
        return output_dir

    # 2. Zpracování v dávkách (chuncích)
    total_chunks = (len(to_process) + chunk_size - 1) // chunk_size
    ds_temp_dir = output_dir / "temp_datasets"
    ds_temp_dir.mkdir(parents=True, exist_ok=True)

    for chunk_idx in range(total_chunks):
        chunk_files = to_process[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        ds_file = ds_temp_dir / f"batch_chunk_{chunk_idx + 1:04d}.ds"
        
        with open(ds_file, "w", encoding="utf-8") as f:
            for c_path in chunk_files:
                f.write(f"{str(c_path)}\n")

        print(f"\n[Dávka {chunk_idx + 1}/{total_chunks}] Spouštím P2Rank na {len(chunk_files)} strukturách...")
        cmd = [
            executable, "predict",
            str(ds_file),
            "-o", str(output_dir.resolve()),
            "-c", config,
            "-threads", str(threads),
            "-visualizations", "0"
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Chyba při běhu dávky #{chunk_idx + 1}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Spustitelný soubor P2Rank nebyl nalezen na '{executable}'.")
        finally:
            if ds_file.exists():
                try:
                    ds_file.unlink()
                except OSError:
                    pass

    if ds_temp_dir.exists():
        try:
            shutil.rmtree(ds_temp_dir)
        except OSError:
            pass

    print(f"\n✅ Dávková predikce kapes v P2Rank dokončena. Výsledky uloženy v {output_dir}")
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
    residues_list = []
    
    for model in structure:
        for chain in model:
            chain_id = chain.get_id().strip()
            for residue in chain:
                if is_aa(residue):
                    resname = residue.get_resname().strip()
                    resseq = str(residue.get_id()[1]).strip()
                    one_letter = THREE_TO_ONE.get(resname, 'X')
                    seq_idx = len(sequence)
                    sequence.append(one_letter)
                    r_info = {
                        'residue': residue,
                        'resname': resname,
                        'one_letter': one_letter,
                        'chain_id': chain_id,
                        'resseq': resseq,
                        'seq_idx': seq_idx
                    }
                    residues_list.append(r_info)
                    pdb_residues[(chain_id, resseq)] = r_info
                    if resseq not in pdb_residues:
                        pdb_residues[resseq] = r_info
                    if ('', resseq) not in pdb_residues:
                        pdb_residues[('', resseq)] = r_info
                    if ('A', resseq) not in pdb_residues:
                        pdb_residues[('A', resseq)] = r_info

    pdb_residues['_all_residues_list'] = residues_list
    seq_str = ''.join(sequence)
    return seq_str, pdb_residues, structure

def parse_p2rank_output(prank_output_dir, pdb_path, min_prob=0.0):
    """
    Načte výsledky P2Ranku (_predictions.csv a _residues.csv) pro daný protein.
    Podporuje jak samostatné složky (single inference), tak sdílenou dávkovou složku (batch training).
    """
    prank_dir = Path(prank_output_dir)
    pdb_path = Path(pdb_path)
    
    full_seq, pdb_residues, structure = get_full_sequence_from_pdb(pdb_path)
    if not full_seq:
        raise ValueError(f"Z {pdb_path} se nepodařilo extrahovat žádnou aminokyselinovou sekvenci.")
    
    all_residues = pdb_residues.get('_all_residues_list', [])

    stem = pdb_path.stem
    fname = pdb_path.name

    # Možné složky pro hledání (sdílená složka i podsložka stem_prank_output)
    sub_dir = prank_dir / f"{stem}_prank_output"
    search_dirs = [prank_dir]
    if sub_dir.exists():
        search_dirs.insert(0, sub_dir)

    # 1. Hledání _predictions.csv specifického pro daný protein
    pred_csv = None
    pred_search_patterns = [
        f"{fname}_predictions.csv",
        f"{stem}.pdb_predictions.csv",
        f"{stem}_predictions.csv",
        f"{fname}.predictions.csv",
        f"{stem}.predictions.csv"
    ]
    for s_dir in search_dirs:
        for pat in pred_search_patterns:
            cand = s_dir / pat
            if cand.exists() and cand.stat().st_size > 0:
                pred_csv = cand
                break
        if pred_csv:
            break

    # Fallback na glob pouze pokud jsme nenašli podle přesného jména
    if not pred_csv:
        pred_candidates = list(prank_dir.glob(f"{stem}*_predictions.csv")) + list(prank_dir.glob(f"{stem}*.csv"))
        if not pred_candidates:
            pred_candidates = list(prank_dir.glob("*_predictions.csv"))
        if pred_candidates:
            pred_csv = pred_candidates[0]

    # 2. Hledání _residues.csv specifického pro daný protein
    res_csv = None
    res_search_patterns = [
        f"{fname}_residues.csv",
        f"{stem}.pdb_residues.csv",
        f"{stem}_residues.csv",
        f"{fname}.residues.csv",
        f"{stem}.residues.csv"
    ]
    for s_dir in search_dirs:
        for pat in res_search_patterns:
            cand = s_dir / pat
            if cand.exists() and cand.stat().st_size > 0:
                res_csv = cand
                break
        if res_csv:
            break

    if not res_csv:
        res_candidates = list(prank_dir.glob(f"{stem}*_residues.csv"))
        if not res_candidates:
            res_candidates = list(prank_dir.glob("*_residues.csv"))
        if res_candidates:
            res_csv = res_candidates[0]

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
                    pocket_entry = {
                        'pocket_id': pocket_id,
                        'name': name,
                        'probability': prob,
                        'score': score,
                        'center': [cx, cy, cz],
                        'residues': [],
                        'sequence': ''
                    }

                    # Extrakce reziduí přímo ze sloupce 'residue_ids' v _predictions.csv
                    residue_ids_str = clean_row.get('residue_ids', '')
                    if residue_ids_str:
                        seen_res = set()
                        for token in residue_ids_str.split():
                            token = token.strip()
                            if not token:
                                continue
                            if '_' in token:
                                c, r_num = token.split('_', 1)
                            else:
                                c, r_num = '', token
                            
                            r_info = None
                            for test_key in [(c, r_num), (c.strip(), r_num.strip()), r_num.strip(), ('', r_num.strip()), ('A', r_num.strip())]:
                                if test_key in pdb_residues:
                                    r_info = pdb_residues[test_key]
                                    break
                            
                            if r_info is not None:
                                u_key = (r_info['chain_id'], r_info['resseq'])
                                if u_key not in seen_res:
                                    seen_res.add(u_key)
                                    pocket_entry['residues'].append(r_info)

                    pockets_dict[pocket_id] = pocket_entry

    # 3. Načtení / doplnění reziduí z _residues.csv
    if res_csv and res_csv.exists():
        with open(res_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                chain_id = clean_row.get('chain', clean_row.get('chain_id', '')).strip()
                # P2Rank používá název sloupce 'residue_label'
                resseq = clean_row.get('residue_label', clean_row.get('resseq', clean_row.get('residue_number', ''))).strip()
                pname = clean_row.get('pocket', clean_row.get('pocket_name', '')).strip()

                m = re.search(r'(\d+)', pname)
                if m:
                    pid = int(m.group(1))
                    if pid > 0 and pid in pockets_dict:
                        r_info = None
                        for test_key in [(chain_id, resseq), resseq, ('', resseq), ('A', resseq)]:
                            if test_key in pdb_residues:
                                r_info = pdb_residues[test_key]
                                break
                        if r_info is not None:
                            existing = {(r['chain_id'], r['resseq']) for r in pockets_dict[pid]['residues']}
                            if (r_info['chain_id'], r_info['resseq']) not in existing:
                                pockets_dict[pid]['residues'].append(r_info)

    # 4. Fallback: Pokud nebyly nalezeny kapsy v CSV, zkusíme hledat fyzické *_pocket_*.pdb soubory
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

    # 5. Sestavení sekvencí pro jednotlivé kapsy
    pocket_list = []
    for pid in sorted(pockets_dict.keys()):
        p_data = pockets_dict[pid]
        
        # Seřazení reziduí podle pořadí v primární sekvenci proteinu (N -> C terminus)
        if p_data['residues']:
            p_data['residues'].sort(key=lambda r: (r.get('chain_id', ''), r.get('seq_idx', 0)))
            p_data['sequence'] = ''.join([r['one_letter'] for r in p_data['residues']])
        
        # Prostorový fallback: Pokud je sekvence stále prázdná, najdeme rezidua do 8.5 Å od středu
        if not p_data['sequence'] and p_data.get('center') and p_data['center'] != [0.0, 0.0, 0.0] and all_residues:
            cx, cy, cz = p_data['center']
            center_arr = np.array([cx, cy, cz])
            nearby = []
            for r_info in all_residues:
                r_obj = r_info['residue']
                min_d = float('inf')
                for atom in r_obj:
                    d = float(np.linalg.norm(atom.get_coord() - center_arr))
                    if d < min_d:
                        min_d = d
                if min_d <= 8.5:
                    nearby.append(r_info)
            if nearby:
                nearby.sort(key=lambda r: (r.get('chain_id', ''), r.get('seq_idx', 0)))
                p_data['residues'] = nearby
                p_data['sequence'] = ''.join([r['one_letter'] for r in nearby])

        p_data['residue_count'] = len(p_data['sequence'])
        if p_data['residue_count'] > 0:
            pocket_list.append(p_data)

    return {
        'full_sequence': full_seq,
        'pockets': pocket_list
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="P2Rank Pocket Prediction CLI (Single & Batch pro trénovací data)")
    parser.add_argument("--pdb", type=str, default=None, help="Cesta k jednomu PDB souboru")
    parser.add_argument("--batch-dir", type=str, default=None, help="Cesta ke složce se všemi PDB strukturami (např. structures/all_pdbs)")
    parser.add_argument("--metadata", type=str, default=None, help="Cesta k dataset_metadata.tsv pro výběr struktur")
    parser.add_argument("--output-dir", type=str, default="structures/p2rank_outputs", help="Cesta pro uložení predikcí P2Ranku")
    parser.add_argument("--config", type=str, default="alphafold", help="P2Rank konfigurace (alphafold / default)")
    parser.add_argument("--threads", type=int, default=8, help="Počet vláken pro P2Rank")
    parser.add_argument("--chunk-size", type=int, default=500, help="Velikost dávky pro dávkový režim")
    parser.add_argument("--prank-exec", type=str, default=None, help="Vlastní cesta k binárce prank")
    args = parser.parse_args()

    if args.pdb:
        out = run_p2rank(args.pdb, prank_exec=args.prank_exec, output_dir=args.output_dir, config=args.config)
        parsed = parse_p2rank_output(out, args.pdb)
        print(f"\nVýsledek pro {args.pdb}:")
        print(f"Sekvence: {len(parsed['full_sequence'])} aa")
        print(f"Nalezeno kapes: {len(parsed['pockets'])}")
        for p in parsed['pockets'][:5]:
            print(f" - Kapsa #{p['pocket_id']} ({p['name']}): Score={p['score']:.2f}, Rezidua={p['residue_count']}, Střed={p['center']}")
    elif args.batch_dir or args.metadata:
        pdb_files = []
        if args.metadata and os.path.exists(args.metadata):
            with open(args.metadata, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader, None)
                for row in reader:
                    if row and len(row) > 1 and row[1].strip() != "NONE":
                        fn = row[1].strip()
                        base_d = args.batch_dir if args.batch_dir else "structures/all_pdbs"
                        p_cand = Path(base_d) / fn
                        if p_cand.exists():
                            pdb_files.append(p_cand)
        elif args.batch_dir and os.path.exists(args.batch_dir):
            pdb_files = [f for f in Path(args.batch_dir).glob("*.pdb") if "_pocket_" not in f.name]

        if not pdb_files:
            print("❌ Nenalezeny žádné PDB soubory k predikci.")
        else:
            run_p2rank_batch(
                pdb_files,
                prank_exec=args.prank_exec,
                output_dir=args.output_dir,
                config=args.config,
                threads=args.threads,
                chunk_size=args.chunk_size
            )
    else:
        parser.print_help()
