import os
import sys
import glob
import json
import shutil
import tempfile
import argparse
import subprocess
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

script_dir = os.path.dirname(os.path.abspath(__file__))
TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
NAME_TO_LABEL = {name: str(i) for i, name in enumerate(TARGET_NAMES)}

def create_alias_pdb(src_pdb, dst_pdb):
    """Vytvoří kopii PDB souboru do cílového umístění."""
    shutil.copy2(src_pdb, dst_pdb)

def extract_pocket_from_entry(entry, full_pdb_path, out_pocket_pdb):
    """
    Vytvoří/extrahuje PDB kapsy buď z původního PDB (plné atomy) nebo z CA koordinátů v datasetu.
    """
    coords = entry.get('protein_coords', [])
    seq = entry.get('binding_site_sequence', '')
    
    # 1. Zkusíme extrahovat kompletní rezidua z originálního PDB podle CA souřadnic
    if full_pdb_path and os.path.exists(full_pdb_path) and len(coords) > 0:
        try:
            from Bio.PDB import PDBParser, PDBIO, Select
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", full_pdb_path)
            target_coords = np.array(coords)
            
            class CoordPocketSelect(Select):
                def accept_residue(self, residue):
                    if 'CA' in residue:
                        ca = residue['CA'].get_coord()
                        dists = np.linalg.norm(target_coords - ca, axis=1)
                        return np.min(dists) < 0.1
                    return False
                    
            io = PDBIO()
            io.set_structure(structure)
            io.save(out_pocket_pdb, CoordPocketSelect())
            if os.path.exists(out_pocket_pdb) and os.path.getsize(out_pocket_pdb) > 100:
                return True
        except Exception:
            pass

    # 2. Případ, kdy máme explicitní seznam reziduí [(chain, res_num), ...]
    residues = entry.get('residues', entry.get('pocket_residues', []))
    if residues and full_pdb_path and os.path.exists(full_pdb_path):
        if extract_pocket_pdb_from_residues(full_pdb_path, residues, out_pocket_pdb):
            return True

    # 3. Fallback: vytvoření validního CA PDB přímo ze sekvence a souřadnic
    if coords and seq and len(coords) == len(seq):
        ONE_TO_THREE = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
            'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
            'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
            'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
        }
        try:
            with open(out_pocket_pdb, 'w') as f:
                for i, (aa, pt) in enumerate(zip(seq, coords), start=1):
                    three = ONE_TO_THREE.get(aa, 'ALA')
                    x, y, z = pt[0], pt[1], pt[2]
                    f.write(f"ATOM  {i:5d}  CA  {three:3s} A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n")
                f.write("END\n")
            return os.path.exists(out_pocket_pdb) and os.path.getsize(out_pocket_pdb) > 0
        except Exception:
            return False

    return False

def find_pdb_files(pdb_roots, test_limit=None):
    """Najde všechny plné PDB soubory a fyzické soubory kapes."""
    full_pdb_files = {}
    pocket_pdb_files = {}
    labels_by_pid = {}
    
    for root in pdb_roots:
        if not os.path.exists(root):
            continue
        for p in glob.glob(os.path.join(root, '**', '*.pdb'), recursive=True):
            fname = os.path.basename(p)
            base_id = fname.replace('.pdb', '')
            
            # Detekce třídy z cesty
            parts = os.path.normpath(p).split(os.sep)
            label = '-1'
            for part in reversed(parts):
                clean_part = part.replace('_prank_output', '')
                if clean_part in NAME_TO_LABEL:
                    label = NAME_TO_LABEL[clean_part]
                    break
            
            if '_pocket' in fname or '_pocket_' in p or 'prank_output' in p:
                pocket_pdb_files[p] = p
            else:
                if base_id not in full_pdb_files:
                    full_pdb_files[base_id] = p
                    labels_by_pid[base_id] = label
                    
    if test_limit:
        limited_pids = list(full_pdb_files.keys())[:test_limit]
        full_pdb_files = {pid: full_pdb_files[pid] for pid in limited_pids}
        labels_by_pid = {pid: labels_by_pid[pid] for pid in limited_pids}
        print(f"--- Testovací režim zapnut: zpracovávám pouze {len(full_pdb_files)} plných struktur ---")
        
    return full_pdb_files, pocket_pdb_files, labels_by_pid

def two_level_structure_clustering(
    full_tmscore_threshold=0.5,
    pocket_tmscore_threshold=0.5,
    alignment_type=2,
    pockets_json=None,
    suffix=None,
    test_limit=None,
    nr_threshold=None,
    threads=8
):
    cwd = os.getcwd()
    pdb_roots = [
        os.path.join(script_dir, 'structures'),
        os.path.join(script_dir, '..', 'structures'),
        os.path.join(script_dir, 'Binding_Sites'),
        os.path.join(script_dir, '..', 'Binding_Sites'),
        os.path.join(cwd, 'structures'),
        os.path.join(cwd, '..', 'structures'),
        os.path.join(cwd, 'Binding_Sites'),
        os.path.join(cwd, '..', 'Binding_Sites'),
        os.path.join(cwd, 'data_prep', 'structures'),
        os.path.join(cwd, 'data_prep', 'Binding_Sites')
    ]
    
    if suffix is None:
        suffix = f"_struct_pocket_{full_tmscore_threshold}_{pocket_tmscore_threshold}"
        if not suffix.startswith('_'):
            suffix = f"_{suffix}"

    print("\n========================================================")
    print("DVOUÚROVŇOVÉ STRUKTURNÍ SHLUKOVÁNÍ (FOLDSEEK TM-SCORE)")
    print("========================================================")
    print(f"1. Úroveň (Globální fold TM-score práh): {full_tmscore_threshold}")
    print(f"2. Úroveň (Kapsa/Binding site TM-score práh): {pocket_tmscore_threshold}")
    print(f"Výstupní suffix: {suffix}")
    print("========================================================\n")
    
    full_pdb_files, pocket_pdb_files, labels_by_pid = find_pdb_files(pdb_roots, test_limit=test_limit)
    print(f"Nalezeno {len(full_pdb_files)} unikátních plných PDB struktur.")
    if len(full_pdb_files) == 0:
        print("Chyba: Nebyly nalezeny žádné PDB soubory.")
        return None, None, None, None

    sorted_pids = sorted(list(full_pdb_files.keys()))
    pid_to_idx = {pid: i for i, pid in enumerate(sorted_pids)}
    n_proteins = len(sorted_pids)
    
    # Inicializace matice sousednosti pro graf proteinů
    adj = np.eye(n_proteins, dtype=int)
    
    with tempfile.TemporaryDirectory(prefix="fs_two_level_") as tmp_dir:
        # ----------------------------------------------------
        # 1. KROK: PŘÍPRAVA PLNÝCH STRUKTUR
        # ----------------------------------------------------
        tmp_full_pdbs = os.path.join(tmp_dir, "full_pdbs")
        os.makedirs(tmp_full_pdbs, exist_ok=True)
        
        for pid, src_path in full_pdb_files.items():
            create_alias_pdb(src_path, os.path.join(tmp_full_pdbs, f"{pid}.pdb"))
            
        # Volitelná NR filtrace
        if nr_threshold is not None:
            print(f"\n--- Spouštím NR předfiltraci (práh: {nr_threshold}) ---")
            nr_out = os.path.join(tmp_dir, "fs_nr_out")
            nr_tmp = os.path.join(tmp_dir, "fs_nr_tmp")
            os.makedirs(nr_tmp, exist_ok=True)
            
            subprocess.run([
                "foldseek", "easy-cluster",
                tmp_full_pdbs, nr_out, nr_tmp,
                "--tmscore-threshold", str(nr_threshold),
                "--alignment-type", str(alignment_type),
                "-c", "0.8",
                "--threads", str(threads),
                "--cluster-mode", "1"
            ], check=True)
            
            nr_tsv = f"{nr_out}_cluster.tsv"
            if os.path.exists(nr_tsv):
                with open(nr_tsv, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            r = parts[0].replace('.pdb', '')
                            m = parts[1].replace('.pdb', '')
                            if r in pid_to_idx and m in pid_to_idx:
                                adj[pid_to_idx[r], pid_to_idx[m]] = 1
                                adj[pid_to_idx[m], pid_to_idx[r]] = 1
        
        # ----------------------------------------------------
        # 2. KROK: GLOBÁLNÍ SHLUKOVÁNÍ PLNÝCH STRUKTUR (FOLDSEEK)
        # ----------------------------------------------------
        print(f"\n--- Spouštím globální Foldseek shlukování struktur (TM-score >= {full_tmscore_threshold}) ---")
        fs_full_out = os.path.join(tmp_dir, "fs_full_out")
        fs_full_tmp = os.path.join(tmp_dir, "fs_full_tmp")
        os.makedirs(fs_full_tmp, exist_ok=True)
        
        cmd_full = [
            "foldseek", "easy-cluster",
            tmp_full_pdbs, fs_full_out, fs_full_tmp,
            "--tmscore-threshold", str(full_tmscore_threshold),
            "--alignment-type", str(alignment_type),
            "-c", "0.8",
            "--threads", str(threads),
            "--cluster-mode", "1"
        ]
        
        try:
            subprocess.run(cmd_full, check=True)
            cluster_tsv = f"{fs_full_out}_cluster.tsv"
            if os.path.exists(cluster_tsv):
                with open(cluster_tsv, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            r = parts[0].replace('.pdb', '')
                            m = parts[1].replace('.pdb', '')
                            if r in pid_to_idx and m in pid_to_idx:
                                adj[pid_to_idx[r], pid_to_idx[m]] = 1
                                adj[pid_to_idx[m], pid_to_idx[r]] = 1
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Chyba při běhu Foldseek na plných strukturách: {e}")
            return None, None, None, None

        # ----------------------------------------------------
        # 3. KROK: PŘÍPRAVA A SHLUKOVÁNÍ STRUKTUR KAPES (POCKETS)
        # ----------------------------------------------------
        print(f"\n--- Příprava a zarovnání struktur kapes (TM-score >= {pocket_tmscore_threshold}) ---")
        tmp_pocket_pdbs = os.path.join(tmp_dir, "pocket_pdbs")
        os.makedirs(tmp_pocket_pdbs, exist_ok=True)
        
        pocket_to_parent_pid = {}
        
        # A) Zkusíme najít fyzické soubory kapes
        for ppath in pocket_pdb_files:
            fname = os.path.basename(ppath)
            clean_name = fname.replace('.pdb', '')
            parent_pid = clean_name.split('_pocket_')[0].replace('_prank_output', '')
            if parent_pid in pid_to_idx:
                dst = os.path.join(tmp_pocket_pdbs, fname)
                create_alias_pdb(ppath, dst)
                pocket_to_parent_pid[clean_name] = parent_pid
                
        # B) Pokud fyzické soubory kapes neexistují nebo je jich málo, zkusíme extrahovat z JSON datasetu
        if len(pocket_to_parent_pid) == 0:
            json_candidates = [
                pockets_json,
                os.path.join(script_dir, "p2rank_pockets_dataset.json"),
                os.path.join(script_dir, "..", "data_prep", "p2rank_pockets_dataset.json"),
                os.path.join(cwd, "p2rank_pockets_dataset.json"),
                os.path.join(cwd, "data_prep", "p2rank_pockets_dataset.json"),
                os.path.join(cwd, "..", "data_prep", "p2rank_pockets_dataset.json")
            ]
            valid_json = next((j for j in json_candidates if j and os.path.exists(j)), None)
            
            if valid_json:
                print(f"Extrahuji sub-struktury kapes z {valid_json}...")
                with open(valid_json, 'r') as f:
                    pockets_data = json.load(f)
                    
                for entry in pockets_data:
                    pocket_id = entry.get('pocket_id', entry.get('protein_id', ''))
                    base_name = os.path.basename(pocket_id)
                    parent_pid = entry.get('protein_id', base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', ''))
                    
                    if parent_pid in pid_to_idx:
                        full_pdb = full_pdb_files.get(parent_pid, entry.get('pdb_file', ''))
                        pocket_fname = f"{base_name}.pdb"
                        out_p_pdb = os.path.join(tmp_pocket_pdbs, pocket_fname)
                        if extract_pocket_from_entry(entry, full_pdb, out_p_pdb):
                            pocket_to_parent_pid[base_name.replace('.pdb', '')] = parent_pid
                                
        print(f"Připraveno {len(pocket_to_parent_pid)} sub-struktur kapes pro Foldseek analýzu.")
        
        # Pokud máme kapsy, spustíme na nich Foldseek vyhledávání / shlukování
        if len(pocket_to_parent_pid) > 1:
            fs_pocket_out = os.path.join(tmp_dir, "fs_pocket_out")
            fs_pocket_tmp = os.path.join(tmp_dir, "fs_pocket_tmp")
            os.makedirs(fs_pocket_tmp, exist_ok=True)
            
            # Použijeme easy-search pro nalezení všech strukturně podobných párů kapes
            cmd_pocket = [
                "foldseek", "easy-search",
                tmp_pocket_pdbs, tmp_pocket_pdbs,
                fs_pocket_out + ".m8", fs_pocket_tmp,
                "--tmscore-threshold", str(pocket_tmscore_threshold),
                "--alignment-type", str(alignment_type),
                "-c", "0.5",
                "--format-output", "query,target,alntmscore",
                "--threads", str(threads)
            ]
            
            try:
                subprocess.run(cmd_pocket, check=True)
                pocket_m8 = fs_pocket_out + ".m8"
                pocket_edges = 0
                if os.path.exists(pocket_m8):
                    with open(pocket_m8, 'r') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                q_pock = parts[0].replace('.pdb', '')
                                t_pock = parts[1].replace('.pdb', '')
                                
                                if q_pock in pocket_to_parent_pid and t_pock in pocket_to_parent_pid:
                                    pid1 = pocket_to_parent_pid[q_pock]
                                    pid2 = pocket_to_parent_pid[t_pock]
                                    
                                    if pid1 != pid2:
                                        idx1 = pid_to_idx[pid1]
                                        idx2 = pid_to_idx[pid2]
                                        if adj[idx1, idx2] == 0:
                                            adj[idx1, idx2] = 1
                                            adj[idx2, idx1] = 1
                                            pocket_edges += 1
                print(f"Foldseek identifikoval strukturně podobné kapsy -> přidáno {pocket_edges} nových vazeb mezi proteiny.")
            except subprocess.CalledProcessError as e:
                print(f"Varování: Foldseek vyhledávání nad kapsami selhalo: {e}")
        else:
            print("Informace: Nebyl nalezen dostatek struktur kapes, shlukování proběhne pouze na úrovni celých struktur.")

        # ----------------------------------------------------
        # 4. KROK: GRAFOVÉ SPOJENÍ (CONNECTED COMPONENTS)
        # ----------------------------------------------------
        print("\nHledání spojených komponent (finální clustery)...")
        graph = csr_matrix(adj)
        n_clusters, cluster_labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        print(f"Výsledný počet nezávislých clusterů: {n_clusters}")
        
        clusters = defaultdict(list)
        for i, pid in enumerate(sorted_pids):
            c = int(cluster_labels[i])
            clusters[c].append(pid)
            
        out_clusters = {}
        cluster_reps = []
        cluster_majority_labels = []
        
        for c, members in clusters.items():
            rep = members[0]
            out_clusters[rep] = members
            cluster_reps.append(rep)
            labels_in_c = [labels_by_pid.get(m, '-1') for m in members]
            majority = Counter(labels_in_c).most_common(1)[0][0]
            cluster_majority_labels.append(majority)
            
        # ----------------------------------------------------
        # 5. KROK: DIAGNOSTIKA A STRATIFIED GROUP SPLIT (80/10/10)
        # ----------------------------------------------------
        print("\n" + "="*65)
        print("DIAGNOSTIKA CLUSTERŮ A TŘÍD")
        print("="*65)
        
        protein_labels = np.array([int(labels_by_pid.get(pid, -1)) for pid in sorted_pids])
        
        for label_idx, name in enumerate(TARGET_NAMES):
            pids_in_cls = [pid for pid in sorted_pids if labels_by_pid.get(pid) == str(label_idx)]
            clusters_in_cls = set([cluster_labels[pid_to_idx[p]] for p in pids_in_cls])
            cluster_sizes_in_cls = [len([p for p in pids_in_cls if cluster_labels[pid_to_idx[p]] == c]) for c in clusters_in_cls]
            max_size = max(cluster_sizes_in_cls) if cluster_sizes_in_cls else 0
            mean_size = np.mean(cluster_sizes_in_cls) if cluster_sizes_in_cls else 0
            print(f"Třída {name:<12s}: {len(pids_in_cls):5d} proteinů v {len(clusters_in_cls):4d} clusterech (Max cluster: {max_size:4d}, Průměr: {mean_size:.1f})")

        print("\n" + "-"*65)
        print("Rozdělování pomocí sklearn.model_selection.StratifiedGroupKFold (80/10/10)...")
        print("-" * 65)
        
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
        
        fold_assignments = np.zeros(len(sorted_pids), dtype=int)
        for fold_idx, (_, test_idx) in enumerate(sgkf.split(sorted_pids, protein_labels, groups=cluster_labels)):
            fold_assignments[test_idx] = fold_idx
            
        train_mask = fold_assignments < 8
        val_mask = fold_assignments == 8
        test_mask = fold_assignments == 9
        
        splits = {
            "train": [sorted_pids[i] for i in range(len(sorted_pids)) if train_mask[i]],
            "validation": [sorted_pids[i] for i in range(len(sorted_pids)) if val_mask[i]],
            "test": [sorted_pids[i] for i in range(len(sorted_pids)) if test_mask[i]]
        }
        
        print(f"\n{'Třída':<12s} | {'Train (80%)':<14s} | {'Val (10%)':<14s} | {'Test (10%)':<14s} | {'Celkem':<8s}")
        print("-" * 65)
        for label_idx, name in enumerate(TARGET_NAMES):
            lbl_str = str(label_idx)
            tr_c = sum(1 for p in splits["train"] if labels_by_pid.get(p) == lbl_str)
            va_c = sum(1 for p in splits["validation"] if labels_by_pid.get(p) == lbl_str)
            te_c = sum(1 for p in splits["test"] if labels_by_pid.get(p) == lbl_str)
            tot = tr_c + va_c + te_c
            tr_pct = (tr_c / tot * 100) if tot > 0 else 0
            va_pct = (va_c / tot * 100) if tot > 0 else 0
            te_pct = (te_c / tot * 100) if tot > 0 else 0
            print(f"{name:<12s} | {tr_c:5d} ({tr_pct:4.1f}%) | {va_c:5d} ({va_pct:4.1f}%) | {te_c:5d} ({te_pct:4.1f}%) | {tot:5d}")
            
        print("-" * 65)
        print(f"{'CELKEM':<12s} | {len(splits['train']):5d} ({(len(splits['train'])/len(sorted_pids)*100):4.1f}%) | {len(splits['validation']):5d} ({(len(splits['validation'])/len(sorted_pids)*100):4.1f}%) | {len(splits['test']):5d} ({(len(splits['test'])/len(sorted_pids)*100):4.1f}%) | {len(sorted_pids):5d}")
        print("=" * 65)
        
        # ----------------------------------------------------
        # 6. KROK: ULOŽENÍ SOUBORŮ
        # ----------------------------------------------------
        save_dir = os.path.join(cwd, "data_prep")
        if not os.path.exists(save_dir):
            save_dir = script_dir
            
        train_path = os.path.join(save_dir, f"train{suffix}.txt")
        val_path = os.path.join(save_dir, f"validation{suffix}.txt")
        test_path = os.path.join(save_dir, f"test{suffix}.txt")
        clusters_path = os.path.join(save_dir, f"clusters{suffix}.json")
        
        with open(train_path, "w") as f:
            f.write("\n".join(splits["train"]) + "\n")
        with open(val_path, "w") as f:
            f.write("\n".join(splits["validation"]) + "\n")
        with open(test_path, "w") as f:
            f.write("\n".join(splits["test"]) + "\n")
        with open(clusters_path, "w") as f:
            json.dump(out_clusters, f, indent=4)
            
        print(f"\nSoubory úspěšně uloženy do {save_dir}:")
        print(f" - train{suffix}.txt")
        print(f" - validation{suffix}.txt")
        print(f" - test{suffix}.txt")
        print(f" - clusters{suffix}.json")
        
        return splits["train"], splits["validation"], splits["test"], out_clusters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dvouúrovňové strukturní shlukování pomocí Foldseek (celý protein + kapsy).")
    parser.add_argument("--full-tmscore-threshold", "--full-tmscore", type=float, default=0.5,
                        help="Práh TM-score pro globální shlukování plných struktur (default: 0.5).")
    parser.add_argument("--pocket-tmscore-threshold", "--pocket-tmscore", type=float, default=0.5,
                        help="Práh TM-score pro zarovnání struktur kapes (default: 0.5).")
    parser.add_argument("--alignment-type", type=int, default=2,
                        help="Typ zarovnání pro Foldseek (1 = TM-align, 2 = 3Di, default: 2).")
    parser.add_argument("--pockets-json", default=None,
                        help="Cesta k p2rank_pockets_dataset.json (pro případnou extrakci sub-struktur kapes).")
    parser.add_argument("--suffix", default=None,
                        help="Přípona pro výstupní soubory (např. _struct_pocket_0.5).")
    parser.add_argument("--nr-threshold", type=float, default=None,
                        help="Práh TM-score pro prvotní Non-Redundant filtraci.")
    parser.add_argument("--threads", type=int, default=8, help="Počet vláken pro Foldseek.")
    parser.add_argument("--test", action="store_true", help="Omezí počet struktur pro testování.")
    
    args = parser.parse_args()
    
    two_level_structure_clustering(
        full_tmscore_threshold=args.full_tmscore_threshold,
        pocket_tmscore_threshold=args.pocket_tmscore_threshold,
        alignment_type=args.alignment_type,
        pockets_json=args.pockets_json,
        suffix=args.suffix,
        test_limit=30 if args.test else None,
        nr_threshold=args.nr_threshold,
        threads=args.threads
    )
