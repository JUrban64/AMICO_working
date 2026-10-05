import os
import json
import random
import shutil
import subprocess
import tempfile
import glob
import argparse
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))


def create_alias_pdb(src_pdb, dst_pdb):
    """Creates a physical copy of PDB file (symlinks may cause issues in container/HPC environments)."""
    shutil.copy2(src_pdb, dst_pdb)


def load_metadata_labels(metadata_path=None, candidate_roots=None):
    """
    Loads protein ID / pdb_file -> cofactor label (0-4) mapping from dataset_metadata.tsv or master_dataset_cache.json.
    """
    target_names = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
    name_to_label = {name: str(i) for i, name in enumerate(target_names)}
    metadata_labels = {}
    found_path = None
    
    candidates = []
    if metadata_path:
        candidates.append(metadata_path)
        
    cwd = os.getcwd()
    search_dirs = [
        os.path.join(script_dir, 'structures'),
        os.path.join(script_dir, '..', 'structures'),
        os.path.join(cwd, 'structures'),
        os.path.join(cwd, '..', 'structures'),
        os.path.join(cwd, 'data_prep', 'structures')
    ]
    if candidate_roots:
        search_dirs.extend(candidate_roots)
        
    for d in search_dirs:
        candidates.append(os.path.join(d, 'dataset_metadata.tsv'))
        candidates.append(os.path.join(d, 'master_dataset_cache.json'))
        
    for cand in candidates:
        if cand and os.path.exists(cand) and os.path.getsize(cand) > 0:
            found_path = cand
            break
            
    if not found_path:
        return {}
        
    print(f"-> Loading metadata and cofactor annotations from: {found_path}")
    if found_path.endswith('.tsv') or found_path.endswith('.txt'):
        import csv
        with open(found_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            headers = next(reader, None)
            for row in reader:
                if not row or len(row) < 3:
                    continue
                acc = row[0].strip()
                pdb_file = row[1].strip()
                cofactors_str = row[2].strip()
                
                cofactors_list = [c.strip() for c in cofactors_str.split(';') if c.strip()]
                assigned_label = None
                for cof in cofactors_list:
                    if cof in name_to_label:
                        assigned_label = name_to_label[cof]
                        break
                        
                if assigned_label is not None:
                    if pdb_file and pdb_file != 'NONE':
                        pid_stem = pdb_file.replace('.pdb', '').strip()
                        metadata_labels[pid_stem] = assigned_label
                    metadata_labels[acc] = assigned_label
    elif found_path.endswith('.json'):
        with open(found_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for acc, entry in data.items():
                cofs = entry.get('cofactors', [])
                if isinstance(cofs, str):
                    cofs = [cofs]
                assigned_label = None
                for cof in cofs:
                    if cof in name_to_label:
                        assigned_label = name_to_label[cof]
                        break
                if assigned_label is not None:
                    metadata_labels[acc] = assigned_label
                    
    print(f"-> Successfully loaded {len(metadata_labels)} records with cofactors from metadata.")
    return metadata_labels


def cluster_structures(test_limit=None, tmscore_threshold=0.5, nr_threshold=None, metadata_file=None):
    cwd = os.getcwd()
    pdb_roots = [
        os.path.join(script_dir, 'structures'), os.path.join(script_dir, 'structures', 'all_pdbs'),
        os.path.join(script_dir, '..', 'structures'), os.path.join(script_dir, '..', 'structures', 'all_pdbs'),
        os.path.join(cwd, 'structures'), os.path.join(cwd, 'structures', 'all_pdbs'),
        os.path.join(cwd, '..', 'structures'), os.path.join(cwd, '..', 'structures', 'all_pdbs'),
        os.path.join(cwd, 'data_prep', 'structures', 'all_pdbs')
    ]
    
    metadata_labels = load_metadata_labels(metadata_path=metadata_file, candidate_roots=pdb_roots)
    
    pdb_files = []
    for root in pdb_roots:
        if os.path.exists(root):
            for p in glob.glob(os.path.join(root, '**', '*.pdb'), recursive=True):
                # Cluster whole structures only, not pocket cutouts
                if '_pocket' not in p and 'prank_output' not in p:
                    pdb_files.append(p)

    if test_limit:
        pdb_files = pdb_files[:test_limit]
        print(f"--- Test mode active: processing only {len(pdb_files)} structures ---")

    with tempfile.TemporaryDirectory(prefix="fs_pdb_") as tmp_dir:
        pdb_data = {}
        tmp_pdb_dir = os.path.join(tmp_dir, "pdbs")
        os.makedirs(tmp_pdb_dir, exist_ok=True)
        
        for pdb_file in pdb_files:
            base_id = os.path.basename(pdb_file).replace(".pdb", "")
            
            if base_id in pdb_data:
                continue
                
            alias_pdb = os.path.join(tmp_pdb_dir, f"{base_id}.pdb")
            create_alias_pdb(pdb_file, alias_pdb)
            pdb_data[base_id] = alias_pdb

        print(f"Loaded {len(pdb_data)} unique PDB structures")
        if len(pdb_data) == 0:
            print("No PDBs found.")
            return None, None, None, None
        
        target_names = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']
        name_to_label = {name: str(i) for i, name in enumerate(target_names)}
        
        orig_labels_by_pid = {}
        for p in pdb_files:
            pid = os.path.basename(p).replace(".pdb", "")
            
            if pid in metadata_labels:
                orig_labels_by_pid[pid] = metadata_labels[pid]
            elif pid.split('_')[0] in metadata_labels:
                orig_labels_by_pid[pid] = metadata_labels[pid.split('_')[0]]
            else:
                parts = os.path.normpath(p).split(os.sep)
                for part in reversed(parts):
                    if part in name_to_label:
                        orig_labels_by_pid[pid] = name_to_label[part]
                        break
        
        labels_by_pid = orig_labels_by_pid
        
        unlabeled = [pid for pid in pdb_data.keys() if labels_by_pid.get(pid, '-1') == '-1']
        if unlabeled:
            print(f"⚠️ Warning: {len(unlabeled)} / {len(pdb_data)} structures have no assigned cofactor class (label -1).")
            
        fs_out_prefix = os.path.join(tmp_dir, "fs_out")
        fs_tmp_dir = os.path.join(tmp_dir, "fs_tmp")
        os.makedirs(fs_tmp_dir, exist_ok=True)
        
        # === STAGE 1: NON-REDUNDANT PRE-FILTERING ===
        if nr_threshold is not None:
            print(f"\n=== Running NR pre-filtering with TM-score threshold {nr_threshold} ===")
            nr_out_prefix = os.path.join(tmp_dir, "fs_nr_out")
            nr_tmp_dir = os.path.join(tmp_dir, "fs_nr_tmp")
            os.makedirs(nr_tmp_dir, exist_ok=True)
            
            nr_command = [
                "foldseek", "easy-cluster", 
                tmp_pdb_dir, nr_out_prefix, nr_tmp_dir,
                "--tmscore-threshold", str(nr_threshold),
                "--alignment-type", "2", 
                "-c", "0.8",
                "--threads", "8",
                "--cluster-mode", "1"
            ]
            
            try:
                subprocess.run(nr_command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running Foldseek NR pre-filtering: {e}")
                return None, None, None, None
            except FileNotFoundError:
                print("Error: Foldseek executable not found.")
                return None, None, None, None
                
            nr_cluster_tsv = f"{nr_out_prefix}_cluster.tsv"
            if not os.path.exists(nr_cluster_tsv):
                print(f"Error: NR Foldseek output {nr_cluster_tsv} not found.")
                return None, None, None, None
                
            nr_reps = set()
            with open(nr_cluster_tsv, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        rep = parts[0].replace('.pdb', '')
                        nr_reps.add(rep)
                        
            print(f"NR filtering reduced dataset from {len(pdb_data)} to {len(nr_reps)} unique representatives.")
            
            tmp_pdb_dir_nr = os.path.join(tmp_dir, "pdbs_nr")
            os.makedirs(tmp_pdb_dir_nr, exist_ok=True)
            
            for pid in nr_reps:
                if pid in pdb_data:
                    try:
                        os.symlink(pdb_data[pid], os.path.join(tmp_pdb_dir_nr, f"{pid}.pdb"))
                    except OSError:
                        shutil.copy2(pdb_data[pid], os.path.join(tmp_pdb_dir_nr, f"{pid}.pdb"))
            
            tmp_pdb_dir = tmp_pdb_dir_nr
            pdb_data = {k: v for k, v in pdb_data.items() if k in nr_reps}
        
        # === STAGE 2: MAIN CLUSTERING FOR TRAIN/VAL/TEST SPLIT ===
        print(f"\n=== Running main clustering with TM-score threshold {tmscore_threshold} ===")
        
        command = [
            "foldseek", "easy-cluster", 
            tmp_pdb_dir, fs_out_prefix, fs_tmp_dir,
            "--tmscore-threshold", str(tmscore_threshold),
            "--alignment-type", "2",
            "-c", "0.8",
            "--threads", "8",
            "--cluster-mode", "1"
        ]
        
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Foldseek: {e}")
            return None, None, None, None
        except FileNotFoundError:
            print("Error: Foldseek executable not found. Please ensure it is installed and in your PATH.")
            return None, None, None, None
            
        cluster_tsv = f"{fs_out_prefix}_cluster.tsv"
        clusters = {}
        if not os.path.exists(cluster_tsv):
            print(f"Error: Foldseek output {cluster_tsv} not found.")
            return None, None, None, None
            
        with open(cluster_tsv, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    rep = parts[0].replace('.pdb', '')
                    member = parts[1].replace('.pdb', '')
                    if rep not in clusters:
                        clusters[rep] = []
                    clusters[rep].append(member)
                    
        print(f"Foldseek identified {len(clusters)} clusters.")

        # Map clusters to individual proteins
        sorted_pids = sorted(list(pdb_data.keys()))
        pid_to_cluster = {}
        for c_idx, (rep, members) in enumerate(clusters.items()):
            for m in members:
                pid_to_cluster[m] = c_idx
                
        cluster_labels = [pid_to_cluster.get(p, 0) for p in sorted_pids]
        protein_labels = np.array([int(labels_by_pid.get(p, -1)) for p in sorted_pids])
        
        print("\n" + "=" * 65)
        print("CLUSTER AND CLASS DIAGNOSTICS")
        print("=" * 65)
        for label_idx, name in enumerate(target_names):
            pids_in_cls = [pid for pid in sorted_pids if labels_by_pid.get(pid) == str(label_idx)]
            clusters_in_cls = set([pid_to_cluster[p] for p in pids_in_cls if p in pid_to_cluster])
            cluster_sizes_in_cls = [len([p for p in pids_in_cls if pid_to_cluster.get(p) == c]) for c in clusters_in_cls]
            max_size = max(cluster_sizes_in_cls) if cluster_sizes_in_cls else 0
            mean_size = np.mean(cluster_sizes_in_cls) if cluster_sizes_in_cls else 0
            print(f"Class {name:<12s}: {len(pids_in_cls):5d} proteins across {len(clusters_in_cls):4d} clusters (Max: {max_size:4d}, Mean: {mean_size:.1f})")

        print("\n" + "-" * 65)
        print("Splitting with sklearn.model_selection.StratifiedGroupKFold (80/10/10)...")
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
        
        print(f"\n{'Class':<12s} | {'Train (80%)':<14s} | {'Val (10%)':<14s} | {'Test (10%)':<14s} | {'Total':<8s}")
        print("-" * 65)
        for label_idx, name in enumerate(target_names):
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
        print(f"{'TOTAL':<12s} | {len(splits['train']):5d} ({(len(splits['train'])/len(sorted_pids)*100):4.1f}%) | {len(splits['validation']):5d} ({(len(splits['validation'])/len(sorted_pids)*100):4.1f}%) | {len(splits['test']):5d} ({(len(splits['test'])/len(sorted_pids)*100):4.1f}%) | {len(sorted_pids):5d}")
        print("=" * 65)
        
        return splits["train"], splits["validation"], splits["test"], clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster PDB structures using Foldseek.")
    parser.add_argument("--metadata", default=None,
                        help="Path to dataset_metadata.tsv or master_dataset_cache.json for cofactor class annotations.")
    parser.add_argument("--test", action="store_true", help="Limit number of PDB files to 30 for quick testing.")
    parser.add_argument("--tmscore-threshold", "--tmscore", type=float, default=0.5,
                        help="TM-score threshold for Foldseek easy-cluster (default: 0.5).")
    parser.add_argument("--nr-threshold", type=float, default=None,
                        help="TM-score threshold for initial Non-Redundant pre-filtering (e.g. 0.95).")
    args = parser.parse_args()

    limit = 30 if args.test else None

    train, validation, test, clusters = cluster_structures(
        test_limit=limit, 
        tmscore_threshold=args.tmscore_threshold,
        nr_threshold=args.nr_threshold,
        metadata_file=args.metadata
    )

    if train is not None:
        target_suffix = "_mil"
        nr_suffix = f"_nr{args.nr_threshold}" if args.nr_threshold is not None else ""
        suffix = f"{target_suffix}_{args.tmscore_threshold}{nr_suffix}"

        with open(os.path.join(script_dir, f'train{suffix}.txt'), 'w') as f:
            for item in train:
                f.write(f"{item}\n")    

        with open(os.path.join(script_dir, f'validation{suffix}.txt'), 'w') as f:
            for item in validation:
                f.write(f"{item}\n")
        
        with open(os.path.join(script_dir, f'test{suffix}.txt'), 'w') as f:
            for item in test:
                f.write(f"{item}\n")
                
        with open(os.path.join(script_dir, f'clusters{suffix}.json'), 'w') as f:
            json.dump(clusters, f, indent=4)
        
        print(f"Files saved: train{suffix}.txt, validation{suffix}.txt, test{suffix}.txt, clusters{suffix}.json")