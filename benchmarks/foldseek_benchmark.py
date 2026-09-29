import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import glob
import json
import argparse
import tempfile
import subprocess
import shutil
import time
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset import load_split_ids, match_id, TARGET_NAMES


def find_foldseek_binary(custom_bin=None):
    """
    Vyhledá spustitelný soubor foldseek v systému nebo v conda prostředích.
    """
    if custom_bin and os.path.exists(custom_bin) and os.access(custom_bin, os.X_OK):
        return custom_bin

    which_path = shutil.which("foldseek")
    if which_path:
        return which_path

    # Obvyklá umístění v conda/miniforge
    home = os.path.expanduser("~")
    candidate_paths = [
        os.path.join(home, "miniforge3", "envs", "foldseek", "bin", "foldseek"),
        os.path.join(home, "anaconda3", "envs", "foldseek", "bin", "foldseek"),
        os.path.join(home, "miniconda3", "envs", "foldseek", "bin", "foldseek"),
        "/opt/homebrew/bin/foldseek",
        "/usr/local/bin/foldseek",
        os.path.join(home, ".local", "bin", "foldseek")
    ]
    for p in candidate_paths:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p

    return None


def get_pdb_paths(root_dir, extra_pdb_dir=None):
    """
    Prohledá známá úložiště PDB struktur v projektu i okolí.
    """
    pdb_roots = []
    if extra_pdb_dir and os.path.exists(extra_pdb_dir):
        pdb_roots.append(extra_pdb_dir)

    pdb_roots.extend([
        os.path.join(root_dir, 'data_prep', 'structures'),
        os.path.join(root_dir, 'data_prep', 'Binding_Sites'),
        os.path.join(root_dir, 'structures', 'all_pdbs'),
        os.path.join(root_dir, 'structures'),
        os.path.join(root_dir, 'Binding_Sites'),
        os.path.join(root_dir, '..', 'EquiPocket-MIL-', 'structures'),
        os.path.join(root_dir, '..', 'EquiPocket-MIL-', 'Binding_Sites'),
        os.path.join(root_dir, '..', 'AMICO_workign', 'structures'),
        os.path.join(root_dir, '..', 'AMICO_workign', 'data_prep', 'structures')
    ])

    pdb_files = {}
    for r in pdb_roots:
        if os.path.exists(r):
            for p in glob.glob(os.path.join(r, '**', '*.pdb'), recursive=True):
                if '_pocket' not in p and 'prank_output' not in p:
                    base_id = os.path.basename(p).replace('.pdb', '')
                    if base_id not in pdb_files:
                        pdb_files[base_id] = p
    return pdb_files


def get_valid_dataset_pids(root_dir):
    """
    Získá ID proteinů, které reálně existují v esm_dataset.pt / esm_full_proteins.pt.
    Slouží pro férové porovnání (stejná podmnožina jako u neuronových sítí).
    """
    dataset_candidates = [
        os.path.join(root_dir, 'data_prep', 'esm_dataset.pt'),
        os.path.join(root_dir, 'esm_dataset.pt'),
        os.path.join(root_dir, '..', 'AMICO_workign', 'data_prep', 'esm_dataset.pt')
    ]
    pockets_path = next((p for p in dataset_candidates if os.path.exists(p)), None)
    if not pockets_path:
        return None

    try:
        import torch
        raw_pockets = torch.load(pockets_path, weights_only=False)
        pids = set()
        for item in raw_pockets:
            raw_pid = item['protein_id']
            base_name = os.path.basename(raw_pid)
            pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
            pids.add(pid)

        full_prot_candidates = [
            os.path.join(os.path.dirname(pockets_path), 'esm_full_proteins.pt'),
            os.path.join(root_dir, 'data_prep', 'esm_full_proteins.pt'),
            os.path.join(root_dir, 'esm_full_proteins.pt')
        ]
        full_prot_path = next((p for p in full_prot_candidates if os.path.exists(p)), None)
        if full_prot_path:
            full_prots = torch.load(full_prot_path, weights_only=False)
            pids = pids.intersection(set(full_prots.keys()))

        return pids
    except Exception as e:
        print(f"Varování při načítání datasetu: {e}")
        return None


def get_labels_mapping(root_dir, pdb_files):
    """
    Získá mapování ID proteinu -> index třídy kofaktoru (0..4).
    Primárně z názvů adresářů, sekundárně z esm_dataset.pt nebo metadata TSV.
    """
    name_to_label = {name: i for i, name in enumerate(TARGET_NAMES)}
    labels_by_pid = {}

    # 1. Z adresářové struktury
    for pid, path in pdb_files.items():
        parts = os.path.normpath(path).split(os.sep)
        for part in reversed(parts):
            if part in name_to_label:
                labels_by_pid[pid] = name_to_label[part]
                break

    # 2. Z esm_dataset.pt jako fallback
    dataset_candidates = [
        os.path.join(root_dir, 'data_prep', 'esm_dataset.pt'),
        os.path.join(root_dir, 'esm_dataset.pt')
    ]
    pockets_path = next((p for p in dataset_candidates if os.path.exists(p)), None)
    if pockets_path:
        try:
            import torch
            raw_pockets = torch.load(pockets_path, weights_only=False)
            for item in raw_pockets:
                raw_pid = item['protein_id']
                base_name = os.path.basename(raw_pid)
                pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
                lbl = item.get('label')
                if lbl is not None and pid not in labels_by_pid:
                    if torch.is_tensor(lbl):
                        lbl = lbl.item()
                    labels_by_pid[pid] = int(lbl)
        except Exception:
            pass

    return labels_by_pid


def run_foldseek_benchmark(split_suffix='mil_0.5', use_nr=False, all_pdbs=False,
                           threads=8, foldseek_bin=None, custom_pdb_dir=None,
                           target_root=None):
    """
    Spustí Foldseek 1-NN Benchmark:
      - Test split slouží jako QUERY.
      - Train split slouží jako DATABASE (Target).
    Vrací slovník s metrikami (accuracy, macro F1, per-class F1, čas).
    """
    r_dir = target_root or project_root
    fs_bin = find_foldseek_binary(foldseek_bin)
    if not fs_bin:
        msg = ("Příkaz 'foldseek' nebyl nalezen v systémové PATH ani v conda prostředí "
               "(/Users/jachymurban/miniforge3/envs/foldseek/bin/foldseek).")
        print(f"CHYBA: {msg}")
        return {"status": f"FAILED: {msg}"}

    print(f"\n=================================================================")
    print(f"       FOLDSEEK 1-NN STRUCTURAL BENCHMARK")
    print(f"=================================================================")
    print(f"Použitý Foldseek: {fs_bin}")
    print(f"Split suffix:    {split_suffix} (use_nr={use_nr})")
    print(f"Threads:         {threads}")

    start_time = time.time()

    # 1. Načtení splitů
    train_ids_set, val_ids_set, test_ids_set = load_split_ids(r_dir, split_suffix=split_suffix, use_nr=use_nr)
    train_ids = list(train_ids_set)
    test_ids = list(test_ids_set)

    if len(train_ids) == 0 or len(test_ids) == 0:
        msg = f"Soubory splitů pro suffix '{split_suffix}' nebyly nalezeny v data_prep/."
        print(f"Chyba: {msg}")
        return {"status": f"FAILED: {msg}"}

    print(f"Načteno {len(train_ids)} train a {len(test_ids)} test proteinových ID.")

    # 2. Filtrování na proteiny, které jsou skutečně v datasetu (pokud není all_pdbs)
    if not all_pdbs:
        valid_pids = get_valid_dataset_pids(r_dir)
        if valid_pids:
            orig_tr, orig_te = len(train_ids), len(test_ids)
            train_ids = [pid for pid in train_ids if match_id(pid, valid_pids)]
            test_ids = [pid for pid in test_ids if match_id(pid, valid_pids)]
            print(f"\n[Férové srovnání - filtrováno podle platných dataset PIDs]:")
            print(f" - Train (Database): {len(train_ids)} (z {orig_tr})")
            print(f" - Test (Query):     {len(test_ids)} (z {orig_te})\n")

    # 3. Vyhledání PDB souborů
    pdb_files = get_pdb_paths(r_dir, extra_pdb_dir=custom_pdb_dir)
    print(f"Nalezeno celkem {len(pdb_files)} unikátních PDB souborů na disku.")

    # 4. Labely proteinů
    labels_by_pid = get_labels_mapping(r_dir, pdb_files)

    train_labels = [labels_by_pid[pid] for pid in train_ids if pid in labels_by_pid]
    if len(train_labels) == 0:
        msg = "Pro train proteiny nebyly nalezeny žádné kofaktorové labely."
        print(f"Chyba: {msg}")
        return {"status": f"FAILED: {msg}"}

    majority_train_label = Counter(train_labels).most_common(1)[0][0]
    majority_name = TARGET_NAMES[majority_train_label]
    print(f"Majority train label (fallback pro nulové hity): {majority_name} (index {majority_train_label})")

    # 5. Dočasná složka pro zarovnání (Test = Query, Train = Database)
    with tempfile.TemporaryDirectory(prefix="foldseek_bench_") as tmp_dir:
        train_dir = os.path.join(tmp_dir, "train_database_pdbs")
        test_dir = os.path.join(tmp_dir, "test_query_pdbs")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        copied_train = 0
        for pid in train_ids:
            if pid in pdb_files:
                dst = os.path.join(train_dir, f"{pid}.pdb")
                try:
                    os.symlink(pdb_files[pid], dst)
                except OSError:
                    shutil.copy2(pdb_files[pid], dst)
                copied_train += 1

        copied_test = 0
        for pid in test_ids:
            if pid in pdb_files:
                dst = os.path.join(test_dir, f"{pid}.pdb")
                try:
                    os.symlink(pdb_files[pid], dst)
                except OSError:
                    shutil.copy2(pdb_files[pid], dst)
                copied_test += 1

        print(f"Připraveno do databáze {copied_train} train PDBs a k dotazování {copied_test} test PDBs.")

        if copied_train == 0 or copied_test == 0:
            msg = f"Nebyly nalezeny fyzické PDB soubory pro train ({copied_train}) nebo test ({copied_test})."
            print(f"Chyba: {msg}")
            return {"status": f"FAILED: {msg}"}

        out_tsv = os.path.join(tmp_dir, "aln.tsv")
        fs_tmp = os.path.join(tmp_dir, "fs_tmp")
        os.makedirs(fs_tmp, exist_ok=True)

        # Spuštění Foldseek easy-search: Query = test_dir, Target = train_dir
        cmd = [
            fs_bin, "easy-search",
            test_dir, train_dir, out_tsv, fs_tmp,
            "--format-output", "query,target,evalue,qtmscore,bits",
            "-e", "10.0",
            "--threads", str(threads)
        ]

        print(f"\nSpouštím Foldseek easy-search (Query: Test -> Database: Train)...")
        try:
            subprocess.run(cmd, check=True)
            print("Foldseek easy-search úspěšně dokončen.")
        except subprocess.CalledProcessError as e:
            msg = f"Foldseek selhal s chybou: {e}"
            print(f"CHYBA: {msg}")
            return {"status": f"FAILED: {msg}"}

        if not os.path.exists(out_tsv):
            msg = "Foldseek nevygeneroval výstupní soubor aln.tsv."
            print(f"Chyba: {msg}")
            return {"status": f"FAILED: {msg}"}

        # 6. Zpracování 1-NN (Top-1 hit podle nejvyššího qtmscore a nejnižšího evalue)
        df = pd.read_csv(out_tsv, sep='\t', header=None, names=["query", "target", "evalue", "qtmscore", "bits"])
        df['query'] = df['query'].astype(str).str.replace('.pdb', '', regex=False)
        df['target'] = df['target'].astype(str).str.replace('.pdb', '', regex=False)

        # Řazení: primárně podle qtmscore (sestupně), sekundárně evalue (vzestupně)
        df = df.sort_values(by=['query', 'qtmscore', 'evalue'], ascending=[True, False, True])
        top1_hits = df.drop_duplicates(subset=['query'], keep='first').set_index('query')

        y_true = []
        y_pred = []
        hit_counts = 0

        for pid in test_ids:
            if pid not in labels_by_pid:
                continue

            true_lbl = labels_by_pid[pid]
            y_true.append(true_lbl)

            if pid in top1_hits.index:
                hit_counts += 1
                target_pid = top1_hits.loc[pid, 'target']
                if isinstance(target_pid, pd.Series):
                    target_pid = target_pid.iloc[0]
                pred_lbl = labels_by_pid.get(target_pid, majority_train_label)
            else:
                pred_lbl = majority_train_label

            y_pred.append(pred_lbl)

        elapsed = time.time() - start_time
        acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
        f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0) if len(y_true) > 0 else 0.0
        f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0) if len(y_true) > 0 else 0.0

        per_class_f1 = {name: 0.0 for name in TARGET_NAMES}
        rep_dict = classification_report(y_true, y_pred, target_names=TARGET_NAMES, output_dict=True, zero_division=0)
        for name in TARGET_NAMES:
            if name in rep_dict:
                per_class_f1[name] = rep_dict[name].get('f1-score', 0.0)

        print("\n" + "=" * 55)
        print("          FOLDSEEK 1-NN BENCHMARK VÝSLEDKY        ")
        print("=" * 55)
        print(f"Testovaných proteinů (Queries):  {len(y_true)}")
        print(f"Nalezených Foldseek hitů:       {hit_counts} / {len(y_true)} ({hit_counts/max(len(y_true),1)*100:.1f}%)")
        print(f"Test Accuracy:                  {acc * 100:.2f} %")
        print(f"Test Macro F1:                  {f1_m * 100:.2f} %")
        print(f"Test Weighted F1:               {f1_w * 100:.2f} %")
        print(f"Čas běhu:                       {elapsed:.1f} s")
        print("-" * 55)
        print("Detailní klasifikační report:")
        print(classification_report(y_true, y_pred, target_names=TARGET_NAMES, zero_division=0))
        print("=" * 55 + "\n")

        return {
            "val_acc": None,
            "val_macro_f1": None,
            "test_acc": acc,
            "test_macro_f1": f1_m,
            "test_weighted_f1": f1_w,
            "per_class_f1": per_class_f1,
            "time_sec": round(elapsed, 1),
            "status": "SUCCESS"
        }


def main():
    parser = argparse.ArgumentParser(description="Foldseek 1-NN Benchmark (Query: Test split, Database: Train split)")
    parser.add_argument("--split-suffix", "--suffix", default="mil_0.5", help="Přípona split souborů (např. mil_0.5, struct_pocket_0.5_0.5)")
    parser.add_argument("--use-esm-split", action="store_true", help="Použít ESM embedding clustering split (_esm_0.2)")
    parser.add_argument("--use-nr", action="store_true", help="Použít Non-Redundant (NR) variantu splitu")
    parser.add_argument("--all-pdbs", action="store_true", help="Vyhodnotit všechny PDB na disku bez filtrace podle esm_dataset.pt")
    parser.add_argument("--threads", type=int, default=8, help="Počet výpočetních vláken pro Foldseek")
    parser.add_argument("--foldseek-bin", type=str, default=None, help="Vlastní cesta k binárce foldseek")
    parser.add_argument("--pdb-dir", type=str, default=None, help="Vlastní adresář s PDB soubory")
    args = parser.parse_args()

    if getattr(args, 'use_esm_split', False):
        args.split_suffix = 'esm_0.2'

    run_foldseek_benchmark(
        split_suffix=args.split_suffix,
        use_nr=args.use_nr,
        all_pdbs=args.all_pdbs,
        threads=args.threads,
        foldseek_bin=args.foldseek_bin,
        custom_pdb_dir=args.pdb_dir
    )


if __name__ == '__main__':
    main()
