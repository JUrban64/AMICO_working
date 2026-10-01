#!/usr/bin/env python3
"""
=============================================================================
AMICO: Benchmark nehomologních isofunkčních enzymů (NISE Benchmark)
=============================================================================

Tento skript demonstruje schopnost modelů AMICO generalizovat přes nehomologní
strukturní foldy (konvergentní evoluce) na datech z databáze NISE:
  1. Výběr reprezentativního vzorku (např. 15 proteinů na kofaktor = 75 celkem),
     kde enzymy se STEJNÝM kofaktorem a stejným EC číslem patří do RŮZNÝCH
     SCOP strukturních superfamilií (různé 3D foldy).
  2. Automatické stažení predikovaných 3D struktur z AlphaFold DB.
  3. Predikce vazebných kapes pomocí P2Rank.
  4. Extrakce ESM-2 embeddingů (pockets + full protein).
  5. Vyhodnocení cílových modelů:
       - Foldseek (1-NN 3D strukturní alignment baseline)
       - Sequence MLP (čistě sekvenční baseline nad ESM-2)
       - Self-Attention MIL (AMICO bez chemických queries)
       - Ligand-Cross Attention MIL (hlavní AMICO model s chemickými queries)
  6. Srovnávací report (Accuracy, Macro F1, čas inference, per-class metriky).
"""

import os
import sys
import glob
import json
import time
import shutil
import tempfile
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
import urllib.request

import numpy as np
import pandas as pd
import torch

try:
    from sklearn.metrics import accuracy_score, f1_score, classification_report
except ImportError:
    accuracy_score = None
    f1_score = None
    classification_report = None

# Automatické nastavení cesty k projektu
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) if os.path.basename(SCRIPT_DIR) == 'benchmarks' else SCRIPT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import (
    TARGET_NAMES,
    SequenceMLPClassifier,
    SelfAttentionMIL,
    LigandCrossAttentionMIL
)

try:
    from dataset import load_split_ids
except ImportError:
    load_split_ids = None


# ============================================================================
# KROK 1: VÝBĚR REPREZENTATIVNÍHO VZORKU Z NISE
# ============================================================================

def resolve_file_path(path, fallback_names=None):
    """Najde soubor na zadané cestě nebo v známých alternativních umístěních."""
    if path and os.path.exists(path):
        return os.path.abspath(path)
    
    candidates = []
    if path:
        candidates.extend([
            os.path.join(PROJECT_ROOT, path),
            os.path.join(PROJECT_ROOT, "data_prep", path),
            os.path.join(PROJECT_ROOT, "..", "AMICO_workign", path),
        ])
    if fallback_names:
        for f in fallback_names:
            candidates.extend([
                f,
                os.path.join(PROJECT_ROOT, f),
                os.path.join(PROJECT_ROOT, "data_prep", f),
                os.path.join(PROJECT_ROOT, "..", "AMICO_workign", f),
            ])
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return path


def select_nise_sample(nise_tsv_path, sample_per_cofactor=15, out_tsv="nise_selected_sample.tsv", force_resample=False):
    """
    Vybere vyvážený reprezentativní vzorek enzymů z NISE.
    Prioritizuje EC čísla s více různými SCOP superfamiliemi (true NISE).
    """
    if not force_resample and os.path.exists(out_tsv):
        print(f"📦 Načítám existující výběr vzorků z: {out_tsv}")
        return pd.read_csv(out_tsv, sep='\t')

    resolved_tsv = resolve_file_path(nise_tsv_path, ["NISE_amico_cofactors.tsv", "NISE.tsv"])
    if not os.path.exists(resolved_tsv):
        existing_sample = resolve_file_path("nise_selected_sample.tsv")
        if os.path.exists(existing_sample):
            print(f"📦 NISE TSV nenalezeno, ale nalezen připravený výběr: {existing_sample}")
            return pd.read_csv(existing_sample, sep='\t')
        raise FileNotFoundError(f"NISE dataset nenalezen: {nise_tsv_path} (hledáno i v okolních složkách)")

    print(f"\n📋 Načítám NISE dataset z: {resolved_tsv}")
    df = pd.read_csv(resolved_tsv, sep='\t')
    
    # Detekce EC čísel, která mají v rámci kofaktoru více než 1 superfamilii
    true_nise = df.groupby(['cofactor', 'ec'])['supfam'].nunique().reset_index()
    true_nise = true_nise[true_nise['supfam'] > 1]
    
    selected_groups = []
    for cof in TARGET_NAMES:
        cof_df = df[df['cofactor'] == cof].copy()
        if cof_df.empty:
            continue
            
        nise_ecs = true_nise[true_nise['cofactor'] == cof]['ec'].tolist()
        sub = cof_df[cof_df['ec'].isin(nise_ecs)].copy() if nise_ecs else cof_df.copy()
            
        representatives = sub.groupby(['ec', 'supfam']).first().reset_index()
        if len(representatives) > sample_per_cofactor:
            representatives = representatives.sample(sample_per_cofactor, random_state=42)
            
        if len(representatives) < sample_per_cofactor:
            needed = sample_per_cofactor - len(representatives)
            remaining = cof_df[~cof_df['entry'].isin(representatives['entry'])]
            if not remaining.empty:
                extra = remaining.sample(min(needed, len(remaining)), random_state=42)
                representatives = pd.concat([representatives, extra], ignore_index=True)
                
        selected_groups.append(representatives)

    sample_df = pd.concat(selected_groups, ignore_index=True)
    sample_df.to_csv(out_tsv, sep='\t', index=False)
    
    print(f"✅ Vybráno {len(sample_df)} vzorků napříč {sample_df['cofactor'].nunique()} kofaktory:")
    for cof in TARGET_NAMES:
        c_sub = sample_df[sample_df['cofactor'] == cof]
        n_ec = c_sub['ec'].nunique()
        n_sup = c_sub['supfam'].nunique()
        print(f"   - {cof:<12s}: {len(c_sub):>2d} proteinů | {n_ec:>2d} různých EC čísel | {n_sup:>2d} různých SCOP superfamilií (foldů)")
        
    print(f"💾 Seznam vzorků uložen do: {out_tsv}")
    return sample_df


# ============================================================================
# KROK 2: STAŽENÍ STRUKTUR Z ALPHAFOLD DB
# ============================================================================

def download_alphafold_structures(sample_df, structures_dir="nise_structures"):
    """
    Stáhne predikované 3D modely z AlphaFold DB (EBI) pro vybraná UniProt ID.
    """
    struct_root = Path(structures_dir)
    struct_root.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    cached = 0
    failed = 0
    
    print(f"\n🌐 Stahuji 3D struktury z AlphaFold DB (cílová složka: {struct_root.resolve()})...")
    downloaded_paths = {}
    
    for idx, row in sample_df.iterrows():
        uniprot_id = str(row['entry']).strip()
        cofactor = str(row['cofactor']).strip()
        cof_dir = struct_root / cofactor
        cof_dir.mkdir(parents=True, exist_ok=True)
        
        target_pdb = cof_dir / f"{uniprot_id}.pdb"
        
        if target_pdb.exists() and target_pdb.stat().st_size > 1000:
            cached += 1
            downloaded_paths[uniprot_id] = str(target_pdb.resolve())
            continue
            
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0 (AMICO-Benchmark)'})
        
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if isinstance(data, list) and len(data) > 0:
                    pdb_url = data[0].get('pdbUrl')
                    if pdb_url:
                        pdb_req = urllib.request.Request(pdb_url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(pdb_req, timeout=15) as pdb_resp:
                            content = pdb_resp.read()
                            with open(target_pdb, 'wb') as f_out:
                                f_out.write(content)
                        downloaded += 1
                        downloaded_paths[uniprot_id] = str(target_pdb.resolve())
                    else:
                        failed += 1
                else:
                    failed += 1
        except Exception:
            failed += 1
            
        time.sleep(0.12)
        if (idx + 1) % 15 == 0 or (idx + 1) == len(sample_df):
            print(f"   [{idx + 1:2d}/{len(sample_df)}] Staženo: {downloaded:2d} | Z cache: {cached:2d} | Selhalo: {failed:2d}")

    print(f"✅ Dokončeno stahování struktur: {len(downloaded_paths)} k dispozici (z celkových {len(sample_df)}).")
    return downloaded_paths


# ============================================================================
# KROK 3: PREDIKCE KAPES POMOCÍ P2RANK
# ============================================================================

def find_prank_executable(custom_path=None):
    """Najde spustitelný soubor P2Ranku."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    candidates = [
        "prank",
        os.path.join(PROJECT_ROOT, "p2rank_2.5.1", "prank"),
        os.path.join(PROJECT_ROOT, "p2rank", "prank"),
        os.path.join(PROJECT_ROOT, "..", "p2rank_2.5.1", "prank"),
    ]
    for c in candidates:
        if shutil.which(c):
            return c
        if os.path.exists(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    return None


def run_p2rank_for_structures(downloaded_paths, prank_exec=None, threads=6):
    """Spustí P2Rank predikci kapes na stažených strukturách."""
    prank_bin = find_prank_executable(prank_exec)
    if not prank_bin:
        print("⚠️ P2Rank spustitelný soubor nenalezen. Bude použit sekvenční full-protein embedding jako fallback.")
        return {}

    print(f"\n🔬 Spouštím P2Rank ({prank_bin}) pro predikci vazebných kapes...")
    to_process = []
    prank_dirs = {}
    
    for uid, pdb_str in downloaded_paths.items():
        pdb_p = Path(pdb_str)
        target_dir = pdb_p.parent / f"{pdb_p.stem}_prank_output"
        pred_csv = target_dir / f"{pdb_p.name}_predictions.csv"
        
        if target_dir.exists() and pred_csv.exists():
            prank_dirs[uid] = str(target_dir.resolve())
        else:
            to_process.append(pdb_p)

    if not to_process:
        print(f"⚡ Všechny P2Rank predikce již existují v cache ({len(prank_dirs)} hotovo).")
        return prank_dirs

    print(f"   Ke zpracování zbývá: {len(to_process)} struktur (Threads: {threads}).")
    temp_out = Path("./temp_prank_nise")
    temp_out.mkdir(parents=True, exist_ok=True)
    ds_file = Path("nise_batch.ds")
    
    with open(ds_file, "w") as f:
        for p in to_process:
            f.write(f"{p.resolve()}\n")

    cmd = [
        prank_bin, "predict",
        "-threads", str(threads),
        "-visualizations", "0",
        "-o", str(temp_out),
        str(ds_file)
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        for p in to_process:
            target_dir = p.parent / f"{p.stem}_prank_output"
            target_dir.mkdir(parents=True, exist_ok=True)
            for f in temp_out.glob(f"{p.name}*"):
                if f.is_file():
                    shutil.move(str(f), str(target_dir / f.name))
            prank_dirs[p.stem] = str(target_dir.resolve())
        print("✅ P2Rank úspěšně dokončil predikci kapes.")
    except Exception as e:
        print(f"⚠️ Chyba při běhu P2Ranku: {e}")
    finally:
        if ds_file.exists(): ds_file.unlink()
        if temp_out.exists(): shutil.rmtree(temp_out, ignore_errors=True)

    return prank_dirs


# ============================================================================
# KROK 4: EXTRAKCE ESM-2 EMBEDDINGŮ (POCKETS + FULL PROTEIN)
# ============================================================================

def parse_full_sequence_from_pdb(pdb_path):
    """Extrahuje sekvenci aminokyselin ze všech řetězců PDB souboru."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    try:
        struct = parser.get_structure('protein', pdb_path)
        seq = []
        for model in struct:
            for chain in model:
                for res in chain:
                    if res.get_id()[0] == ' ':
                        rname = res.get_resname()
                        seq.append(three_to_one.get(rname, 'X'))
        return ''.join(seq)
    except Exception:
        return None


def parse_pockets_from_prank(prank_dir, pdb_path):
    """Načte sekvence kapes z P2Rank výstupů."""
    import csv
    from Bio.PDB import PDBParser
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    pockets = []
    res_csvs = glob.glob(os.path.join(prank_dir, "*_residues.csv"))
    if res_csvs and os.path.exists(pdb_path):
        try:
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure('p', pdb_path)
            res_dict = {}
            for model in struct:
                for chain in model:
                    cid = chain.get_id().strip()
                    for r in chain:
                        if r.get_id()[0] == ' ':
                            rid = str(r.get_id()[1]).strip()
                            res_dict[(cid, rid)] = r.get_resname()
                            
            pocket_residues = defaultdict(list)
            with open(res_csvs[0], 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, skipinitialspace=True)
                for row in reader:
                    clean = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                    pnum = clean.get('pocket', '0')
                    if pnum != '0' and pnum != '':
                        c = clean.get('chain', '').strip()
                        rlabel = clean.get('residue_label', '').strip()
                        pocket_residues[pnum].append((c, rlabel))
                        
            for pnum in sorted(pocket_residues.keys(), key=lambda x: int(x) if x.isdigit() else 999):
                seq = []
                for c, rlabel in pocket_residues[pnum]:
                    key = (c, rlabel)
                    if key in res_dict:
                        seq.append(three_to_one.get(res_dict[key], 'X'))
                if seq:
                    pockets.append(''.join(seq))
        except Exception:
            pass
    return pockets


def extract_features_nise(downloaded_paths, prank_dirs, cache_path="nise_extracted_features.pt", device=None):
    """Extrahuje ESM-2 embeddingy pro NISE vzorky."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    features_dict = {}
    resolved_cache = resolve_file_path(cache_path, ["nise_extracted_features.pt"])
    if os.path.exists(resolved_cache):
        try:
            features_dict = torch.load(resolved_cache, map_location='cpu', weights_only=False)
            print(f"📦 Načteno {len(features_dict)} extrahovaných features z cache: {resolved_cache}")
        except Exception:
            features_dict = {}

    missing_uids = [uid for uid in downloaded_paths if uid not in features_dict and not str(downloaded_paths[uid]).startswith("cached://")]
    
    if missing_uids:
        print(f"\n⚡ Extrahuji ESM-2 embeddingy pro {len(missing_uids)} struktur (facebook/esm2_t33_650M_UR50D)...")
        from transformers import AutoTokenizer, EsmModel
        model_name = "facebook/esm2_t33_650M_UR50D"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        esm_model = EsmModel.from_pretrained(model_name).to(device)
        esm_model.eval()

        def get_emb(seq_str):
            if not seq_str: return None
            inputs = tokenizer(seq_str, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = esm_model(**inputs)
            emb = out.last_hidden_state[0, 1:-1, :]
            return emb.mean(dim=0).cpu()

        for idx, uid in enumerate(missing_uids):
            pdb_path = downloaded_paths[uid]
            pdir = prank_dirs.get(uid)
            
            full_seq = parse_full_sequence_from_pdb(pdb_path)
            if not full_seq: continue
            full_emb = get_emb(full_seq)
            if full_emb is None: continue
            
            pocket_seqs = parse_pockets_from_prank(pdir, pdb_path) if pdir else []
            pocket_embs = [get_emb(s) for s in pocket_seqs if get_emb(s) is not None]
            
            if not pocket_embs:
                pocket_tensor = full_emb.unsqueeze(0)
            else:
                pocket_tensor = torch.stack(pocket_embs)
                
            features_dict[uid] = {
                'pocket_features': pocket_tensor,
                'full_protein_feature': full_emb
            }
            if (idx + 1) % 15 == 0 or (idx + 1) == len(missing_uids):
                print(f"   [{idx + 1:2d}/{len(missing_uids)}] Dokončeno...")

        torch.save(features_dict, cache_path)
        print(f"💾 Features uloženy do cache: {cache_path}")

    return features_dict


# ============================================================================
# KROK 5: FOLDSEEK 1-NN STRUKTURNÍ BENCHMARK
# ============================================================================

def find_foldseek_binary(custom_bin=None):
    """Vyhledá spustitelný soubor foldseek."""
    if custom_bin and os.path.exists(custom_bin) and os.access(custom_bin, os.X_OK):
        return custom_bin
    which_path = shutil.which("foldseek")
    if which_path:
        return which_path
    home = os.path.expanduser("~")
    for p in [
        os.path.join(home, "miniforge3", "envs", "foldseek", "bin", "foldseek"),
        os.path.join(home, "anaconda3", "envs", "foldseek", "bin", "foldseek"),
        os.path.join(home, "miniconda3", "envs", "foldseek", "bin", "foldseek"),
        "/opt/homebrew/bin/foldseek",
        "/usr/local/bin/foldseek"
    ]:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p
    return None


def run_foldseek_benchmark(downloaded_paths, sample_df, train_dir=None, split_suffix="mil_0.5", use_nr=True, threads=8):
    """Spustí Foldseek 1-NN zarovnání testovacích NISE struktur proti trénovací sadě."""
    foldseek_bin = find_foldseek_binary()
    if foldseek_bin is None:
        print("⚠️ Foldseek binárka nenalezena. Přeskakuji Foldseek benchmark.")
        return None

    # Autodetekce train_dir
    train_dir_candidates = [
        train_dir,
        os.path.join(PROJECT_ROOT, "structures"),
        os.path.join(PROJECT_ROOT, "data_prep", "structures"),
        os.path.join(PROJECT_ROOT, "..", "AMICO_workign", "structures"),
        os.path.join(PROJECT_ROOT, "..", "AMICO_workign", "data_prep", "structures")
    ]
    resolved_train_dir = next((d for d in train_dir_candidates if d and os.path.exists(d)), None)
    if not resolved_train_dir:
        print("⚠️ Adresář se strukturami pro Foldseek nenalezen. Přeskakuji.")
        return None

    print(f"\n🔍 Spouštím Foldseek 1-NN zarovnání ({foldseek_bin})...")
    print(f"   Query set (NISE): {len(downloaded_paths)} struktur")
    print(f"   Target databáze (AMICO Train): {resolved_train_dir}")

    # Filtr podle train splitu
    train_ids = None
    if split_suffix and load_split_ids is not None:
        try:
            tr_set, _, _ = load_split_ids(PROJECT_ROOT, split_suffix=split_suffix, use_nr=use_nr)
            if not tr_set and os.path.exists(os.path.join(PROJECT_ROOT, '..', 'AMICO_workign')):
                tr_set, _, _ = load_split_ids(os.path.join(PROJECT_ROOT, '..', 'AMICO_workign'), split_suffix=split_suffix, use_nr=use_nr)
            if tr_set:
                train_ids = {os.path.basename(pid).lower().replace('clean_', '').replace('.pdb', '').replace('_merged', '') for pid in tr_set}
                print(f"   🔒 Aplikován filtr trénovacího splitu ({split_suffix}, use_nr={use_nr}): {len(train_ids)} povolených proteinů.")
        except Exception as e:
            print(f"   ⚠️ Nepodařilo se načíst split {split_suffix}: {e}")

    # Mapování target PDB na kofaktory
    train_pdbs = glob.glob(os.path.join(resolved_train_dir, "**", "*.pdb"), recursive=True)
    target_labels = {}
    for p in train_pdbs:
        t_id = os.path.basename(p).replace('.pdb', '').replace('clean_', '').replace('_merged', '').lower()
        if t_id not in target_labels:
            for cname in TARGET_NAMES:
                if cname.lower() in p.lower() or cname.upper() in p.upper():
                    target_labels[t_id] = TARGET_NAMES.index(cname)
                    break

    with tempfile.TemporaryDirectory() as tmp_dir:
        q_dir = os.path.join(tmp_dir, "queries")
        fs_tmp = os.path.join(tmp_dir, "fs_tmp")
        os.makedirs(q_dir, exist_ok=True)
        os.makedirs(fs_tmp, exist_ok=True)
        
        for uid, p in downloaded_paths.items():
            if str(p).startswith("cached://"): continue
            dst = os.path.join(q_dir, f"{uid}.pdb")
            abs_p = os.path.abspath(p)
            if os.path.exists(abs_p):
                shutil.copy2(abs_p, dst)

        out_tsv = os.path.join(tmp_dir, "aln.tsv")
        cmd = [
            foldseek_bin, "easy-search",
            q_dir, resolved_train_dir, out_tsv, fs_tmp,
            "--format-output", "query,target,evalue,qtmscore,bits",
            "-e", "10.0",
            "--max-seqs", "2000",
            "--threads", str(threads)
        ]
        
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed_sec = time.time() - t0
        except Exception as e:
            print(f"❌ Chyba při spuštění Foldseeku: {e}")
            return None

        if not os.path.exists(out_tsv) or os.path.getsize(out_tsv) == 0:
            print("⚠️ Foldseek nevygeneroval žádné výstupní zarovnání.")
            return None

        df = pd.read_csv(out_tsv, sep='\t', header=None, names=["query", "target", "evalue", "qtmscore", "bits"])
        df['query'] = df['query'].apply(lambda x: os.path.basename(str(x)).replace('.pdb', '').lower())
        df['target'] = df['target'].apply(lambda x: os.path.basename(str(x)).replace('.pdb', '').replace('clean_', '').replace('_merged', '').lower())
        
        if train_ids is not None:
            df = df[df['target'].isin(train_ids)]
            
        top1_preds = {}
        if not df.empty:
            df = df.sort_values(by=['query', 'qtmscore'], ascending=[True, False])
            top1 = df.drop_duplicates(subset=['query'], keep='first')
            for _, r in top1.iterrows():
                q, t = r['query'], r['target']
                if t in target_labels:
                    top1_preds[q] = target_labels[t]
                    
        y_true, y_pred = [], []
        preds_by_uid = {}
        for _, row in sample_df.iterrows():
            uid = str(row['entry']).strip()
            if uid not in downloaded_paths: continue
            true_l = TARGET_NAMES.index(row['cofactor'])
            y_true.append(true_l)
            pred_l = top1_preds.get(uid.lower(), -1)
            y_pred.append(pred_l)
            preds_by_uid[uid] = pred_l
            
        return {
            'y_true': y_true, 
            'y_pred': y_pred, 
            'preds_by_uid': preds_by_uid,
            'time_sec': elapsed_sec
        }


# ============================================================================
# KROK 6: INFERENCE AMICO MODELŮ
# ============================================================================

def evaluate_amico_models(sample_df, downloaded_paths, features_dict, models_dir, device, custom_ckpts=None, target_models=None):
    """
    Spustí inferenci 3 podporovaných AMICO modelů:
      1. sequence_mlp (SequenceMLPClassifier)
      2. self_attention_mil (SelfAttentionMIL)
      3. ligand_cross_mil (LigandCrossAttentionMIL)
    """
    if custom_ckpts is None:
        custom_ckpts = {}

    all_models_config = [
        ("sequence_mlp", custom_ckpts.get("sequence_mlp") or [
            "sequence_mlp_best.pt", "model_sequence_mlp_best.pt", "seq_mlp_best.pt"
        ]),
        ("self_attention_mil", custom_ckpts.get("self_attention_mil") or [
            "self_attention_mil_best.pt", "model_self_attention_mil_best.pt"
        ]),
        ("ligand_cross_mil", custom_ckpts.get("ligand_cross_mil") or [
            "ligand_cross_mil_best.pt", "ligand_cross_attention_mil_best.pt", "model_ligand_cross_attention_mil_best.pt"
        ])
    ]
    
    if target_models:
        target_set = {m.lower().strip() for m in target_models}
        models_to_test = [item for item in all_models_config if item[0] in target_set]
    else:
        models_to_test = all_models_config

    results = {}
    valid_samples = [row for _, row in sample_df.iterrows() if str(row['entry']).strip() in features_dict]
    print(f"\n🧠 Spouštím inferenci AMICO modelů pro {len(valid_samples)} NISE struktur...")
    
    for mkey, ckpt_candidates in models_to_test:
        if isinstance(ckpt_candidates, str):
            ckpt_candidates = [ckpt_candidates]
            
        ckpt_path = None
        for cand in ckpt_candidates:
            if os.path.isabs(cand) and os.path.exists(cand):
                ckpt_path = cand
                break
            search_dirs = [models_dir, PROJECT_ROOT, os.path.join(PROJECT_ROOT, "..", "AMICO_workign")]
            for sdir in search_dirs:
                p = os.path.join(sdir, cand)
                if os.path.exists(p):
                    ckpt_path = os.path.abspath(p)
                    break
            if ckpt_path:
                break
            if os.path.exists(cand):
                ckpt_path = os.path.abspath(cand)
                break
                
        if not ckpt_path:
            cand_str = ckpt_candidates[0] if ckpt_candidates else "checkpoint"
            print(f"⚠️ Checkpoint pro {mkey} ({cand_str}) nenalezen, přeskakuji {mkey}.")
            continue
            
        print(f"   Načítám model {mkey} z: {ckpt_path}")
        
        # Inicializace přímo ze tříd z model.py
        if mkey == "sequence_mlp":
            model = SequenceMLPClassifier(in_features=1280, hidden_dim=256, num_classes=5, dropout=0.3)
        elif mkey == "self_attention_mil":
            model = SelfAttentionMIL(feature_dim=1280, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.2)
        elif mkey == "ligand_cross_mil":
            model = LigandCrossAttentionMIL(feature_dim=1280, ecfp_dim=1024, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.2)
        else:
            continue

        raw_state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(raw_state, dict):
            if "model_state_dict" in raw_state:
                raw_state = raw_state["model_state_dict"]
            elif "state_dict" in raw_state:
                raw_state = raw_state["state_dict"]
            elif "model" in raw_state:
                raw_state = raw_state["model"]
                
        model.load_state_dict(raw_state)
        model = model.to(device)
        model.eval()
        
        y_true, y_pred = [], []
        preds_by_uid = {}
        
        t0 = time.time()
        with torch.no_grad():
            for row in valid_samples:
                uid = str(row['entry']).strip()
                true_lbl = TARGET_NAMES.index(row['cofactor'])
                feat_data = features_dict[uid]
                
                full_feat = feat_data['full_protein_feature'].unsqueeze(0).to(device)
                
                if mkey == "sequence_mlp":
                    logits = model(full_feat)
                else:
                    p_feat = feat_data['pocket_features'].unsqueeze(0).to(device)
                    mask = torch.zeros(1, p_feat.size(1), dtype=torch.bool, device=device)
                    out = model(p_feat, mask, full_feat)
                    logits = out[0] if isinstance(out, tuple) else out
                
                pred_lbl = torch.argmax(logits, dim=-1).item()
                y_true.append(true_lbl)
                y_pred.append(pred_lbl)
                preds_by_uid[uid] = pred_lbl
                
        elapsed_sec = time.time() - t0
        results[mkey] = {
            'y_true': y_true, 
            'y_pred': y_pred, 
            'preds_by_uid': preds_by_uid,
            'time_sec': elapsed_sec
        }
        
    return results


# ============================================================================
# KROK 7: STATISTICKÉ VYHODNOCENÍ A REPORT
# ============================================================================

def df_to_markdown_simple(df):
    """Převede DataFrame na Markdown tabulku."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[h]) for h in df.columns) + " |")
        return "\n".join(lines)


def generate_nise_report(all_results, sample_df, out_prefix="nise_benchmark", merge_existing=True):
    """Vypíše přehledné výsledky benchmarku na NISE nehomologních enzymech."""
    print("\n" + "=" * 135)
    print("                VÝSLEDKY NISE BENCHMARKU (NEHOMOLOGNÍ STRUKTURNÍ GENERALIZACE)                ")
    print("=" * 135)
    print(f"{'Model':<24s} | {'Vzorků':<7s} | {'Accuracy':<10s} | {'Macro F1':<10s} | {'Čas (s)':<9s} | {'ms/vzorek':<10s} | {'acetyl-CoA':<10s} | {'ATP':<8s} | {'B12':<8s} | {'FAD':<8s} | {'NAD':<8s}")
    print("-" * 135)

    summary_rows_dict = {}

    results_csv = f"{out_prefix}_results.csv"
    if merge_existing and os.path.exists(results_csv):
        try:
            prev_df = pd.read_csv(results_csv)
            for _, r in prev_df.iterrows():
                summary_rows_dict[str(r['Model']).strip()] = r.to_dict()
        except Exception as e:
            print(f"⚠️ Nelze načíst předchozí výsledky z {results_csv}: {e}")

    for mkey, res in all_results.items():
        if not res: continue
        y_t = res['y_true']
        y_p = res['y_pred']
        t_sec = res.get('time_sec', 0.0)
        ms_per_sample = (t_sec / max(len(y_t), 1)) * 1000.0

        if accuracy_score is not None:
            acc = accuracy_score(y_t, y_p)
            f1_macro = f1_score(y_t, y_p, average='macro', zero_division=0)
            f1_weighted = f1_score(y_t, y_p, average='weighted', zero_division=0)
            rep = classification_report(y_t, y_p, target_names=TARGET_NAMES, labels=list(range(5)), output_dict=True, zero_division=0)
            per_class = {c: rep.get(c, {}).get('f1-score', 0.0) * 100 for c in TARGET_NAMES}
        else:
            total = len(y_t)
            correct = sum(1 for yt, yp in zip(y_t, y_p) if yt == yp)
            acc = correct / max(total, 1)
            per_class = {}
            f1_list = []
            for i, name in enumerate(TARGET_NAMES):
                tp = sum(1 for yt, yp in zip(y_t, y_p) if yt == i and yp == i)
                fp = sum(1 for yt, yp in zip(y_t, y_p) if yt != i and yp == i)
                fn = sum(1 for yt, yp in zip(y_t, y_p) if yt == i and yp != i)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                per_class[name] = f1 * 100.0
                f1_list.append(f1)
            f1_macro = sum(f1_list) / len(f1_list) if f1_list else 0.0
            f1_weighted = f1_macro

        summary_rows_dict[mkey] = {
            'Model': mkey,
            'Total_Samples': len(y_t),
            'Accuracy': round(acc * 100, 2),
            'Macro_F1': round(f1_macro * 100, 2),
            'Weighted_F1': round(f1_weighted * 100, 2),
            'Time_Sec': round(t_sec, 2),
            'ms_per_sample': round(ms_per_sample, 1),
            'F1_acetyl-CoA': round(per_class['acetyl-CoA'], 2),
            'F1_ATP': round(per_class['ATP'], 2),
            'F1_B12': round(per_class['B12'], 2),
            'F1_FAD': round(per_class['FAD'], 2),
            'F1_NAD': round(per_class['NAD'], 2)
        }

    # Čtyři hlavní modely
    CANONICAL_ORDER = [
        "foldseek_1nn",
        "sequence_mlp",
        "self_attention_mil",
        "ligand_cross_mil"
    ]
    def model_sort_key(m):
        return CANONICAL_ORDER.index(m) if m in CANONICAL_ORDER else 999

    sorted_models = sorted(summary_rows_dict.keys(), key=model_sort_key)
    summary_rows = [summary_rows_dict[m] for m in sorted_models]

    for row_dict in summary_rows:
        mkey = row_dict['Model']
        n_samp = int(row_dict.get('Total_Samples', 0))
        acc_v = float(row_dict.get('Accuracy', 0.0))
        f1_m_v = float(row_dict.get('Macro_F1', 0.0))
        t_sec_v = float(row_dict.get('Time_Sec', 0.0)) if pd.notnull(row_dict.get('Time_Sec')) else 0.0
        ms_v = float(row_dict.get('ms_per_sample', 0.0)) if pd.notnull(row_dict.get('ms_per_sample')) else 0.0
        t_str = f"{t_sec_v:6.2f} s" if t_sec_v > 0 else "   N/A  "
        ms_str = f"{ms_v:7.1f} ms" if ms_v > 0 else "   N/A   "
        f1_ac = float(row_dict.get('F1_acetyl-CoA', 0.0))
        f1_atp = float(row_dict.get('F1_ATP', 0.0))
        f1_b12 = float(row_dict.get('F1_B12', 0.0))
        f1_fad = float(row_dict.get('F1_FAD', 0.0))
        f1_nad = float(row_dict.get('F1_NAD', 0.0))
        print(f"{mkey:<24s} | {n_samp:<7d} | {acc_v:6.2f} %  | {f1_m_v:6.2f} %  | {t_str:<9s} | {ms_str:<10s} | {f1_ac:6.1f} %   | {f1_atp:6.1f} % | {f1_b12:6.1f} % | {f1_fad:6.1f} % | {f1_nad:6.1f} %")

    print("=" * 135 + "\n")

    # Detailní tabulka po jednotlivých proteinech
    detailed_csv = f"{out_prefix}_detailed.csv"
    existing_det_map = {}
    if merge_existing and os.path.exists(detailed_csv):
        try:
            prev_det = pd.read_csv(detailed_csv)
            for _, r in prev_det.iterrows():
                existing_det_map[str(r['UniProt_ID']).strip()] = r.to_dict()
        except Exception:
            existing_det_map = {}

    det_rows = []
    for _, row in sample_df.iterrows():
        uid = str(row['entry']).strip()
        r_item = existing_det_map.get(uid, {
            'UniProt_ID': uid,
            'Cofactor_GroundTruth': row['cofactor'],
            'EC_Number': row['ec'],
            'SCOP_Superfamily': row['supfam'],
            'Protein_Name': str(row['protein'])[:50]
        })
        for mkey, res in all_results.items():
            if res and 'preds_by_uid' in res:
                p_idx = res['preds_by_uid'].get(uid, -1)
                p_name = TARGET_NAMES[p_idx] if 0 <= p_idx < 5 else "Miss (-1)"
                r_item[f"{mkey}_Pred"] = p_name
                r_item[f"{mkey}_Correct"] = (p_name == row['cofactor'])
        det_rows.append(r_item)

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(f"{out_prefix}_results.csv", index=False)

    det_df = pd.DataFrame(det_rows)
    det_df.to_csv(f"{out_prefix}_detailed.csv", index=False)

    # Markdown report
    with open(f"{out_prefix}_report.md", "w", encoding="utf-8") as f:
        f.write("# AMICO: Benchmark Nehomologních Isofunkčních Enzymů (NISE)\n\n")
        f.write(f"Vygenerováno: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Dataset obsahuje nehomologní enzymy se stejnou funkcí a kofaktorem, ale z odlišných strukturních superfamilií (různé 3D foldy).\n\n")
        f.write("### Srovnávané modely:\n")
        f.write("- **foldseek_1nn**: 3D strukturní alignment 1-NN baseline\n")
        f.write("- **sequence_mlp**: Čistě sekvenční ESM-2 baseline model\n")
        f.write("- **self_attention_mil**: Self-Attention MIL nad kapsami a celým proteinem\n")
        f.write("- **ligand_cross_mil**: Chemický Ligand Cross-Attention MIL model (AMICO)\n\n")
        f.write("### Souhrnné výsledky\n\n")
        f.write(df_to_markdown_simple(sum_df))
        f.write("\n\n")
        f.write("### Detailní ukázka z predikcí (prvních 15 vzorků)\n\n")
        f.write(df_to_markdown_simple(det_df.head(15)))
        f.write("\n")

    print(f"💾 Výstupy úspěšně uloženy:")
    print(f" - CSV souhrn: {out_prefix}_results.csv")
    print(f" - CSV detail: {out_prefix}_detailed.csv")
    print(f" - Markdown:   {out_prefix}_report.md\n")


# ============================================================================
# HLAVNÍ SPOUŠTĚCÍ FUNKCE (MAIN)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="NISE Benchmark pro Foldseek, Sequence MLP, Self-Attention MIL a Ligand Cross-Attention MIL.")
    parser.add_argument("--nise-tsv", default="NISE_amico_cofactors.tsv", help="Cesta k NISE TSV souboru.")
    parser.add_argument("--sample-per-cofactor", type=int, default=15, help="Počet vzorků na kofaktor (default: 15, celkem 75 proteinů).")
    parser.add_argument("--force-resample", action="store_true", help="Vynutí nový náhodný výběr vzorků z NISE TSV.")
    parser.add_argument("--structures-dir", default="nise_structures", help="Složka pro stažené AlphaFold PDB struktury.")
    parser.add_argument("--models-dir", default=PROJECT_ROOT, help="Složka s váhami modelů (*_best.pt).")
    parser.add_argument("--models", nargs="+", default=["foldseek", "sequence_mlp", "self_attention_mil", "ligand_cross_mil"],
                        help="Modely k vyhodnocení: foldseek, sequence_mlp, self_attention_mil, ligand_cross_mil.")
    parser.add_argument("--sequence-mlp-ckpt", default=None, help="Vlastní cesta k vahám Sequence MLP (např. sequence_mlp_best.pt).")
    parser.add_argument("--self-attn-ckpt", default=None, help="Vlastní cesta k vahám Self-Attention MIL (např. self_attention_mil_best.pt).")
    parser.add_argument("--ligand-cross-ckpt", default=None, help="Vlastní cesta k vahám Ligand Cross-Attention MIL (např. ligand_cross_mil_best.pt).")
    parser.add_argument("--train-dir", default=None, help="Trénovací struktury pro Foldseek.")
    parser.add_argument("--split-suffix", default="mil_0.5", help="Přípona splitu pro trénovací sadu Foldseeku (např. mil_0.5).")
    parser.add_argument("--use-nr", action="store_true", default=True, help="Použít Non-Redundant variantu trénovací sady.")
    parser.add_argument("--no-nr", dest="use_nr", action="store_false", help="Vypnout Non-Redundant filtr.")
    parser.add_argument("--no-split-filter", action="store_true", help="Vypnout filtr splitů pro Foldseek.")
    parser.add_argument("--prank-exec", default=None, help="Cesta k binárce P2Rank (prank).")
    parser.add_argument("--out-prefix", default="nise_benchmark", help="Prefix výstupních souborů.")
    parser.add_argument("--no-merge-existing", dest="merge_existing", action="store_false", default=True, help="Vypnout sloučení s existujícími výsledky z CSV.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--skip-download", action="store_true", help="Přeskočí stahování struktur z AFDB.")
    parser.add_argument("--skip-foldseek", action="store_true", help="Přeskočí Foldseek benchmark.")
    
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("      AMICO: NISE BENCHMARK (Foldseek + Sequence MLP + Self-Attn + Ligand-Cross)      ")
    print("=" * 80)
    print(f"Zařízení: {device} | Vzorků na kofaktor: {args.sample_per_cofactor}")
    print(f"Vybrané modely: {args.models}")

    # 1. Výběr vzorku
    sample_df = select_nise_sample(
        args.nise_tsv,
        sample_per_cofactor=args.sample_per_cofactor,
        force_resample=args.force_resample
    )

    # 2. Stažení struktur
    downloaded_paths = {}
    if not args.skip_download:
        downloaded_paths = download_alphafold_structures(sample_df, structures_dir=args.structures_dir)
    else:
        for _, row in sample_df.iterrows():
            uid = str(row['entry']).strip()
            cof = str(row['cofactor']).strip()
            p = os.path.abspath(os.path.join(args.structures_dir, cof, f"{uid}.pdb"))
            if os.path.exists(p):
                downloaded_paths[uid] = p

    # Fallback na features cache
    features_cache = resolve_file_path("nise_extracted_features.pt")
    if not downloaded_paths and features_cache and os.path.exists(features_cache):
        print(f"📦 Žádné lokální PDB nenalezeny, ale nalezena cache features ({features_cache}). Pokračuji v neuronové inferenci.")
        try:
            cached_f = torch.load(features_cache, map_location='cpu', weights_only=False)
            downloaded_paths = {uid: f"cached://{uid}" for uid in cached_f}
        except Exception:
            pass

    if not downloaded_paths:
        print("❌ Žádné PDB struktury nebyly staženy ani nalezeny v cache.")
        sys.exit(1)

    # Normalizace jmen vybraných modelů
    selected_models = set(m.lower().strip() for m in args.models)
    needs_neural = any(m in selected_models for m in ["sequence_mlp", "self_attention_mil", "ligand_cross_mil", "self_att", "ligand_cross", "ligandcross"])
    needs_pockets = any(m in selected_models for m in ["self_attention_mil", "ligand_cross_mil", "self_att", "ligand_cross", "ligandcross"])

    # 3. P2Rank kapsy (pokud je potřeba pro Self-Attn nebo Ligand-Cross)
    prank_dirs = {}
    if needs_pockets and not any(str(p).startswith("cached://") for p in downloaded_paths.values()):
        prank_dirs = run_p2rank_for_structures(downloaded_paths, prank_exec=args.prank_exec, threads=args.threads)

    # 4. ESM-2 extrakce
    features_dict = {}
    if needs_neural:
        features_dict = extract_features_nise(downloaded_paths, prank_dirs, device=device)

    all_results = {}

    # 5. Foldseek 1-NN
    run_fs = not args.skip_foldseek and ("foldseek" in selected_models or "foldseek_1nn" in selected_models)
    if run_fs and not any(str(p).startswith("cached://") for p in downloaded_paths.values()):
        sfx = None if args.no_split_filter else args.split_suffix
        fs_res = run_foldseek_benchmark(
            downloaded_paths, sample_df, 
            train_dir=args.train_dir, 
            split_suffix=sfx,
            use_nr=args.use_nr,
            threads=args.threads
        )
        if fs_res:
            all_results['foldseek_1nn'] = fs_res

    # 6. AMICO modely (Sequence MLP, Self-Attention, Ligand-Cross Attention)
    custom_ckpts = {}
    if args.sequence_mlp_ckpt:
        custom_ckpts["sequence_mlp"] = args.sequence_mlp_ckpt
    if args.self_attn_ckpt:
        custom_ckpts["self_attention_mil"] = args.self_attn_ckpt
    if args.ligand_cross_ckpt:
        custom_ckpts["ligand_cross_mil"] = args.ligand_cross_ckpt

    # Mapování zadaných jmen na kanonické klíče
    model_alias_map = {
        "sequence_mlp": "sequence_mlp",
        "seq_mlp": "sequence_mlp",
        "self_attention_mil": "self_attention_mil",
        "self_attn": "self_attention_mil",
        "self_att": "self_attention_mil",
        "ligand_cross_mil": "ligand_cross_mil",
        "ligand_cross": "ligand_cross_mil",
        "ligandcross": "ligand_cross_mil",
    }
    target_amico = [model_alias_map[m] for m in selected_models if m in model_alias_map]

    if target_amico and features_dict:
        amico_res = evaluate_amico_models(
            sample_df, downloaded_paths, features_dict, 
            models_dir=args.models_dir, device=device,
            custom_ckpts=custom_ckpts,
            target_models=target_amico
        )
        all_results.update(amico_res)

    # 7. Vyhodnocení a report
    if all_results or (args.merge_existing and os.path.exists(f"{args.out_prefix}_results.csv")):
        generate_nise_report(all_results, sample_df, out_prefix=args.out_prefix, merge_existing=args.merge_existing)
    else:
        print("⚠️ Žádný model nebyl vyhodnocen.")


if __name__ == '__main__':
    main()
