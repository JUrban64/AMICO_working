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
    """Checks whether a residue is a standard amino acid."""
    return residue.get_id()[0] == ' '


def find_p2rank_executable(custom_path=None):
    """
    Locates the P2Rank executable (prank).
    Checks the provided path, PATH environment variable, and common project locations.
    """
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))

    # Standard locations within the project workspace and PATH
    candidates.extend([
        Path("p2rank_2.5.1/prank"),
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
    Runs P2Rank on a given PDB file and returns the path to the output directory.
    
    Args:
        pdb_path: Path to the target PDB file.
        prank_exec: Path to prank binary (if None, attempts auto-detection).
        output_dir: Directory to save prediction outputs (default: ./temp_p2rank/<stem>_prank_output).
        config: P2Rank configuration profile ('alphafold' or 'default').
        
    Returns:
        Path: Output directory path.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    if output_dir is None:
        output_dir = Path("./temp_p2rank") / f"{pdb_path.stem}_prank_output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    existing_preds = list(output_dir.glob("*_predictions.csv")) + list(output_dir.glob("*.csv"))
    if existing_preds:
        print(f"-> Found existing P2Rank outputs in {output_dir} (skipping re-run).")
        return output_dir

    executable = find_p2rank_executable(prank_exec)

    cmd = [
        executable, "predict",
        "-c", config,
        "-f", str(pdb_path.resolve()),
        "-o", str(output_dir.resolve()),
        "-visualizations", "0"
    ]

    print(f"-> Executing P2Rank on {pdb_path.name}...")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Error during P2Rank execution:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        ) from e
    except FileNotFoundError:
        raise FileNotFoundError(
            f"P2Rank executable not found at '{executable}'. "
            f"Please specify the correct path using --prank."
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
    Runs P2Rank in high-efficiency batch mode (-l dataset.ds) for training sets.
    
    Advantages for large dataset training:
      1. Starts the JVM and loads random forest models only ONCE.
      2. Uses multi-threading across specified CPU cores (-threads).
      3. Automatically skips previously processed structures (resume support).
      4. Splits structures into manageable chunks (e.g. 500 files).
    
    Args:
        pdb_paths: List of file paths to PDB structures.
        prank_exec: Path to prank binary.
        output_dir: Destination directory.
        config: P2Rank configuration ('alphafold' or 'default').
        threads: Number of CPU threads.
        chunk_size: Batch chunk size.
        
    Returns:
        Path: Output directory path.
    """
    if output_dir is None:
        output_dir = Path("./structures/p2rank_outputs")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    executable = find_p2rank_executable(prank_exec)

    # 1. Filter out already processed structures for resume support
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

    print(f"P2Rank Batch Mode: Total {len(pdb_paths)} structures.")
    if skipped_count > 0:
        print(f"Skipped (already predicted): {skipped_count} structures.")
    print(f"Remaining to process: {len(to_process)} structures (threads: {threads}, chunk size: {chunk_size}).")

    if not to_process:
        print("All structures have already been processed.")
        return output_dir

    # 2. Process in chunks
    total_chunks = (len(to_process) + chunk_size - 1) // chunk_size
    ds_temp_dir = output_dir / "temp_datasets"
    ds_temp_dir.mkdir(parents=True, exist_ok=True)

    for chunk_idx in range(total_chunks):
        chunk_files = to_process[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        ds_file = ds_temp_dir / f"batch_chunk_{chunk_idx + 1:04d}.ds"
        
        with open(ds_file, "w", encoding="utf-8") as f:
            for c_path in chunk_files:
                f.write(f"{str(c_path)}\n")

        print(f"\n[Batch Chunk {chunk_idx + 1}/{total_chunks}] Running P2Rank on {len(chunk_files)} structures...")
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
            print(f"Error during batch chunk #{chunk_idx + 1}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"P2Rank executable not found at '{executable}'.")
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

    print(f"P2Rank batch pocket prediction completed. Results saved in {output_dir}")
    return output_dir


def get_full_sequence_from_pdb(pdb_path):
    """
    Extracts full amino acid sequence and residue metadata from a PDB file.
    
    Returns:
        tuple: (seq_str, pdb_residues_dict, structure)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(pdb_path))
    
    sequence = []
    pdb_residues = {}
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
    Parses P2Rank prediction outputs (_predictions.csv and _residues.csv) for a protein.
    Supports single prediction directories as well as shared batch folders.
    """
    prank_dir = Path(prank_output_dir)
    pdb_path = Path(pdb_path)
    
    full_seq, pdb_residues, structure = get_full_sequence_from_pdb(pdb_path)
    if not full_seq:
        raise ValueError(f"Could not extract amino acid sequence from {pdb_path}.")
    
    all_residues = pdb_residues.get('_all_residues_list', [])

    stem = pdb_path.stem
    fname = pdb_path.name

    sub_dir = prank_dir / f"{stem}_prank_output"
    search_dirs = [prank_dir]
    if sub_dir.exists():
        search_dirs.insert(0, sub_dir)

    # 1. Search for _predictions.csv specific to target protein
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

    if not pred_csv:
        pred_candidates = list(prank_dir.glob(f"{stem}*_predictions.csv")) + list(prank_dir.glob(f"{stem}*.csv"))
        if not pred_candidates:
            pred_candidates = list(prank_dir.glob("*_predictions.csv"))
        if pred_candidates:
            pred_csv = pred_candidates[0]

    # 2. Search for _residues.csv specific to target protein
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

                m = re.search(r'(\d+)', name)
                pocket_id = int(m.group(1)) if m else rank

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

                    # Extract residues from 'residue_ids' column if present
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

    # 3. Read/supplement residues from _residues.csv
    if res_csv and res_csv.exists():
        with open(res_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                chain_id = clean_row.get('chain', clean_row.get('chain_id', '')).strip()
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

    # 4. Fallback: Check for physical *_pocket_*.pdb files if CSV parsing produced no pockets
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

    # 5. Build ordered sequences for each pocket
    pocket_list = []
    for pid in sorted(pockets_dict.keys()):
        p_data = pockets_dict[pid]
        
        # Sort residues according to primary sequence order (N -> C terminus)
        if p_data['residues']:
            p_data['residues'].sort(key=lambda r: (r.get('chain_id', ''), r.get('seq_idx', 0)))
            p_data['sequence'] = ''.join([r['one_letter'] for r in p_data['residues']])
        
        # Spatial fallback: If sequence is empty, query residues within 8.5 Å from center
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
    parser = argparse.ArgumentParser(description="P2Rank Pocket Prediction CLI (Single & Batch Mode)")
    parser.add_argument("--pdb", type=str, default=None, help="Path to single PDB file")
    parser.add_argument("--batch-dir", type=str, default=None, help="Path to folder containing PDB structures")
    parser.add_argument("--metadata", type=str, default=None, help="Path to dataset_metadata.tsv for structure selection")
    parser.add_argument("--output-dir", type=str, default="structures/p2rank_outputs", help="Directory to save P2Rank predictions")
    parser.add_argument("--config", type=str, default="alphafold", help="P2Rank configuration (alphafold / default)")
    parser.add_argument("--threads", type=int, default=8, help="Number of CPU threads for P2Rank")
    parser.add_argument("--chunk-size", type=int, default=500, help="Batch chunk size")
    parser.add_argument("--prank-exec", type=str, default=None, help="Custom path to prank executable binary")
    args = parser.parse_args()

    if args.pdb:
        out = run_p2rank(args.pdb, prank_exec=args.prank_exec, output_dir=args.output_dir, config=args.config)
        parsed = parse_p2rank_output(out, args.pdb)
        print(f"\nResult for {args.pdb}:")
        print(f"Sequence length: {len(parsed['full_sequence'])} aa")
        print(f"Pockets identified: {len(parsed['pockets'])}")
        for p in parsed['pockets'][:5]:
            print(f" - Pocket #{p['pocket_id']} ({p['name']}): Score={p['score']:.2f}, Residues={p['residue_count']}, Center={p['center']}")
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
            print("No PDB files found for prediction.")
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
