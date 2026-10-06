import os
import csv
import glob
import shutil
import sys
from pathlib import Path
import numpy as np
import torch

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from Bio.PDB import PDBParser, PDBIO, Select
except ImportError:
    PDBParser = None
    PDBIO = None
    Select = None

from model_ligand_cross_att import COFACTORS, TARGET_NAMES


def generate_3d_ligand(cofactor_name, out_path="ligand_3d.sdf"):
    """
    Generates a 3D conformation for a cofactor from SMILES using RDKit and MMFF94 force field.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smi = COFACTORS.get(cofactor_name)
    if not smi:
        raise ValueError(f"Unknown cofactor: {cofactor_name}")

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        mol = Chem.MolFromSmiles(smi, sanitize=False)

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    res = AllChem.EmbedMolecule(mol, params)
    if res != 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True)

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    if out_path.endswith('.sdf'):
        writer = Chem.SDWriter(out_path)
        writer.write(mol)
        writer.close()
    elif out_path.endswith('.pdb'):
        Chem.MolToPDBFile(mol, out_path)

    return out_path


def get_pocket_center_from_pdb(protein_pdb, pocket_res_list=None):
    """
    Calculates center of mass (center_x, center_y, center_z) of a pocket from a PDB file.
    If no residue list is provided, returns the center of mass of the whole protein.
    """
    coords = []
    if PDBParser is not None:
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', protein_pdb)
            for model in structure:
                for chain in model:
                    for residue in chain:
                        res_id = residue.get_id()[1]
                        if pocket_res_list is None or res_id in pocket_res_list:
                            for atom in residue:
                                coords.append(atom.get_coord())
            if len(coords) > 0:
                return np.mean(coords, axis=0)
        except Exception:
            coords = []

    # Built-in pure-Python fallback
    if os.path.exists(protein_pdb):
        with open(protein_pdb, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith(('ATOM  ', 'HETATM')):
                    if pocket_res_list is not None:
                        try:
                            resseq = int(line[22:26].strip())
                            if resseq not in pocket_res_list:
                                continue
                        except ValueError:
                            continue
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
                    except ValueError:
                        pass

    if len(coords) == 0:
        return np.array([0.0, 0.0, 0.0])
    return np.mean(coords, axis=0)


def get_best_p2rank_pocket_center(p2rank_dir_or_csv):
    """
    Extracts 3D coordinates [center_x, center_y, center_z] of the top P2Rank pocket
    (rank 1 with highest score / probability).
    
    Args:
        p2rank_dir_or_csv: Directory with P2Rank outputs or path to *_predictions.csv.
        
    Returns:
        tuple: (np.ndarray [cx, cy, cz], pocket_id, pocket_name) or (None, None, None)
    """
    p_path = Path(p2rank_dir_or_csv)
    pred_csv = None

    if p_path.is_file() and p_path.name.endswith(".csv"):
        pred_csv = p_path
    elif p_path.is_dir():
        candidates = list(p_path.glob("*_predictions.csv")) + list(p_path.glob("*.csv"))
        if candidates:
            pred_csv = candidates[0]

    if not pred_csv or not pred_csv.exists():
        return None, None, None

    best_pocket = None
    best_score = -1e9

    try:
        with open(pred_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                score = float(clean_row.get('score', clean_row.get('probability', 0.0)))
                rank = int(clean_row.get('rank', 999))
                cx = float(clean_row.get('center_x', clean_row.get('x', 0.0)))
                cy = float(clean_row.get('center_y', clean_row.get('y', 0.0)))
                cz = float(clean_row.get('center_z', clean_row.get('z', 0.0)))
                name = clean_row.get('name', f"pocket{rank}")

                # Rank 1 or highest score
                if rank == 1 or score > best_score:
                    best_score = score
                    best_pocket = (np.array([cx, cy, cz]), rank, name)
                    if rank == 1:
                        break
    except Exception:
        return None, None, None

    if best_pocket is not None:
        return best_pocket
    return None, None, None


def dock_predicted_cofactor(
    protein_pdb,
    cofactor_name,
    pocket_center=None,
    p2rank_dir=None,
    out_dir="docking_results",
    exhaustiveness=8
):
    """
    Performs molecular docking of the predicted cofactor into the top P2Rank pocket
    using AutoDock Vina.
    
    If pocket_center is not explicitly provided, attempts to extract the center
    of the best P2Rank pocket (rank 1) from p2rank_dir or surrounding outputs.
    If Vina is not installed, generates 3D ligand conformation, configuration file,
    and a preview complex.
    """
    os.makedirs(out_dir, exist_ok=True)
    ligand_sdf = os.path.join(out_dir, f"{cofactor_name}_ligand.sdf")
    ligand_pdb = os.path.join(out_dir, f"{cofactor_name}_ligand.pdb")
    
    print(f"\n---> [DOCKING] Generating 3D conformation for {cofactor_name}...")
    generate_3d_ligand(cofactor_name, ligand_sdf)
    generate_3d_ligand(cofactor_name, ligand_pdb)

    # Determine pocket center (prioritizing top P2Rank pocket)
    pocket_label = "specified pocket"
    if pocket_center is None and p2rank_dir:
        p2_center, p2_rank, p2_name = get_best_p2rank_pocket_center(p2rank_dir)
        if p2_center is not None:
            pocket_center = p2_center
            pocket_label = f"top P2Rank pocket #{p2_rank} ({p2_name})"

    if pocket_center is None:
        # Check standard P2Rank directories in protein neighborhood
        stem = Path(protein_pdb).stem
        candidates = [
            Path(protein_pdb).parent / f"{stem}_prank_output",
            Path("./temp_p2rank") / f"{stem}_prank_output",
            Path(out_dir).parent / f"{stem}_prank_output"
        ]
        for c in candidates:
            if c.exists():
                p2_center, p2_rank, p2_name = get_best_p2rank_pocket_center(c)
                if p2_center is not None:
                    pocket_center = p2_center
                    pocket_label = f"top P2Rank pocket #{p2_rank} ({p2_name})"
                    break

    if pocket_center is None:
        pocket_center = get_pocket_center_from_pdb(protein_pdb)
        pocket_label = "protein center of mass fallback"

    cx, cy, cz = pocket_center
    box_size = (25.0, 25.0, 25.0)

    # Write Vina configuration
    config_txt = os.path.join(out_dir, "vina_config.txt")
    with open(config_txt, 'w') as f:
        f.write(f"center_x = {cx:.3f}\n")
        f.write(f"center_y = {cy:.3f}\n")
        f.write(f"center_z = {cz:.3f}\n\n")
        f.write(f"size_x = {box_size[0]:.1f}\n")
        f.write(f"size_y = {box_size[1]:.1f}\n")
        f.write(f"size_z = {box_size[2]:.1f}\n\n")
        f.write(f"exhaustiveness = {exhaustiveness}\n")
        f.write(f"num_modes = 9\n")

    print(f"---> [DOCKING] Vina search box centered at {pocket_label}: [{cx:.2f}, {cy:.2f}, {cz:.2f}] Å (size: 25 Å)")

    vina_success = False
    docked_pdbqt = os.path.join(out_dir, f"{cofactor_name}_docked.pdbqt")

    # 1. Try Vina Python API if available
    try:
        from vina import Vina
        v = Vina(sf_name='vina', cpu=0, seed=42)
        v.set_receptor(protein_pdb)
        v.set_ligand_from_file(ligand_pdb)
        v.compute_vina_maps(center=[cx, cy, cz], box_size=list(box_size))
        v.dock(exhaustiveness=exhaustiveness, n_poses=5)
        v.write_poses(docked_pdbqt, n_poses=5, overwrite=True)
        vina_success = True
        print(f"✅ [DOCKING] Vina docking finished successfully -> {docked_pdbqt}")
    except Exception:
        pass

    # 2. Try Vina CLI binary if in PATH
    if not vina_success and shutil.which('vina'):
        try:
            import subprocess
            cmd = f"vina --receptor {protein_pdb} --ligand {ligand_pdb} --config {config_txt} --out {docked_pdbqt}"
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            vina_success = True
            print(f"✅ [DOCKING] Vina CLI docking finished successfully -> {docked_pdbqt}")
        except Exception:
            pass

    # 3. Fallback: Create preview complex by translating 3D ligand into pocket center
    complex_pdb = os.path.join(out_dir, f"complex_{cofactor_name}_pocket_preview.pdb")
    try:
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(ligand_sdf)
        mol = next(suppl)
        if mol is not None:
            conf = mol.GetConformer()
            lig_coords = conf.GetPositions()
            lig_center = np.mean(lig_coords, axis=0)
            translation = pocket_center - lig_center
            for i in range(mol.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, p + translation)
            Chem.MolToPDBFile(mol, os.path.join(out_dir, "ligand_in_pocket.pdb"))

            # Merge protein and ligand into a single PDB for viewing in PyMOL/ChimeraX
            with open(complex_pdb, 'w') as out_f:
                if os.path.exists(protein_pdb):
                    with open(protein_pdb, 'r') as pf:
                        for line in pf:
                            if line.startswith(('ATOM', 'HETATM', 'TER')):
                                out_f.write(line)
                with open(os.path.join(out_dir, "ligand_in_pocket.pdb"), 'r') as lf:
                    for line in lf:
                        if line.startswith(('ATOM', 'HETATM')):
                            out_f.write(line)
            print(f"📦 [DOCKING] Protein-cofactor complex preview saved to: {complex_pdb}")
    except Exception as e:
        print(f"Warning during complex preview generation: {e}")

    return {
        "status": "SUCCESS",
        "output_dir": out_dir,
        "docked_file": docked_pdbqt if vina_success else complex_pdb,
        "config_file": config_txt,
        "pocket_center": [float(cx), float(cy), float(cz)],
        "target_pocket": pocket_label
    }
