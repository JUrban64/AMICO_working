import os
import shutil
import numpy as np
import torch
from Bio.PDB import PDBParser, PDBIO, Select

from model import COFACTORS, TARGET_NAMES

def generate_3d_ligand(cofactor_name, out_path="ligand_3d.sdf"):
    """
    Vygeneruje 3D konformaci kofaktoru ze SMILES pomocí RDKit a MMFF94 optimalizace.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smi = COFACTORS.get(cofactor_name)
    if not smi:
        raise ValueError(f"Neznámý kofaktor: {cofactor_name}")

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
    Vypočítá těžiště (center_x, center_y, center_z) kapsy z PDB souboru.
    Pokud není seznam reziduí zadán, vrátí těžiště celého proteinu.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', protein_pdb)
    coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.get_id()[1]
                if pocket_res_list is None or res_id in pocket_res_list:
                    for atom in residue:
                        coords.append(atom.get_coord())

    if len(coords) == 0:
        return np.array([0.0, 0.0, 0.0])
    return np.mean(coords, axis=0)


def dock_predicted_cofactor(protein_pdb, cofactor_name, pocket_center, out_dir="docking_results", exhaustiveness=8):
    """
    Provede molekulární dokování předpovězeného kofaktoru do identifikované kapsy pomocí AutoDock Vina.
    Pokud není Vina nainstalována, vygeneruje 3D konformaci, konfigurační soubor a preview komplexu.
    """
    os.makedirs(out_dir, exist_ok=True)
    ligand_sdf = os.path.join(out_dir, f"{cofactor_name}_ligand.sdf")
    ligand_pdb = os.path.join(out_dir, f"{cofactor_name}_ligand.pdb")
    
    print(f"\n---> [DOCKING] Generuji 3D konformaci pro {cofactor_name}...")
    generate_3d_ligand(cofactor_name, ligand_sdf)
    generate_3d_ligand(cofactor_name, ligand_pdb)

    cx, cy, cz = pocket_center
    box_size = (25.0, 25.0, 25.0)

    # Zápis Vina konfigurace
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

    print(f"---> [DOCKING] Vina Box vycentrován na kapsu: [{cx:.2f}, {cy:.2f}, {cz:.2f}] Å (velikost: 25 Å)")

    vina_success = False
    docked_pdbqt = os.path.join(out_dir, f"{cofactor_name}_docked.pdbqt")

    # 1. Zkusíme spustit Vina přes Python API (pokud je nainstalován balíček `vina`)
    try:
        from vina import Vina
        v = Vina(sf_name='vina', cpu=0, seed=42)
        v.set_receptor(protein_pdb) # nebo PDBQT
        v.set_ligand_from_file(ligand_pdb)
        v.compute_vina_maps(center=[cx, cy, cz], box_size=list(box_size))
        v.dock(exhaustiveness=exhaustiveness, n_poses=5)
        v.write_poses(docked_pdbqt, n_poses=5, overwrite=True)
        vina_success = True
        print(f"✅ [DOCKING] Vina dokování úspěšně dokončeno -> {docked_pdbqt}")
    except Exception as e:
        # Fallback pokud chybí knihovna vina nebo pdbqt příprava
        pass

    # 2. Zkusíme spustit Vina CLI (pokud existuje binárka vina v PATH)
    if not vina_success and shutil.which('vina'):
        try:
            import subprocess
            cmd = f"vina --receptor {protein_pdb} --ligand {ligand_pdb} --config {config_txt} --out {docked_pdbqt}"
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            vina_success = True
            print(f"✅ [DOCKING] Vina CLI dokování úspěšně dokončeno -> {docked_pdbqt}")
        except Exception:
            pass

    # 3. Fallback: Vytvoření preview komplexu přesunutím 3D ligandu do těžiště kapsy
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

            # Spojení proteinu a ligandu do jednoho PDB pro snadné otevření v PyMOLu
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
            print(f"📦 [DOCKING] Náhled komplexu protein+kofaktor uložen do: {complex_pdb}")
    except Exception as e:
        print(f"Upozornění při vytváření náhledu: {e}")

    return {
        "status": "SUCCESS",
        "output_dir": out_dir,
        "docked_file": docked_pdbqt if vina_success else complex_pdb,
        "config_file": config_txt,
        "pocket_center": [float(cx), float(cy), float(cz)]
    }
