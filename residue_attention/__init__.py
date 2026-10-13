"""
AMICO Residue-Level Attention Package
======================================
Implements:
1. ResidueSelfAttentionMIL (Whole-protein context + binding site residues)
2. ResidueLigandCrossAttentionMIL (Chemical cofactor queries + pocket residues)
3. Pairwise3DDistanceBias (3D geometric distance attention bias for C-alpha coordinates)
4. ResidueDataset & residue_collate_fn
5. ResidueFeatureExtractor
6. Key residue ranking and PyMOL / PDB B-factor interpretability export
"""

from residue_attention.geometric_bias import Pairwise3DDistanceBias
from residue_attention.model_residue_self_attention import (
    ResidueSelfAttentionMIL,
    MultiHeadAttentionWith3DBias,
    TARGET_NAMES
)
from residue_attention.model_residue_ligand_cross_att import (
    ResidueLigandCrossAttentionMIL,
    COFACTORS,
    generate_ecfp4_fingerprints
)
from residue_attention.dataset_residues import ResidueDataset, residue_collate_fn
from residue_attention.feature_extractor_residues import (
    ResidueFeatureExtractor,
    parse_p2rank_residues_for_protein,
    extract_ca_coords_from_pdb
)
from residue_attention.interpretability import (
    rank_key_residues,
    format_key_residues_summary,
    export_pymol_script,
    write_pdb_with_attention_bfactor
)

__all__ = [
    'Pairwise3DDistanceBias',
    'ResidueSelfAttentionMIL',
    'ResidueLigandCrossAttentionMIL',
    'MultiHeadAttentionWith3DBias',
    'ResidueDataset',
    'residue_collate_fn',
    'ResidueFeatureExtractor',
    'parse_p2rank_residues_for_protein',
    'extract_ca_coords_from_pdb',
    'rank_key_residues',
    'format_key_residues_summary',
    'export_pymol_script',
    'write_pdb_with_attention_bfactor',
    'TARGET_NAMES',
    'COFACTORS',
    'generate_ecfp4_fingerprints'
]
