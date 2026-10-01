"""
AMICO Model Suite
=================
Sdružuje architektury modelů:
- LigandCrossAttentionMIL (model_ligand_cross_att.py): Chemický Ligand Cross-Attention MIL model
- SelfAttentionMIL (model_self_attention.py): Self-Attention MIL model nad kapsami a celým proteinem
- SequenceMLPClassifier (model_sequence_mlp.py): Čistě sekvenční baseline model nad ESM-2
"""

from model_ligand_cross_att import (
    COFACTORS,
    TARGET_NAMES,
    generate_ecfp4_fingerprints,
    LigandCrossAttentionMIL
)
from model_self_attention import SelfAttentionMIL
from model_sequence_mlp import SequenceMLPClassifier

__all__ = [
    'COFACTORS',
    'TARGET_NAMES',
    'generate_ecfp4_fingerprints',
    'LigandCrossAttentionMIL',
    'SelfAttentionMIL',
    'SequenceMLPClassifier',
]
