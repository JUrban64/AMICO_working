"""
Kompatibilní modul pro model_sequence_mlp.
Přesměrovává na implementaci v benchmarks.sequence_mlp.
"""

from benchmarks.sequence_mlp import SequenceMLPClassifier

__all__ = ['SequenceMLPClassifier']
