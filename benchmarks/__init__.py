"""
AMICO Benchmarks Suite
======================
Obsahuje baseline benchmarky pro predikci specificity kofaktorů:
1. Foldseek 1-NN Benchmark: 3D strukturní zarovnání (Test split = Query, Train split = Database).
2. Sequence ESM-2 MLP Benchmark: Čistě sekvenční baseline nad full protein sequence embeddingy bez kapes a ligandů.
"""

from benchmarks.foldseek_benchmark import run_foldseek_benchmark
from benchmarks.sequence_mlp import SequenceMLPClassifier
from benchmarks.sequence_mlp_benchmark import run_sequence_mlp
from benchmarks.nise_benchmark import evaluate_amico_models, generate_nise_report

__all__ = [
    'run_foldseek_benchmark',
    'SequenceMLPClassifier',
    'run_sequence_mlp',
    'evaluate_amico_models',
    'generate_nise_report',
]

