"""
Kompatibilní modul pro train_sequence_mlp.
Spouští Sequence ESM-2 MLP trénovací a benchmarkovou pipeline z benchmarks/sequence_mlp_benchmark.py.
"""

from benchmarks.sequence_mlp_benchmark import main, run_sequence_mlp

if __name__ == '__main__':
    main()
