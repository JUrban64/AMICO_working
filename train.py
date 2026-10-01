#!/usr/bin/env python3
"""
Trénovací dispečer pro modely AMICO.
===================================
1. ligand_cross_att (model_ligand_cross_att.py / train_ligand_cross_att.py)
2. self_attention (model_self_attention.py / train_self_attention.py)
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Trénování modelů AMICO (ligand_cross_att / self_attention)")
    parser.add_argument('--model', type=str, default='ligand_cross_att',
                        choices=['ligand_cross_att', 'self_attention', 'sequence_mlp'],
                        help='Architektura modelu: ligand_cross_att nebo self_attention')
    parser.add_argument('--config-json', type=str, default=None, help='Cesta k JSON konfiguraci')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=4.86e-5)
    parser.add_argument('--weight-decay', type=float, default=1.07e-5)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--label-smoothing', type=float, default=0.20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--split-suffix', type=str, default='mil_0.5')
    parser.add_argument('--pockets-path', default='data_prep/esm_dataset.pt')
    parser.add_argument('--full-proteins-path', default='data_prep/esm_full_proteins.pt')
    parser.add_argument('--save-model', type=str, default=None, help='Cesta pro uložení modelu')
    args = parser.parse_args()

    if args.model == 'self_attention':
        from train_self_attention import train_self_attention
        save_model = args.save_model or 'self_attention_mil_best.pt'
        train_self_attention(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            label_smoothing=args.label_smoothing,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            patience=args.patience,
            split_suffix=args.split_suffix,
            pockets_path=args.pockets_path,
            full_proteins_path=args.full_proteins_path,
            save_model=save_model,
            config_json=args.config_json
        )
    elif args.model == 'sequence_mlp':
        from benchmarks.sequence_mlp_benchmark import run_sequence_mlp
        save_model = args.save_model or 'sequence_mlp_best.pt'
        run_sequence_mlp(
            split_suffix=args.split_suffix,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            label_smoothing=args.label_smoothing,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            patience=args.patience,
            pockets_path=args.pockets_path,
            full_proteins_path=args.full_proteins_path,
            save_model_path=save_model
        )
    else:
        from train_ligand_cross_att import train_ligand_cross_att
        save_model = args.save_model or 'ligand_cross_mil_best.pt'
        train_ligand_cross_att(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            label_smoothing=args.label_smoothing,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            patience=args.patience,
            split_suffix=args.split_suffix,
            pockets_path=args.pockets_path,
            full_proteins_path=args.full_proteins_path,
            save_model=save_model,
            config_json=args.config_json
        )


if __name__ == '__main__':
    main()
