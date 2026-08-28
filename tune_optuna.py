import os
import argparse
import json
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from dataset import load_cross_mil_data, load_split_ids, match_id, custom_collate_fn
from model import LigandCrossAttentionMIL

def objective(trial, all_bags, train_ids, val_ids, device, epochs=35):
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    lr = trial.suggest_float('lr', 1e-5, 3e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.4, step=0.05)
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.25, step=0.05)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    scheduler_type = trial.suggest_categorical('scheduler_type', ['plateau', 'cosine', 'none'])

    train_bags = [b for b in all_bags if match_id(b['protein_id'], train_ids)]
    val_bags = [b for b in all_bags if match_id(b['protein_id'], val_ids)]

    train_loader = DataLoader(train_bags, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_bags, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    train_labels = [b['label'].item() for b in train_bags]
    class_counts = np.bincount(train_labels, minlength=5)
    class_weights = torch.FloatTensor(len(train_labels) / (5.0 * np.maximum(class_counts, 1))).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    model = LigandCrossAttentionMIL(
        feature_dim=1280,
        ecfp_dim=1024,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_classes=5,
        dropout=dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        scheduler = None

    best_val_macro_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for pocket_feats, mask, full_prot, labels in train_loader:
            pocket_feats, mask, full_prot, labels = pocket_feats.to(device), mask.to(device), full_prot.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(pocket_feats, mask, full_prot)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Evaluace na validační sadě
        model.eval()
        val_preds, val_truths = [], []
        val_loss_sum = 0.0
        with torch.no_grad():
            for pocket_feats, mask, full_prot, labels in val_loader:
                pocket_feats, mask, full_prot, labels = pocket_feats.to(device), mask.to(device), full_prot.to(device), labels.to(device)
                logits, _ = model(pocket_feats, mask, full_prot)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * len(labels)
                p = torch.argmax(logits, dim=-1)
                val_preds.extend(p.cpu().numpy())
                val_truths.extend(labels.cpu().numpy())

        val_loss = val_loss_sum / max(len(val_bags), 1)
        val_macro_f1 = f1_score(val_truths, val_preds, average='macro', zero_division=0)

        if scheduler:
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            elif scheduler_type == 'cosine':
                scheduler.step()

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1

        trial.report(val_macro_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_macro_f1

def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Search pro LigandCrossAttentionMIL")
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--split-suffix', type=str, default='mil_0.5')
    parser.add_argument('--pockets-path', default='data_prep/esm_dataset.pt')
    parser.add_argument('--full-proteins-path', default='data_prep/esm_full_proteins.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Zařízení: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pockets_path = os.path.join(base_dir, args.pockets_path) if not os.path.isabs(args.pockets_path) else args.pockets_path
    full_proteins_path = os.path.join(base_dir, args.full_proteins_path) if not os.path.isabs(args.full_proteins_path) else args.full_proteins_path

    all_bags = load_cross_mil_data(pockets_path, full_proteins_path, mode='pockets')
    train_ids, val_ids, _ = load_split_ids(base_dir, split_suffix=args.split_suffix)

    clean_sfx = args.split_suffix.replace('/', '_').replace('.', '_')
    db_name = f"optuna_ligand_cross_{clean_sfx}.db"
    storage_url = f"sqlite:///{os.path.join(base_dir, db_name)}"
    study_name = f"ligand_cross_optuna_{clean_sfx}"

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=8)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction='maximize',
        pruner=pruner,
        load_if_exists=True
    )

    out_json = os.path.join(base_dir, f"best_params_ligand_cross_{clean_sfx}.json")

    def save_best_callback(study, trial):
        if study.best_trial.number == trial.number:
            best_dict = {
                "best_trial_number": study.best_trial.number,
                "best_val_macro_f1": study.best_value,
                "split_suffix": args.split_suffix,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **study.best_params
            }
            with open(out_json, 'w') as f:
                json.dump(best_dict, f, indent=4)
            print(f"\n>>> Nový nejlepší trial #{trial.number} (Val Macro F1: {study.best_value:.4f}) uložen do {out_json}")

    print(f"\nSpouštím {args.n_trials} trialů (databáze: {db_name})...")
    study.optimize(
        lambda trial: objective(trial, all_bags, train_ids, val_ids, device, epochs=args.epochs),
        n_trials=args.n_trials,
        callbacks=[save_best_callback]
    )

    print("\n" + "="*50)
    print("      OPTUNA DOKONČENA     ")
    print("="*50)
    print(f"Nejlepší Val Macro F1: {study.best_value:.4f}")
    print("Nejlepší parametry:")
    for k, v in study.best_params.items():
        print(f" - {k}: {v}")

if __name__ == '__main__':
    main()
