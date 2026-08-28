import os
import argparse
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dataset import load_cross_mil_data, load_split_ids, match_id, custom_collate_fn, TARGET_NAMES
from model import LigandCrossAttentionMIL

class EarlyStopping:
    def __init__(self, patience=12, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def main():
    parser = argparse.ArgumentParser(description="Trénování finálního LigandCrossAttentionMIL modelu")
    parser.add_argument('--config-json', type=str, default=None, help='Cesta k JSON s nejlepšími parametry z Optuny')
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
    parser.add_argument('--save-model', type=str, default='ligand_cross_mil_best.pt')
    args = parser.parse_args()

    # Načtení Optuna konfigurace, pokud je zadána
    if args.config_json and os.path.exists(args.config_json):
        print(f"Načítám konfiguraci z {args.config_json}...")
        with open(args.config_json, 'r') as f:
            cfg = json.load(f)
        args.hidden_dim = cfg.get('hidden_dim', args.hidden_dim)
        args.num_heads = cfg.get('num_heads', args.num_heads)
        args.lr = cfg.get('lr', args.lr)
        args.weight_decay = cfg.get('weight_decay', args.weight_decay)
        args.dropout = cfg.get('dropout', args.dropout)
        args.label_smoothing = cfg.get('label_smoothing', args.label_smoothing)
        args.batch_size = cfg.get('batch_size', args.batch_size)
        if 'split_suffix' in cfg:
            args.split_suffix = cfg['split_suffix']

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Používám zařízení: {device}")
    print(f"Hyperparametry: lr={args.lr:.2e}, weight_decay={args.weight_decay:.2e}, dropout={args.dropout}, label_smoothing={args.label_smoothing}, hidden_dim={args.hidden_dim}, heads={args.num_heads}, batch_size={args.batch_size}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pockets_path = os.path.join(base_dir, args.pockets_path) if not os.path.isabs(args.pockets_path) else args.pockets_path
    full_proteins_path = os.path.join(base_dir, args.full_proteins_path) if not os.path.isabs(args.full_proteins_path) else args.full_proteins_path

    all_bags = load_cross_mil_data(pockets_path, full_proteins_path, mode='pockets')
    train_ids, val_ids, test_ids = load_split_ids(base_dir, split_suffix=args.split_suffix)

    train_bags, val_bags, test_bags = [], [], []
    for b in all_bags:
        pid = b['protein_id']
        if match_id(pid, train_ids):
            train_bags.append(b)
        elif match_id(pid, val_ids):
            val_bags.append(b)
        elif match_id(pid, test_ids):
            test_bags.append(b)

    print(f"Rozdělení ({args.split_suffix}) -> Train: {len(train_bags)}, Val: {len(val_bags)}, Test: {len(test_bags)}")
    if len(train_bags) == 0:
        print("Chyba: prázdný train set!")
        return

    train_loader = DataLoader(train_bags, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_bags, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_bags, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn) if len(test_bags) > 0 else None

    train_labels = [b['label'].item() for b in train_bags]
    class_counts = np.bincount(train_labels, minlength=5)
    class_weights = torch.FloatTensor(len(train_labels) / (5.0 * np.maximum(class_counts, 1))).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    model = LigandCrossAttentionMIL(
        feature_dim=1280,
        ecfp_dim=1024,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_classes=5,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    early_stopping = EarlyStopping(patience=args.patience)

    def evaluate(loader):
        model.eval()
        preds, truths = [], []
        loss_sum = 0.0
        with torch.no_grad():
            for pocket_feats, mask, full_prot, labels in loader:
                pocket_feats, mask, full_prot, labels = pocket_feats.to(device), mask.to(device), full_prot.to(device), labels.to(device)
                logits, _ = model(pocket_feats, mask, full_prot)
                loss = criterion(logits, labels)
                loss_sum += loss.item() * len(labels)
                p = torch.argmax(logits, dim=-1)
                preds.extend(p.cpu().numpy())
                truths.extend(labels.cpu().numpy())
        acc = accuracy_score(truths, preds) if len(truths) > 0 else 0.0
        f1_m = f1_score(truths, preds, average='macro', zero_division=0) if len(truths) > 0 else 0.0
        rep = classification_report(truths, preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0) if len(truths) > 0 else {}
        return loss_sum / max(len(truths), 1), acc, f1_m, rep

    print("\n--- Spouštím trénování LigandCrossAttentionMIL ---")
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for pocket_feats, mask, full_prot, labels in train_loader:
            pocket_feats, mask, full_prot, labels = pocket_feats.to(device), mask.to(device), full_prot.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(pocket_feats, mask, full_prot)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(labels)

        train_loss = train_loss_sum / max(len(train_bags), 1)
        val_loss, val_acc, val_f1, _ = evaluate(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_weights, args.save_model)
            save_msg = "🔥 (Model uložen)"
        else:
            save_msg = ""

        if epoch % 2 == 0 or epoch == 1 or save_msg:
            print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f} {save_msg}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping aktivován po {epoch} epochách.")
            break

    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    print("\n" + "="*50)
    print("      VÝSLEDKY EVALUACE (Best Checkpoint)     ")
    print("="*50)
    _, val_acc, val_f1, _ = evaluate(val_loader)
    print(f"VALIDACE -> Acc: {val_acc:.4f} | Macro F1: {val_f1:.4f}")

    if test_loader:
        _, test_acc, test_f1, test_rep = evaluate(test_loader)
        print(f"TEST     -> Acc: {test_acc:.4f} | Macro F1: {test_f1:.4f}")
        print("\nDetailní Testovací Report:")
        print(classification_report([b['label'].item() for b in test_bags], 
                                    [evaluate(DataLoader([b], batch_size=1, collate_fn=custom_collate_fn))[3] for b in []] if False else [], 
                                    target_names=TARGET_NAMES, zero_division=0) if False else "")
        for name in TARGET_NAMES:
            if name in test_rep:
                print(f" - {name:<12}: Precision={test_rep[name]['precision']:.4f}, Recall={test_rep[name]['recall']:.4f}, F1={test_rep[name]['f1-score']:.4f}")

if __name__ == '__main__':
    main()
