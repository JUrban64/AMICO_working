import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import argparse
import json
import time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset import load_split_ids, match_id, TARGET_NAMES
from benchmarks.sequence_mlp import SequenceMLPClassifier


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


def find_file(filename, candidates, target_root):
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    # Zkusit v data_prep a kořeni projektu
    p1 = os.path.join(target_root, 'data_prep', filename)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(target_root, filename)
    if os.path.exists(p2):
        return p2
    # Zkusit v okolních složkách
    p3 = os.path.join(target_root, '..', 'AMICO_workign', 'data_prep', filename)
    if os.path.exists(p3):
        return p3
    return None


def load_sequence_dataset(target_root, pockets_path=None, full_proteins_path=None):
    """
    Načte full protein ESM-2 embeddingy [B, 1280] a příslušné kofaktorové labely.
    Podporuje jak spárování přes load_cross_mil_data, tak přímé načtení esm_full_proteins.pt.
    """
    resolved_pockets = find_file('esm_dataset.pt', [pockets_path], target_root)
    resolved_full = find_file('esm_full_proteins.pt', [full_proteins_path], target_root)

    if not resolved_full:
        raise FileNotFoundError("Chyba: 'esm_full_proteins.pt' nebyl nalezen v projektu ani v data_prep/.")

    print(f"Načítám full protein embeddings z: {resolved_full}")
    full_prots = torch.load(resolved_full, weights_only=False)

    labels_dict = {}
    if resolved_pockets and os.path.exists(resolved_pockets):
        print(f"Načítám labely z esm_dataset.pt: {resolved_pockets}")
        raw_pockets = torch.load(resolved_pockets, weights_only=False)
        for item in raw_pockets:
            raw_pid = item['protein_id']
            base_name = os.path.basename(raw_pid)
            pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
            lbl = item['label']
            if torch.is_tensor(lbl):
                lbl = lbl.item()
            labels_dict[pid] = int(lbl)

    # Vytvoření seznamu položek
    items = []
    for pid, feat in full_prots.items():
        if pid in labels_dict:
            lbl = labels_dict[pid]
        else:
            clean_pid = pid.split('_')[0]
            if clean_pid in labels_dict:
                lbl = labels_dict[clean_pid]
            else:
                continue

        if isinstance(feat, np.ndarray):
            feat_tensor = torch.FloatTensor(feat)
        elif torch.is_tensor(feat):
            feat_tensor = feat.float()
        else:
            feat_tensor = torch.tensor(feat, dtype=torch.float32)

        items.append({
            'protein_id': pid,
            'full_protein_feature': feat_tensor,
            'label': torch.tensor(lbl, dtype=torch.long)
        })

    print(f"Načteno celkem {len(items)} proteinů s full sequence embeddingy a labely.")
    return items


def run_sequence_mlp(split_suffix='mil_0.5', device=None, epochs=50, lr=5e-5,
                     weight_decay=1e-3, dropout=0.3, label_smoothing=0.1,
                     batch_size=64, hidden_dim=256, patience=12,
                     use_nr=False, pockets_path=None, full_proteins_path=None,
                     target_root=None, save_model_path=None):
    """
    Trénuje a vyhodnocuje SequenceMLPClassifier:
      - Vstup: pouze full protein ESM-2 sequence embedding [1280] (žádné 3D kapsy, žádné ligandy).
      - Výstup: 5 kofaktorových logitů.
    Vrací slovník s metrikami.
    """
    r_dir = target_root or project_root

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    print(f"\n=================================================================")
    print(f"       SEQUENCE ESM-2 MLP BENCHMARK (Pure Sequence Baseline)")
    print(f"=================================================================")
    print(f"Zařízení:        {device}")
    print(f"Split suffix:    {split_suffix} (use_nr={use_nr})")
    print(f"Hyperparametry:  lr={lr:.2e}, weight_decay={weight_decay:.2e}, dropout={dropout}, "
          f"label_smoothing={label_smoothing}, hidden_dim={hidden_dim}, batch_size={batch_size}")

    start_time = time.time()

    # 1. Načtení dat a splitů
    try:
        all_items = load_sequence_dataset(r_dir, pockets_path=pockets_path, full_proteins_path=full_proteins_path)
    except Exception as e:
        msg = f"Selhalo načítání datasetu: {e}"
        print(f"CHYBA: {msg}")
        return {"status": f"FAILED: {msg}"}

    train_ids, val_ids, test_ids = load_split_ids(r_dir, split_suffix=split_suffix, use_nr=use_nr)
    if len(train_ids) == 0 or len(test_ids) == 0:
        msg = f"Soubory splitů pro suffix '{split_suffix}' nebyly nalezeny."
        print(f"CHYBA: {msg}")
        return {"status": f"FAILED: {msg}"}

    train_items, val_items, test_items = [], [], []
    for item in all_items:
        pid = item['protein_id']
        if match_id(pid, train_ids):
            train_items.append(item)
        elif match_id(pid, val_ids):
            val_items.append(item)
        elif match_id(pid, test_ids):
            test_items.append(item)

    print(f"Rozdělení -> Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}")
    if len(train_items) == 0:
        msg = "Prázdná trénovací množina (žádná ID se neshodovala)."
        print(f"CHYBA: {msg}")
        return {"status": f"FAILED: {msg}"}

    def collate_fn(batch):
        feats = torch.stack([b['full_protein_feature'] for b in batch])
        labels = torch.stack([b['label'] for b in batch])
        return feats, labels

    train_loader = DataLoader(train_items, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_items, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_items, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if len(test_items) > 0 else None

    # Vyvážení vah tříd
    train_labels = [item['label'].item() for item in train_items]
    class_counts = np.bincount(train_labels, minlength=5)
    class_weights = torch.FloatTensor(len(train_labels) / (5.0 * np.maximum(class_counts, 1))).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    model = SequenceMLPClassifier(
        in_features=1280,
        hidden_dim=hidden_dim,
        num_classes=5,
        dropout=dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    early_stopping = EarlyStopping(patience=patience)

    def evaluate(loader):
        if loader is None or len(loader.dataset) == 0:
            return 0.0, 0.0, 0.0, 0.0, {}
        model.eval()
        preds, truths = [], []
        loss_sum = 0.0
        with torch.no_grad():
            for feats, lbl in loader:
                feats, lbl = feats.to(device), lbl.to(device)
                logits = model(feats)
                loss = criterion(logits, lbl)
                loss_sum += loss.item() * len(lbl)
                p = torch.argmax(logits, dim=1)
                preds.extend(p.cpu().numpy())
                truths.extend(lbl.cpu().numpy())

        avg_loss = loss_sum / max(len(truths), 1)
        acc = accuracy_score(truths, preds) if len(truths) > 0 else 0.0
        f1_m = f1_score(truths, preds, average='macro', zero_division=0) if len(truths) > 0 else 0.0
        f1_w = f1_score(truths, preds, average='weighted', zero_division=0) if len(truths) > 0 else 0.0
        rep = classification_report(truths, preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0) if len(truths) > 0 else {}
        return avg_loss, acc, f1_m, f1_w, rep

    best_val_loss = float('inf')
    best_weights = None

    print("\n--- Spouštím trénování Sequence ESM-2 MLP ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for feats, lbl in train_loader:
            feats, lbl = feats.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(lbl)

        train_loss = train_loss_sum / max(len(train_items), 1)
        val_loss, val_acc, val_f1, _, _ = evaluate(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_indicator = "🔥 (Best Val)"
        else:
            save_indicator = ""

        if epoch % 2 == 0 or epoch == 1 or save_indicator:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1*100:.2f}% {save_indicator}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping aktivován po {epoch} epochách.")
            break

    elapsed = time.time() - start_time
    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    if save_model_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_model_path)), exist_ok=True)
        torch.save(model.state_dict(), save_model_path)
        print(f"Model byl uložen do: {save_model_path}")

    # Finální evaluace
    _, val_acc, val_f1, val_w_f1, _ = evaluate(val_loader)
    test_acc, test_f1, test_w_f1 = 0.0, 0.0, 0.0
    per_class_f1 = {name: 0.0 for name in TARGET_NAMES}

    print("\n" + "=" * 55)
    print("      SEQUENCE ESM-2 MLP BENCHMARK VÝSLEDKY       ")
    print("=" * 55)
    print(f"Val Accuracy:     {val_acc * 100:.2f} %")
    print(f"Val Macro F1:     {val_f1 * 100:.2f} %")

    if test_loader:
        _, test_acc, test_f1, test_w_f1, test_rep = evaluate(test_loader)
        for name in TARGET_NAMES:
            if name in test_rep:
                per_class_f1[name] = test_rep[name].get('f1-score', 0.0)

        print(f"Test Accuracy:    {test_acc * 100:.2f} %")
        print(f"Test Macro F1:    {test_f1 * 100:.2f} %")
        print(f"Test Weighted F1: {test_w_f1 * 100:.2f} %")
        print(f"Čas běhu:         {elapsed:.1f} s")
        print("-" * 55)
        print("Detailní Testovací Klasifikační Report:")
        # Tisk textové podoby
        y_test_true, y_test_pred = [], []
        model.eval()
        with torch.no_grad():
            for feats, lbl in test_loader:
                feats = feats.to(device)
                logits = model(feats)
                p = torch.argmax(logits, dim=1)
                y_test_pred.extend(p.cpu().numpy())
                y_test_true.extend(lbl.numpy())
        print(classification_report(y_test_true, y_test_pred, target_names=TARGET_NAMES, zero_division=0))
        print("=" * 55 + "\n")

    return {
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "test_weighted_f1": test_w_f1,
        "per_class_f1": per_class_f1,
        "time_sec": round(elapsed, 1),
        "status": "SUCCESS"
    }


def main():
    parser = argparse.ArgumentParser(description="Sequence ESM-2 MLP Benchmark (Pure Sequence Baseline bez kapes a bez ligandů)")
    parser.add_argument('--split-suffix', type=str, default='mil_0.5', help='Přípona splitu (např. mil_0.5, struct_pocket_0.5_0.5)')
    parser.add_argument('--use-nr', action='store_true', help='Použít Non-Redundant variantu splitu (_nr)')
    parser.add_argument('--epochs', type=int, default=50, help='Maximální počet epoch')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout pravděpodobnost')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing koeficient')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Skrytá dimenze MLP')
    parser.add_argument('--patience', type=int, default=12, help='Patience pro early stopping')
    parser.add_argument('--pockets-path', default=None, help='Cesta k esm_dataset.pt (pro načtení labelů)')
    parser.add_argument('--full-proteins-path', default=None, help='Cesta k esm_full_proteins.pt')
    parser.add_argument('--save-model', type=str, default=None, help='Cesta k uložení checkpointu')
    args = parser.parse_args()

    run_sequence_mlp(
        split_suffix=args.split_suffix,
        use_nr=args.use_nr,
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
        save_model_path=args.save_model
    )


if __name__ == '__main__':
    main()
