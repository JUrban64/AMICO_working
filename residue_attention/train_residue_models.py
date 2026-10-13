import os
import sys
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Ensure root directory is accessible
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from residue_attention.model_residue_self_attention import ResidueSelfAttentionMIL
from residue_attention.model_residue_ligand_cross_att import ResidueLigandCrossAttentionMIL
from residue_attention.dataset_residues import ResidueDataset, residue_collate_fn, TARGET_NAMES

try:
    from sklearn.metrics import accuracy_score, f1_score, classification_report
except ImportError:
    accuracy_score = None
    f1_score = None
    classification_report = None


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


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            padded_feats, padding_mask, full_prots, padded_coords, labels, _, _ = batch
            padded_feats = padded_feats.to(device)
            padding_mask = padding_mask.to(device)
            full_prots = full_prots.to(device)
            padded_coords = padded_coords.to(device)
            labels = labels.to(device)

            if isinstance(model, ResidueLigandCrossAttentionMIL):
                logits, _, _ = model(padded_feats, padding_mask, full_prots, padded_coords)
            else:
                logits, _ = model(padded_feats, padding_mask, full_prots, padded_coords)

            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    n_samples = len(all_labels)
    avg_loss = total_loss / max(n_samples, 1)

    acc = np.mean(np.array(all_preds) == np.array(all_labels)) if n_samples > 0 else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if f1_score and n_samples > 0 else acc

    return avg_loss, acc, macro_f1, all_preds, all_labels


def train_residue_model(
    model_type='residue_ligand_cross',
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    epochs=50,
    lr=5e-5,
    weight_decay=1e-5,
    dropout=0.2,
    label_smoothing=0.15,
    batch_size=32,
    hidden_dim=256,
    num_heads=4,
    use_3d_bias=True,
    bias_3d_mode='rbf_gaussian',
    patience=12,
    save_path='residue_model_best.pt',
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    print(f"-> Initializing {model_type} on device {device}...")
    print(f"   Hidden dim: {hidden_dim}, Heads: {num_heads}, 3D Bias: {use_3d_bias} ({bias_3d_mode})")

    # Initialize model
    if model_type == 'residue_ligand_cross':
        model = ResidueLigandCrossAttentionMIL(
            feature_dim=1280,
            ecfp_dim=2048,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_classes=5,
            dropout=dropout,
            use_3d_bias=use_3d_bias,
            bias_3d_mode=bias_3d_mode
        ).to(device)
    else:
        model = ResidueSelfAttentionMIL(
            feature_dim=1280,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_classes=5,
            dropout=dropout,
            use_3d_bias=use_3d_bias,
            bias_3d_mode=bias_3d_mode
        ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=residue_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=residue_collate_fn) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=residue_collate_fn) if test_dataset else None

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    early_stopper = EarlyStopping(patience=patience)

    best_val_loss = float('inf')
    best_epoch = 0

    print("\nStarting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_t = time.time()

        for batch in train_loader:
            padded_feats, padding_mask, full_prots, padded_coords, labels, _, _ = batch
            padded_feats = padded_feats.to(device)
            padding_mask = padding_mask.to(device)
            full_prots = full_prots.to(device)
            padded_coords = padded_coords.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if isinstance(model, ResidueLigandCrossAttentionMIL):
                logits, _, _ = model(padded_feats, padding_mask, full_prots, padded_coords)
            else:
                logits, _ = model(padded_feats, padding_mask, full_prots, padded_coords)

            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)

        train_loss /= len(train_dataset)
        elapsed = time.time() - start_t

        # Validation
        if val_loader:
            val_loss, val_acc, val_f1, _, _ = evaluate_model(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1*100:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'model_type': model_type,
                    'hparams': {
                        'hidden_dim': hidden_dim,
                        'num_heads': num_heads,
                        'use_3d_bias': use_3d_bias,
                        'bias_3d_mode': bias_3d_mode,
                        'dropout': dropout
                    }
                }, save_path)

            if early_stopper(val_loss):
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                break
        else:
            print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | Train Loss: {train_loss:.4f}")

    if val_loader and os.path.exists(save_path):
        print(f"\nLoading best checkpoint from {save_path}...")
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    if test_loader:
        test_loss, test_acc, test_f1, preds, targets = evaluate_model(model, test_loader, criterion, device)
        print("=" * 60)
        print(f"TEST RESULTS: Acc = {test_acc*100:.2f}%, Macro F1 = {test_f1*100:.2f}%, Loss = {test_loss:.4f}")
        if classification_report:
            print(classification_report(targets, preds, target_names=TARGET_NAMES, zero_division=0))
        print("=" * 60)

    return model
