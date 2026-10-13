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
from dataset import load_split_ids, match_id

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
                save_dir = os.path.dirname(os.path.abspath(save_path))
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
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
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    if test_loader:
        test_loss, test_acc, test_f1, preds, targets = evaluate_model(model, test_loader, criterion, device)
        print("=" * 60)
        print(f"TEST RESULTS: Acc = {test_acc*100:.2f}%, Macro F1 = {test_f1*100:.2f}%, Loss = {test_loss:.4f}")
        if classification_report:
            print(classification_report(targets, preds, target_names=TARGET_NAMES, zero_division=0))
        print("=" * 60)

    return model


def generate_synthetic_dataset(num_samples=120, min_m=25, max_m=50, feature_dim=1280):
    """Generates synthetic residue data for pipeline smoke-testing."""
    data = []
    for i in range(num_samples):
        m = np.random.randint(min_m, max_m + 1)
        data.append({
            'protein_id': f'SYNTH_{i:04d}',
            'residue_features': torch.randn(m, feature_dim),
            'residue_coords': torch.rand(m, 3) * 30.0,
            'full_protein_feature': torch.randn(feature_dim),
            'label': torch.tensor(np.random.randint(0, 5), dtype=torch.long),
            'residue_metadata': [{'resseq': str(j + 1), 'resname': 'ALA', 'chain': 'A'} for j in range(m)]
        })
    return data


def main():
    parser = argparse.ArgumentParser(description="Train AMICO Residue-Level Attention Models")
    parser.add_argument('--model-type', type=str, default='residue_ligand_cross',
                        choices=['residue_ligand_cross', 'residue_self_attention'],
                        help="Model architecture: 'residue_ligand_cross' or 'residue_self_attention'")
    parser.add_argument('--dataset-path', type=str, default=None,
                        help="Path to precomputed residue dataset (.pt)")
    parser.add_argument('--synthetic-demo', action='store_true',
                        help="Run smoke-test training on generated synthetic residue data")
    parser.add_argument('--config-json', type=str, default=None,
                        help="Path to hyperparameters JSON configuration file")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--weight-decay', type=float, default=1e-5, help="Weight decay for AdamW")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout probability")
    parser.add_argument('--label-smoothing', type=float, default=0.15, help="Label smoothing")
    parser.add_argument('--hidden-dim', type=int, default=256, help="Hidden projection dimension")
    parser.add_argument('--num-heads', type=int, default=4, help="Number of attention heads")
    parser.add_argument('--no-3d-bias', action='store_true', help="Disable 3D pairwise distance attention bias")
    parser.add_argument('--bias-3d-mode', type=str, default='rbf_gaussian',
                        choices=['rbf_gaussian', 'linear_decay'],
                        help="3D geometric bias kernel mode")
    parser.add_argument('--patience', type=int, default=12, help="Early stopping patience")
    parser.add_argument('--save-path', type=str, default='residue_model_best.pt',
                        help="Path to save best checkpoint")
    parser.add_argument('--split-suffix', type=str, default=None,
                        help="Structural clustering split suffix (e.g. 'mil_0.5_nr0.9' or 'mil0.5_nr0.9')")

    args = parser.parse_args()

    # Load JSON config if provided
    if args.config_json and os.path.exists(args.config_json):
        print(f"Loading configuration from {args.config_json}...")
        with open(args.config_json, 'r') as f:
            cfg = json.load(f)
        args.hidden_dim = cfg.get('hidden_dim', args.hidden_dim)
        args.num_heads = cfg.get('num_heads', args.num_heads)
        args.lr = cfg.get('lr', args.lr)
        args.weight_decay = cfg.get('weight_decay', args.weight_decay)
        args.dropout = cfg.get('dropout', args.dropout)
        args.label_smoothing = cfg.get('label_smoothing', args.label_smoothing)
        args.batch_size = cfg.get('batch_size', args.batch_size)
        if 'model_type' in cfg:
            args.model_type = cfg['model_type']
        if 'split_suffix' in cfg and not args.split_suffix:
            args.split_suffix = cfg['split_suffix']

    use_3d_bias = not args.no_3d_bias

    # Load or generate dataset
    train_dataset, val_dataset, test_dataset = None, None, None

    if args.dataset_path and os.path.exists(args.dataset_path):
        print(f"Loading residue dataset from {args.dataset_path}...")
        raw_data = torch.load(args.dataset_path, map_location='cpu', weights_only=False)

        # Unpack items from file
        all_items = []
        if isinstance(raw_data, dict):
            if 'all' in raw_data and len(raw_data['all']) > 0:
                all_items = raw_data['all']
            elif 'train' in raw_data:
                all_items = list(raw_data['train']) + list(raw_data.get('val', [])) + list(raw_data.get('test', []))
            else:
                for v in raw_data.values():
                    if isinstance(v, list):
                        all_items.extend(v)
        elif isinstance(raw_data, list):
            all_items = raw_data
        else:
            raise ValueError(f"Unrecognized format in dataset file {args.dataset_path}")

        print(f"-> Total samples in dataset: {len(all_items)}")

        # Check if pre-existing splits in raw_data are valid and not overridden
        has_valid_splits = (
            isinstance(raw_data, dict)
            and 'train' in raw_data and len(raw_data['train']) > 0
            and 'val' in raw_data and len(raw_data['val']) > 0
            and not args.split_suffix
        )

        if has_valid_splits:
            print(f"-> Using pre-partitioned splits from dataset: Train={len(raw_data['train'])}, Val={len(raw_data.get('val', []))}, Test={len(raw_data.get('test', []))}")
            train_dataset = ResidueDataset(raw_data['train'])
            val_dataset = ResidueDataset(raw_data['val']) if 'val' in raw_data and len(raw_data['val']) > 0 else None
            test_dataset = ResidueDataset(raw_data['test']) if 'test' in raw_data and len(raw_data['test']) > 0 else None
        else:
            # Need to partition using split files
            split_suffix = args.split_suffix or 'mil_0.5_nr0.9'
            print(f"-> Partitioning dataset using split suffix '{split_suffix}'...")
            train_ids, val_ids, test_ids = load_split_ids(root_dir, split_suffix=split_suffix)

            if len(train_ids) > 0 and (len(val_ids) > 0 or len(test_ids) > 0):
                train_items, val_items, test_items, unassigned = [], [], [], []
                for item in all_items:
                    pid = item['protein_id']
                    if match_id(pid, train_ids):
                        train_items.append(item)
                    elif match_id(pid, val_ids):
                        val_items.append(item)
                    elif match_id(pid, test_ids):
                        test_items.append(item)
                    else:
                        unassigned.append(item)

                print(f"   Split breakdown ({split_suffix}) -> Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}")
                if unassigned:
                    print(f"   Note: {len(unassigned)} proteins were not in split lists (unassigned).")

                if len(val_items) == 0:
                    print(f"⚠️ Warning: Validation set has 0 samples for suffix '{split_suffix}'!")

                train_dataset = ResidueDataset(train_items)
                val_dataset = ResidueDataset(val_items) if len(val_items) > 0 else None
                test_dataset = ResidueDataset(test_items) if len(test_items) > 0 else None
            else:
                print("⚠️ No valid split files matched. Falling back to 80/10/10 split.")
                n_total = len(all_items)
                n_train = int(0.8 * n_total)
                n_val = int(0.1 * n_total)
                train_dataset = ResidueDataset(all_items[:n_train])
                val_dataset = ResidueDataset(all_items[n_train:n_train + n_val])
                test_dataset = ResidueDataset(all_items[n_train + n_val:])
    elif args.synthetic_demo or not args.dataset_path:
        if not args.dataset_path:
            print("No --dataset-path provided. Running demonstration with synthetic dataset...")
        synth_data = generate_synthetic_dataset(num_samples=90)
        train_dataset = ResidueDataset(synth_data[:60])
        val_dataset = ResidueDataset(synth_data[60:75])
        test_dataset = ResidueDataset(synth_data[75:])
        if args.epochs == 50 and not args.dataset_path:
            args.epochs = 5  # Quick demonstration default

    train_residue_model(
        model_type=args.model_type,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        use_3d_bias=use_3d_bias,
        bias_3d_mode=args.bias_3d_mode,
        patience=args.patience,
        save_path=args.save_path
    )


if __name__ == '__main__':
    main()
