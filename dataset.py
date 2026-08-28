import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import defaultdict

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

def load_split_ids(base_dir, split_suffix='_mil_0.5', use_nr=False):
    """Načte ID proteinů pro train/val/test podle zadaného split suffixu."""
    if not split_suffix.startswith('_'):
        split_suffix = f'_{split_suffix}'

    train_path = os.path.join(base_dir, f'data_prep/train{split_suffix}.txt')
    val_path = os.path.join(base_dir, f'data_prep/validation{split_suffix}.txt')
    test_path = os.path.join(base_dir, f'data_prep/test{split_suffix}.txt')

    if not os.path.exists(train_path):
        train_path = os.path.join(base_dir, f'train{split_suffix}.txt')
        val_path = os.path.join(base_dir, f'validation{split_suffix}.txt')
        test_path = os.path.join(base_dir, f'test{split_suffix}.txt')

    def read_ids(path):
        if not os.path.exists(path):
            return set()
        with open(path, 'r') as f:
            return set(line.strip() for line in f if line.strip())

    return read_ids(train_path), read_ids(val_path), read_ids(test_path)


def match_id(pid, id_set):
    """Zkontroluje shodu ID proteinu (včetně ošetření přípon typu _MERGED)."""
    if pid in id_set:
        return True
    base = pid.split('_')[0]
    return base in id_set


def load_cross_mil_data(pockets_path, full_proteins_path, mode='pockets'):
    """
    Načte ESM pocket embeddingy a ESM full protein embeddingy a spáruje je.
    """
    print(f"Načítám pocket features z {pockets_path}...")
    raw_pockets = torch.load(pockets_path, weights_only=False)
    
    print(f"Načítám full protein features z {full_proteins_path}...")
    full_proteins = torch.load(full_proteins_path, weights_only=False)
    
    bags_dict = defaultdict(list)
    labels_dict = {}
    missing_full_prot = 0
    
    for item in raw_pockets:
        raw_pid = item['protein_id']
        base_name = os.path.basename(raw_pid)
        pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
        
        if pid not in full_proteins:
            missing_full_prot += 1
            continue
            
        feat = item['features'] # [num_residues, 1280]
        label = item['label']
        labels_dict[pid] = label
        
        if mode == 'pockets':
            feat = feat.mean(dim=0)
            bags_dict[pid].append(feat.numpy())
        else:
            bags_dict[pid].append(feat.numpy())
            
    if missing_full_prot > 0:
        print(f"Upozornění: U {missing_full_prot} kapes chyběl full protein embedding.")
        
    bag_list = []
    for pid in bags_dict:
        if mode == 'pockets':
            pocket_features = torch.FloatTensor(np.stack(bags_dict[pid]))
        else:
            pocket_features = torch.FloatTensor(np.concatenate(bags_dict[pid], axis=0))
            
        full_protein_feat = full_proteins[pid] # [1280]
        if isinstance(full_protein_feat, np.ndarray):
            full_protein_feat = torch.FloatTensor(full_protein_feat)
        elif not torch.is_tensor(full_protein_feat):
            full_protein_feat = torch.tensor(full_protein_feat, dtype=torch.float32)
            
        bag_list.append({
            'protein_id': pid,
            'pocket_features': pocket_features,
            'full_protein_feature': full_protein_feat,
            'label': torch.LongTensor([labels_dict[pid]])
        })
        
    print(f"Úspěšně načteno {len(bag_list)} spárovaných proteinů.")
    return bag_list


class CrossMilDataset(Dataset):
    def __init__(self, bags_list):
        self.bags = bags_list
        
    def __len__(self):
        return len(self.bags)
        
    def __getitem__(self, idx):
        return self.bags[idx]


def custom_collate_fn(batch):
    """
    Zarovná kapsy (padding) a vytvoří padding masku.
    """
    pocket_features_list = [item['pocket_features'] for item in batch]
    full_protein_list = [item['full_protein_feature'] for item in batch]
    labels_list = [item['label'] for item in batch]
    
    padded_pockets = pad_sequence(pocket_features_list, batch_first=True, padding_value=0.0) # [B, max_N, 1280]
    lengths = torch.tensor([pf.size(0) for pf in pocket_features_list])
    max_len = padded_pockets.size(1)
    
    padding_mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1) # [B, max_N]
    
    full_proteins = torch.stack(full_protein_list, dim=0) # [B, 1280]
    labels = torch.cat(labels_list, dim=0)                # [B]
    
    return padded_pockets, padding_mask, full_proteins, labels


def get_cross_mil_splits(pockets_path, full_proteins_path, base_dir, split_suffix='mil_0.5'):
    all_bags = load_cross_mil_data(pockets_path, full_proteins_path, mode='pockets')
    train_ids, val_ids, test_ids = load_split_ids(base_dir, split_suffix=split_suffix)
    
    train_bags, val_bags, test_bags = [], [], []
    for b in all_bags:
        pid = b['protein_id']
        if match_id(pid, train_ids):
            train_bags.append(b)
        elif match_id(pid, val_ids):
            val_bags.append(b)
        elif match_id(pid, test_ids):
            test_bags.append(b)
            
    print(f"Rozdělení ({split_suffix}) -> Train: {len(train_bags)}, Val: {len(val_bags)}, Test: {len(test_bags)}")
    return train_bags, val_bags, test_bags
