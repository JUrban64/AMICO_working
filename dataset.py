import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import defaultdict

from preprocessing import config_from_records, describe

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

def load_split_ids(base_dir, split_suffix='_mil_0.5', use_nr=False):
    """Loads protein identifiers for train/val/test splits based on the specified split suffix."""
    if not split_suffix.startswith('_'):
        split_suffix = f'_{split_suffix}'

    if use_nr and '_nr' not in split_suffix:
        nr_candidates = [
            f"{split_suffix}_nr0.95",
            f"{split_suffix}_nr0.9",
            f"{split_suffix}_nr"
        ]
        found = False
        for cand in nr_candidates:
            if os.path.exists(os.path.join(base_dir, f'data_prep/train{cand}.txt')) or os.path.exists(os.path.join(base_dir, f'train{cand}.txt')):
                split_suffix = cand
                found = True
                break
        if not found:
            dp_dir = os.path.join(base_dir, 'data_prep')
            if os.path.exists(dp_dir):
                for f in os.listdir(dp_dir):
                    if f.startswith(f'train{split_suffix}_nr') and f.endswith('.txt'):
                        split_suffix = f.replace('train', '').replace('.txt', '')
                        found = True
                        break
        if not found:
            split_suffix = f"{split_suffix}_nr"

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
    """Checks whether a protein ID matches any entry in id_set, handling suffixes like _MERGED, fragments _F1, and .pdb."""
    clean_p = pid.replace('.pdb', '')
    if clean_p in id_set:
        return True
    base = clean_p.split('_')[0]
    return base in id_set


def load_cross_mil_data(pockets_path, full_proteins_path, mode='pockets', return_config=False):
    """
    Loads ESM pocket embeddings and ESM whole-protein embeddings, pairing them into MIL bags.

    If return_config is True, returns (bag_list, preprocessing_config), where the config
    describes how the features were built (min_prob, pocket_embedding, ...). Datasets built
    before configs were recorded are reported as LEGACY_PREPROCESSING.
    """
    print(f"Loading pocket features from {pockets_path}...")
    raw_pockets = torch.load(pockets_path, weights_only=False)
    prep_cfg = config_from_records(raw_pockets)
    if prep_cfg is not None:
        print(f"Dataset preprocessing: {describe(prep_cfg)}")
    
    print(f"Loading full protein features from {full_proteins_path}...")
    full_proteins = torch.load(full_proteins_path, weights_only=False)
    
    bags_dict = defaultdict(list)
    labels_dict = {}
    missing_full_prot = 0
    
    for item in raw_pockets:
        raw_pid = item['protein_id']
        base_name = os.path.basename(raw_pid)
        pid = base_name.split('_pocket_')[0].replace('.pdb', '').replace('_prank_output', '')
        
        # Verify presence of full protein embedding (supporting fragments like _F1)
        if pid not in full_proteins:
            clean_pid = pid.split('_')[0]
            if clean_pid in full_proteins:
                full_proteins[pid] = full_proteins[clean_pid]
            else:
                missing_full_prot += 1
                continue
            
        feat = item['features']  # [num_residues, 1280] or [1280]
        label = item['label']
        labels_dict[pid] = label
        
        if mode == 'pockets':
            if feat.ndim > 1:
                feat = feat.mean(dim=0)
            bags_dict[pid].append(feat.cpu().numpy() if torch.is_tensor(feat) else np.array(feat))
        else:
            bags_dict[pid].append(feat.cpu().numpy() if torch.is_tensor(feat) else np.array(feat))
            
    if missing_full_prot > 0:
        print(f"Warning: Missing full protein embedding for {missing_full_prot} pockets.")
        
    bag_list = []
    for pid in bags_dict:
        if mode == 'pockets':
            pocket_features = torch.FloatTensor(np.stack(bags_dict[pid]))
        else:
            pocket_features = torch.FloatTensor(np.concatenate(bags_dict[pid], axis=0))
            
        full_protein_feat = full_proteins.get(pid, full_proteins.get(pid.split('_')[0]))
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
        
    print(f"Successfully loaded {len(bag_list)} paired protein bags.")
    if return_config:
        return bag_list, prep_cfg
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
    Pads pocket sequences to uniform batch length and constructs the attention padding mask.
    """
    pocket_features_list = [item['pocket_features'] for item in batch]
    full_protein_list = [item['full_protein_feature'] for item in batch]
    labels_list = [item['label'] for item in batch]
    
    padded_pockets = pad_sequence(pocket_features_list, batch_first=True, padding_value=0.0)  # [B, max_N, 1280]
    lengths = torch.tensor([pf.size(0) for pf in pocket_features_list])
    max_len = padded_pockets.size(1)
    
    padding_mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)  # [B, max_N]
    
    full_proteins = torch.stack(full_protein_list, dim=0)  # [B, 1280]
    labels = torch.cat(labels_list, dim=0)                 # [B]
    
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
            
    print(f"Splits ({split_suffix}) -> Train: {len(train_bags)}, Val: {len(val_bags)}, Test: {len(test_bags)}")
    return train_bags, val_bags, test_bags
