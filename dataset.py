import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import defaultdict

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

import re
import glob


def normalize_split_tag(s):
    if not s:
        return ''
    s = os.path.basename(s)
    s = re.sub(r'^(train|validation|val|test)[_-]?', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\.txt$', '', s, flags=re.IGNORECASE)
    return re.sub(r'[^a-zA-Z0-9]', '', s).lower()


def load_split_ids(base_dir, split_suffix='_mil_0.5', use_nr=False):
    """
    Loads protein identifiers for train/val/test splits based on the specified split suffix.
    Supports exact matching as well as normalized fuzzy matching (e.g. 'mil0.5_nr0.9' matches 'mil_0.5_nr0.9').
    """
    search_dirs = [
        os.path.join(base_dir, 'data_prep'),
        base_dir,
        os.path.abspath('data_prep'),
        os.path.abspath('.')
    ]
    unique_dirs = []
    for d in search_dirs:
        if d and os.path.exists(d) and d not in unique_dirs:
            unique_dirs.append(d)

    # 1. Try direct exact match with prefix '_'
    exact_candidates = [split_suffix]
    if not split_suffix.startswith('_'):
        exact_candidates.append(f'_{split_suffix}')
    else:
        exact_candidates.append(split_suffix.lstrip('_'))

    train_path, val_path, test_path = None, None, None

    for cand_suf in exact_candidates:
        for d in unique_dirs:
            cand_t = os.path.join(d, f'train{cand_suf}.txt')
            if not os.path.exists(cand_t):
                cand_t = os.path.join(d, f'train_{cand_suf}.txt')
            if os.path.exists(cand_t):
                train_path = cand_t
                actual_suffix = os.path.basename(cand_t)[len('train'):].replace('.txt', '')
                for v_name in [f'validation{actual_suffix}.txt', f'val{actual_suffix}.txt']:
                    if os.path.exists(os.path.join(d, v_name)):
                        val_path = os.path.join(d, v_name)
                        break
                cand_test = os.path.join(d, f'test{actual_suffix}.txt')
                if os.path.exists(cand_test):
                    test_path = cand_test
                break
        if train_path:
            break

    # 2. If not found, use normalized matching across all existing train*.txt files
    if not train_path:
        norm_query = normalize_split_tag(split_suffix)
        for d in unique_dirs:
            train_files = glob.glob(os.path.join(d, 'train*.txt'))
            for tf in train_files:
                norm_file = normalize_split_tag(tf)
                if norm_file == norm_query or norm_query in norm_file or norm_file in norm_query:
                    train_path = tf
                    suffix_from_file = os.path.basename(tf)[len('train'):].replace('.txt', '')
                    for v_name in [f'validation{suffix_from_file}.txt', f'val{suffix_from_file}.txt']:
                        if os.path.exists(os.path.join(d, v_name)):
                            val_path = os.path.join(d, v_name)
                            break
                    cand_test = os.path.join(d, f'test{suffix_from_file}.txt')
                    if os.path.exists(cand_test):
                        test_path = cand_test
                    break
            if train_path:
                break

    def read_ids(path):
        if not path or not os.path.exists(path):
            return set()
        with open(path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())

    train_ids = read_ids(train_path)
    val_ids = read_ids(val_path)
    test_ids = read_ids(test_path)

    if len(train_ids) == 0:
        available_files = []
        for d in unique_dirs:
            available_files.extend(os.path.basename(f) for f in glob.glob(os.path.join(d, 'train*.txt')))
        print(f"⚠️ Warning: Could not find split files for '{split_suffix}'. Available train files: {available_files}")
    else:
        print(f"-> Found split files for '{split_suffix}':")
        print(f"   Train: {train_path} ({len(train_ids)} IDs)")
        print(f"   Val:   {val_path} ({len(val_ids)} IDs)")
        print(f"   Test:  {test_path} ({len(test_ids)} IDs)")

    return train_ids, val_ids, test_ids


def match_id(pid, id_set):
    """Checks whether a protein ID matches any entry in id_set, handling suffixes like _MERGED, fragments _F1, and .pdb."""
    clean_p = pid.replace('.pdb', '')
    if clean_p in id_set:
        return True
    base = clean_p.split('_')[0]
    return base in id_set


def load_cross_mil_data(pockets_path, full_proteins_path, mode='pockets', min_prob=0.0, max_pockets=None, return_config=False):
    """
    Loads ESM pocket embeddings and ESM whole-protein embeddings, pairing them into MIL bags.
    Supports filtering pockets by minimum probability threshold (min_prob) and capping max pockets per protein.
    """
    print(f"Loading pocket features from {pockets_path} (min_prob={min_prob})...")
    raw_pockets = torch.load(pockets_path, weights_only=False)
    
    print(f"Loading full protein features from {full_proteins_path}...")
    full_proteins = torch.load(full_proteins_path, weights_only=False)
    
    bags_dict = defaultdict(list)
    labels_dict = {}
    missing_full_prot = 0
    filtered_prob_count = 0
    prep_cfg = None
    
    for item in raw_pockets:
        if prep_cfg is None and 'preprocessing' in item:
            prep_cfg = item['preprocessing']

        # Filter by minimum pocket probability threshold (e.g. min_prob=0.30)
        prob = item.get('probability', item.get('prob', None))
        if prob is not None and min_prob > 0.0:
            try:
                if float(prob) < float(min_prob):
                    filtered_prob_count += 1
                    continue
            except (ValueError, TypeError):
                pass

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
        
        # Limit max pockets per protein if specified
        if max_pockets is not None and pid in bags_dict and len(bags_dict[pid]) >= max_pockets:
            continue

        if mode == 'pockets':
            if feat.ndim > 1:
                feat = feat.mean(dim=0)
            bags_dict[pid].append(feat.cpu().numpy() if torch.is_tensor(feat) else np.array(feat))
        else:
            bags_dict[pid].append(feat.cpu().numpy() if torch.is_tensor(feat) else np.array(feat))
            
    if filtered_prob_count > 0:
        print(f"-> Filtered out {filtered_prob_count} pockets with probability < {min_prob}.")
    if missing_full_prot > 0:
        print(f"Warning: Missing full protein embedding for {missing_full_prot} pockets.")
        
    bag_list = []
    for pid, p_list in bags_dict.items():
        if not p_list:
            continue
        if mode == 'pockets':
            pocket_features = torch.FloatTensor(np.stack(p_list))
        else:
            pocket_features = torch.FloatTensor(np.concatenate(p_list, axis=0))
            
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
        
    print(f"Successfully loaded {len(bag_list)} paired protein bags (min_prob={min_prob}).")
    
    if return_config:
        if prep_cfg is not None and isinstance(prep_cfg, dict) and min_prob > 0.0:
            prep_cfg = dict(prep_cfg)
            prep_cfg['min_prob'] = min_prob
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
