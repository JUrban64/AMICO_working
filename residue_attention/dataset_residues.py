import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']


class ResidueDataset(Dataset):
    """
    Dataset for Residue-Level Multi-Instance Learning.
    Each item contains:
    - protein_id: identifier (e.g. UniProt / PDB stem)
    - residue_features: [M, 1280] ESM-2 per-residue embeddings
    - residue_coords: [M, 3] C-alpha coordinates in Angstroms
    - full_protein_feature: [1280] ESM-2 whole sequence context embedding
    - label: LongTensor (0..4)
    - residue_metadata: list of dicts with residue information (chain, resseq, resname, etc.)
    """
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def residue_collate_fn(batch):
    """
    Collate function that dynamically pads residue sequences and coordinates to batch maximum.
    
    Returns:
        padded_features: [B, max_M, 1280]
        padding_mask: [B, max_M] bool (True = padded)
        full_proteins: [B, 1280]
        padded_coords: [B, max_M, 3]
        labels: [B]
        metadata_list: list of length B (per-protein residue metadata lists)
        protein_ids: list of length B
    """
    res_feats_list = [item['residue_features'] for item in batch]
    res_coords_list = [item['residue_coords'] for item in batch]
    full_prots_list = [item['full_protein_feature'] for item in batch]
    labels_list = [item['label'] for item in batch]
    meta_list = [item.get('residue_metadata', []) for item in batch]
    pids_list = [item.get('protein_id', 'unknown') for item in batch]

    # Convert to FloatTensor if needed
    res_feats_list = [
        torch.as_tensor(rf, dtype=torch.float32) for rf in res_feats_list
    ]
    res_coords_list = [
        torch.as_tensor(rc, dtype=torch.float32) for rc in res_coords_list
    ]

    # Pad residue features to [B, max_M, 1280]
    padded_features = pad_sequence(res_feats_list, batch_first=True, padding_value=0.0)
    
    # Pad coordinates to [B, max_M, 3]
    padded_coords = pad_sequence(res_coords_list, batch_first=True, padding_value=0.0)

    # Compute lengths and padding mask
    lengths = torch.tensor([rf.size(0) for rf in res_feats_list], dtype=torch.long)
    max_len = padded_features.size(1)
    padding_mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)

    # Stack whole-protein embeddings & labels
    full_proteins = torch.stack([
        torch.as_tensor(p, dtype=torch.float32) for p in full_prots_list
    ], dim=0)

    labels = torch.cat([
        torch.as_tensor(l, dtype=torch.long).view(-1) for l in labels_list
    ], dim=0)

    return padded_features, padding_mask, full_proteins, padded_coords, labels, meta_list, pids_list
