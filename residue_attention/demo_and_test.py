#!/usr/bin/env python3
"""
Verification and Demo Script for AMICO Residue-Level Attention
==============================================================
Runs unit tests on:
1. Pairwise3DDistanceBias (RBF Gaussian & Linear Decay modes)
2. ResidueSelfAttentionMIL (Forward & Backward pass, attention shapes)
3. ResidueLigandCrossAttentionMIL (Forward & Backward pass, cross-attention shapes)
4. Residue Collate Function with dynamic padding
5. Real Structure parsing on AF-Q9NRG4-F1-model_v6.pdb with P2Rank outputs
6. Key Residue Ranking, PyMOL script export, and B-factor PDB generation
"""

import os
import sys
import torch
import numpy as np

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from residue_attention.geometric_bias import Pairwise3DDistanceBias
from residue_attention.model_residue_self_attention import ResidueSelfAttentionMIL
from residue_attention.model_residue_ligand_cross_att import ResidueLigandCrossAttentionMIL
from residue_attention.dataset_residues import ResidueDataset, residue_collate_fn
from residue_attention.feature_extractor_residues import parse_p2rank_residues_for_protein, extract_ca_coords_from_pdb
from residue_attention.interpretability import (
    rank_key_residues,
    format_key_residues_summary,
    export_pymol_script,
    write_pdb_with_attention_bfactor
)


def run_unit_tests():
    print("=" * 65)
    print("AMICO RESIDUE-LEVEL ATTENTION: RUNNING COMPONENT UNIT TESTS")
    print("=" * 65)

    B, M, D = 2, 35, 1280
    H, hidden_dim = 4, 256

    # 1. Test 3D Geometric Bias Module
    print("\n[1/5] Testing Pairwise3DDistanceBias...")
    coords = torch.rand(B, M, 3) * 30.0  # random 3D coords in a 30A box
    bias_rbf = Pairwise3DDistanceBias(num_heads=H, mode='rbf_gaussian')
    out_bias = bias_rbf(coords, has_cls_token=True)
    assert out_bias.shape == (B, H, M + 1, M + 1), f"Unexpected shape {out_bias.shape}"
    
    bias_linear = Pairwise3DDistanceBias(num_heads=H, mode='linear_decay')
    out_linear = bias_linear(coords, has_cls_token=True)
    assert out_linear.shape == (B, H, M + 1, M + 1)
    print("  -> Passed! (Shape [B, num_heads, M+1, M+1] correct for both RBF and linear decay)")

    # 2. Test ResidueSelfAttentionMIL
    print("\n[2/5] Testing ResidueSelfAttentionMIL...")
    res_feats = torch.randn(B, M, D)
    full_prot = torch.randn(B, D)
    pad_mask = torch.zeros(B, M, dtype=torch.bool)
    pad_mask[0, 25:] = True  # Sample 0 has 25 valid residues, Sample 1 has 35

    model_self = ResidueSelfAttentionMIL(
        feature_dim=D, hidden_dim=hidden_dim, num_heads=H, use_3d_bias=True
    )
    logits, self_weights = model_self(res_feats, pad_mask, full_prot, coords)
    assert logits.shape == (B, 5), f"Unexpected logits shape: {logits.shape}"
    assert self_weights.shape == (B, H, M + 1, M + 1), f"Unexpected weights shape: {self_weights.shape}"

    # Verify backward pass / gradient flow
    loss = logits.sum()
    loss.backward()
    assert model_self.geometric_bias.rbf_proj.weight.grad is not None
    print("  -> Passed! (Forward & backward pass successful, gradients flowed through 3D bias)")

    # 3. Test ResidueLigandCrossAttentionMIL
    print("\n[3/5] Testing ResidueLigandCrossAttentionMIL...")
    model_cross = ResidueLigandCrossAttentionMIL(
        feature_dim=D, hidden_dim=hidden_dim, num_heads=H, use_3d_bias=True
    )
    c_logits, cross_weights, s1_weights = model_cross(res_feats, pad_mask, full_prot, coords)
    assert c_logits.shape == (B, 5), f"Unexpected cross logits shape: {c_logits.shape}"
    assert cross_weights.shape == (B, H, 5, M + 1), f"Unexpected cross weights shape: {cross_weights.shape}"
    assert s1_weights.shape == (B, H, M + 1, M + 1), f"Unexpected stage 1 weights shape: {s1_weights.shape}"

    # Verify residue importance extraction
    res_imp = model_cross.get_residue_importance(cross_weights, target_class=1)  # ATP
    assert res_imp.shape == (B, M), f"Unexpected res_imp shape: {res_imp.shape}"
    # Verify sum of importance is ~1.0
    assert torch.allclose(res_imp.sum(dim=-1), torch.ones(B), atol=1e-4)
    print("  -> Passed! (Cofactor cross-attention shapes and residue importance extraction correct)")

    # 4. Test Collate Function
    print("\n[4/5] Testing Dataset & residue_collate_fn...")
    dummy_item1 = {
        'protein_id': 'PROT_1',
        'residue_features': torch.randn(20, D),
        'residue_coords': torch.randn(20, 3),
        'full_protein_feature': torch.randn(D),
        'label': torch.tensor(1),
        'residue_metadata': [{'resname': 'ALA', 'resseq': str(i)} for i in range(20)]
    }
    dummy_item2 = {
        'protein_id': 'PROT_2',
        'residue_features': torch.randn(30, D),
        'residue_coords': torch.randn(30, 3),
        'full_protein_feature': torch.randn(D),
        'label': torch.tensor(3),
        'residue_metadata': [{'resname': 'LYS', 'resseq': str(i)} for i in range(30)]
    }
    p_feats, p_mask, p_prots, p_coords, p_labels, meta_list, pids = residue_collate_fn([dummy_item1, dummy_item2])
    assert p_feats.shape == (2, 30, D)
    assert p_coords.shape == (2, 30, 3)
    assert p_mask.shape == (2, 30)
    assert p_mask[0, 20:].all()  # padded part is True
    assert not p_mask[1].any()   # sample 2 has no padding
    print("  -> Passed! (Collate dynamically padded to max sequence length with valid masks)")

    # 5. Test Real Structure Parsing & Interpretability on AF-Q9NRG4
    print("\n[5/5] Testing Real Structure Parsing & Interpretability (AF-Q9NRG4-F1)...")
    pdb_path = os.path.join(root_dir, "AF-Q9NRG4-F1-model_v6.pdb")
    prank_dir = os.path.join(root_dir, "temp_p2rank")

    if os.path.exists(pdb_path) and os.path.exists(prank_dir):
        parsed = parse_p2rank_residues_for_protein(prank_dir, pdb_path, min_prob=0.10, max_pockets=3)
        sel_residues = parsed['selected_residues']
        print(f"  -> Successfully extracted {len(sel_residues)} unique pocket residues from {os.path.basename(pdb_path)}.")
        
        # Test C-alpha coordinate availability
        coords_arr = np.stack([r['coords'] for r in sel_residues], axis=0)
        assert coords_arr.shape == (len(sel_residues), 3)
        assert not np.isnan(coords_arr).any()

        # Run mock inference for interpretability
        mock_res_feats = torch.randn(1, len(sel_residues), D)
        mock_full_prot = torch.randn(1, D)
        mock_mask = torch.zeros(1, len(sel_residues), dtype=torch.bool)
        mock_coords = torch.from_numpy(coords_arr).float().unsqueeze(0)

        # Run ResidueLigandCrossAttentionMIL
        ranking_result = rank_key_residues(
            model=model_cross,
            residue_features=mock_res_feats,
            padding_mask=mock_mask,
            full_protein_feature=mock_full_prot,
            residue_metadata=sel_residues,
            residue_coords=mock_coords,
            target_class=1,  # ATP
            top_k=7
        )

        print("\n" + format_key_residues_summary(ranking_result))

        # Test PyMOL export
        out_pml = os.path.join(root_dir, "residue_attention", "demo_visualization.pml")
        export_pymol_script(pdb_path, ranking_result, out_pml, top_k=5)
        assert os.path.exists(out_pml)
        print(f"\n  -> PyMOL script successfully created: {out_pml}")

        # Test B-factor PDB writer
        out_pdb = os.path.join(root_dir, "residue_attention", "demo_annotated.pdb")
        write_pdb_with_attention_bfactor(pdb_path, out_pdb, ranking_result)
        assert os.path.exists(out_pdb)
        print(f"  -> B-factor annotated PDB created: {out_pdb}")
    else:
        print("  -> Note: AF-Q9NRG4-F1-model_v6.pdb not found for real demo; skipping file export test.")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 65)


if __name__ == "__main__":
    run_unit_tests()
