import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

from residue_attention.geometric_bias import Pairwise3DDistanceBias
from residue_attention.model_residue_self_attention import MultiHeadAttentionWith3DBias

# Canonical target cofactors and their SMILES
COFACTORS = {
    'acetyl-CoA': r'CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@H]([C@H](O)[C@@H]1OP(=O)(O)O)n2cnc3c(N)ncnc23)[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)C',
    'ATP': r'Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O',
    'B12': r'CC1=CC2=C(C=C1C)[N+](=CN2)[C@@H]3[C@@H]([C@@H]([C@H](O3)CO)OP(=O)(O)O[C@H](C)CNC(=O)CC[C@@]4([C@H]([C@@H]5[C@]6([C@@]([C@@H](C(=N6)/C(=C\7/[C@@]([C@@H](/C(=C/C8=N/C(=C(\C4=N5)/C)/[C@H](C8(C)C)CCC(=N)[O-])/N7)CCC(=N)[O-])(C)CC(=O)N)/C)CCC(=N)[O-])(C)CC(=O)N)C)CC(=O)N)C)O.[C]#N.[Co+2]',
    'FAD': r'Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]4O[C@H](n5cnc6c(N)ncnc65)[C@H](O)[C@@H]4O)c2cc1C',
    'NAD': r'NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1'
}

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']


def generate_ecfp4_fingerprints(radius=2, n_bits=2048):
    """Generates 2048-bit Morgan ECFP4 fingerprints for all 5 target cofactors."""
    fps = []
    try:
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        has_generator = True
    except (ImportError, AttributeError):
        has_generator = False

    for name in TARGET_NAMES:
        smi = COFACTORS[name]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
        if has_generator and mol is not None:
            arr = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.float32)
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return torch.tensor(np.stack(fps), dtype=torch.float32)


class ResidueLigandCrossAttentionMIL(nn.Module):
    """
    Residue-Level Ligand Cross-Attention Multi-Instance Learning Model.
    ==================================================================
    Architecture:
    1. Stage 1: 3D Spatial Residue Encoder (Optional but recommended):
       Pocket residues undergo 3D geometric self-attention with pairwise distance bias,
       allowing residues within 4-10 Angstroms (catalytic triads, binding clefts)
       to exchange structural context.
    2. Stage 2: Chemical Ligand Cross-Attention:
       5 ECFP4 cofactor fingerprints act as Queries (Q) attending over Keys & Values (K, V)
       consisting of [Token 0 (Protein Context), Residues 1..M].
    3. Stage 3: Transformer FFN + Scorer:
       Produces 5 logits for the 5 cofactors.
       
    Direct Key Residue Attribution:
    Cross-attention weights A[b, c, 1..M] directly quantify the attention paid by
    cofactor 'c' (e.g. ATP) to each individual residue 'r_i', providing direct,
    cofactor-specific key residue identification.
    """
    def __init__(
        self,
        feature_dim=1280,
        ecfp_dim=2048,
        hidden_dim=256,
        num_heads=4,
        num_classes=5,
        dropout=0.2,
        use_3d_bias=True,
        bias_3d_mode='rbf_gaussian',
        num_rbf=16,
        max_distance=25.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.use_3d_bias = use_3d_bias

        # Feature projections
        self.residue_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.protein_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.ligand_proj = nn.Sequential(
            nn.Linear(ecfp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Buffer of 5 cofactor fingerprints
        cofactor_fps = generate_ecfp4_fingerprints(n_bits=ecfp_dim)
        self.register_buffer('cofactor_fps', cofactor_fps)

        # Stage 1: 3D Geometric Self-Attention Encoder (Residue <-> Residue in 3D space)
        if use_3d_bias:
            self.geometric_bias = Pairwise3DDistanceBias(
                num_heads=num_heads,
                mode=bias_3d_mode,
                num_rbf=num_rbf,
                max_distance=max_distance
            )
            self.stage1_self_attn = MultiHeadAttentionWith3DBias(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            self.stage1_norm = nn.LayerNorm(hidden_dim)
        else:
            self.geometric_bias = None
            self.stage1_self_attn = None
            self.stage1_norm = None

        # Stage 2: Cross-Attention (Queries = Ligands, Keys/Values = Residues + Protein)
        self.cross_attn = MultiHeadAttentionWith3DBias(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Stage 3: FFN Block + Residuals
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Final Scorer -> [B, 5]
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, residue_features, padding_mask, full_protein_feature, residue_coords=None):
        """
        Args:
            residue_features: [B, M, 1280]
            padding_mask: [B, M] bool (True where padded)
            full_protein_feature: [B, 1280]
            residue_coords: [B, M, 3] optional C-alpha 3D coordinates
            
        Returns:
            logits: [B, 5]
            cross_attn_weights: [B, num_heads, 5, M + 1]
            stage1_attn_weights: [B, num_heads, M + 1, M + 1] or None
        """
        B, M, _ = residue_features.size()

        # Project protein and residue tokens
        p_tok = self.protein_proj(full_protein_feature).unsqueeze(1)  # [B, 1, d]
        r_toks = self.residue_proj(residue_features)                  # [B, M, d]
        kv = torch.cat([p_tok, r_toks], dim=1)                       # [B, M + 1, d]

        # Padding mask
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=padding_mask.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)       # [B, M + 1]

        stage1_weights = None
        # Stage 1: 3D Spatial Contextualization
        if self.use_3d_bias and residue_coords is not None and self.geometric_bias is not None:
            bias_3d = self.geometric_bias(residue_coords, has_cls_token=True)  # [B, H, M+1, M+1]
            s1_out, stage1_weights = self.stage1_self_attn(
                q=kv, k=kv, v=kv,
                bias_3d=bias_3d,
                key_padding_mask=full_mask
            )
            kv = self.stage1_norm(kv + s1_out)

        # Stage 2: Ligand Cross-Attention
        # Queries: 5 cofactors
        q_ligands = self.ligand_proj(self.cofactor_fps).unsqueeze(0).expand(B, -1, -1)  # [B, 5, d]

        cross_out, cross_weights = self.cross_attn(
            q=q_ligands,
            k=kv,
            v=kv,
            bias_3d=None,
            key_padding_mask=full_mask
        )  # cross_out: [B, 5, d], cross_weights: [B, H, 5, M + 1]

        # Stage 3: Residuals + FFN
        out = self.norm1(cross_out + q_ligands)
        out = self.norm2(out + self.ffn(out))

        logits = self.scorer(out).squeeze(-1)  # [B, 5]

        return logits, cross_weights, stage1_weights

    def get_residue_importance(self, cross_weights, target_class):
        """
        Extracts residue importance scores for a given cofactor class.
        
        Args:
            cross_weights: [B, num_heads, 5, M + 1]
            target_class: int or Tensor [B] of class indices (0..4)
            
        Returns:
            residue_weights: [B, M] normalized attention paid to residues 1..M
        """
        B = cross_weights.size(0)
        # Average across attention heads: [B, 5, M + 1]
        mean_cross = cross_weights.mean(dim=1)

        # Slice attention for the specified cofactor class
        if isinstance(target_class, int):
            c_idx = target_class
            # Attention to residues (Cols 1..M): [B, M]
            res_attn = mean_cross[:, c_idx, 1:]
        else:
            # Batch of classes: [B]
            res_attn = []
            for b in range(B):
                c = target_class[b].item() if torch.is_tensor(target_class) else target_class[b]
                res_attn.append(mean_cross[b, c, 1:])
            res_attn = torch.stack(res_attn, dim=0)

        # Normalize sum to 1.0 across residues
        denom = res_attn.sum(dim=-1, keepdim=True) + 1e-9
        return res_attn / denom


__all__ = [
    'COFACTORS',
    'TARGET_NAMES',
    'generate_ecfp4_fingerprints',
    'ResidueLigandCrossAttentionMIL'
]
