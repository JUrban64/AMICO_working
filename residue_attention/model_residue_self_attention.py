import torch
import torch.nn as nn
import torch.nn.functional as F

from residue_attention.geometric_bias import Pairwise3DDistanceBias

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']


class MultiHeadAttentionWith3DBias(nn.Module):
    """
    Multi-Head Attention supporting additive 3D spatial bias and key padding masks.
    """
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, bias_3d=None, key_padding_mask=None):
        """
        Args:
            q: [B, Sq, D]
            k: [B, Sk, D]
            v: [B, Sk, D]
            bias_3d: [B, num_heads, Sq, Sk] optional 3D distance bias
            key_padding_mask: [B, Sk] bool (True for padded positions)
            
        Returns:
            out: [B, Sq, D]
            attn_weights: [B, num_heads, Sq, Sk]
        """
        B, Sq, _ = q.size()
        _, Sk, _ = k.size()

        Q = self.q_proj(q).view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Sq, d]
        K = self.k_proj(k).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Sk, d]
        V = self.v_proj(v).view(B, Sk, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Sk, d]

        # Scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, Sq, Sk]

        # Add 3D geometric distance bias
        if bias_3d is not None:
            scores = scores + bias_3d

        # Apply padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, Sk] -> expand to [B, 1, 1, Sk]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # [B, H, Sq, Sk]
        dropped_weights = self.dropout(attn_weights)

        context = torch.matmul(dropped_weights, V)  # [B, H, Sq, d]
        context = context.transpose(1, 2).contiguous().view(B, Sq, self.embed_dim)
        out = self.out_proj(context)

        return out, attn_weights


class ResidueSelfAttentionMIL(nn.Module):
    """
    Residue-Level Self-Attention Multi-Instance Learning Model.
    ==========================================================
    1. Token 0: Whole-protein ESM-2 global context embedding [1280].
    2. Tokens 1..M: Individual binding pocket residues [1280].
    3. 3D Geometric Distance Bias: Pairwise spatial decay between residue C-alpha atoms.
    4. Multi-Head Self-Attention: Bidirectional contextual communication across all residues.
    5. Classification Head: Projects contextualized Token 0 into cofactor logits [B, 5].
    """
    def __init__(
        self,
        feature_dim=1280,
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
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
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

        # 3D Geometric Bias Module
        if use_3d_bias:
            self.geometric_bias = Pairwise3DDistanceBias(
                num_heads=num_heads,
                mode=bias_3d_mode,
                num_rbf=num_rbf,
                max_distance=max_distance
            )
        else:
            self.geometric_bias = None

        # Transformer Self-Attention Layer
        self.self_attn = MultiHeadAttentionWith3DBias(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Transformer FFN block + Norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, residue_features, padding_mask, full_protein_feature, residue_coords=None):
        """
        Args:
            residue_features: [B, M, 1280] candidate pocket residues
            padding_mask: [B, M] bool (True where padded)
            full_protein_feature: [B, 1280] global protein sequence context
            residue_coords: [B, M, 3] optional C-alpha 3D coordinates in Angstroms
            
        Returns:
            logits: [B, 5] cofactor specificity logits
            attn_weights: [B, num_heads, M+1, M+1] self-attention weights
        """
        B, M, _ = residue_features.size()

        # Project features
        r_tokens = self.residue_proj(residue_features)  # [B, M, d]
        p_token = self.protein_proj(full_protein_feature).unsqueeze(1)  # [B, 1, d]

        # Prepend Token 0 (Protein context)
        x = torch.cat([p_token, r_tokens], dim=1)  # [B, M + 1, d]

        # Padding mask for full sequence (Token 0 is never masked)
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=padding_mask.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)  # [B, M + 1]

        # Compute 3D geometric bias if enabled and coords provided
        bias_3d = None
        if self.use_3d_bias and residue_coords is not None and self.geometric_bias is not None:
            bias_3d = self.geometric_bias(residue_coords, has_cls_token=True)  # [B, H, M+1, M+1]

        # Multi-Head Self-Attention
        attn_out, attn_weights = self.self_attn(
            q=x,
            k=x,
            v=x,
            bias_3d=bias_3d,
            key_padding_mask=full_mask
        )

        # Residuals + Norms + FFN
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))

        # Classification from updated Token 0 (Global Context)
        cls_out = x[:, 0, :]
        logits = self.classifier(cls_out)

        return logits, attn_weights

    def get_residue_importance(self, attn_weights, target_class=None):
        """
        Extracts how strongly Token 0 (Context) attended to each residue across attention heads.
        
        Args:
            attn_weights: [B, num_heads, M+1, M+1]
            
        Returns:
            residue_weights: [B, M] attention weights normalized over residues 1..M
        """
        # Attention from Token 0 to Tokens 1..M: [B, num_heads, M]
        token0_to_res = attn_weights[:, :, 0, 1:]
        # Average across heads: [B, M]
        mean_weights = token0_to_res.mean(dim=1)
        # Normalize sum to 1.0 for interpretability
        denom = mean_weights.sum(dim=-1, keepdim=True) + 1e-9
        return mean_weights / denom


__all__ = [
    'TARGET_NAMES',
    'MultiHeadAttentionWith3DBias',
    'ResidueSelfAttentionMIL'
]
