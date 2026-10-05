import torch
import torch.nn as nn

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']


class SelfAttentionMIL(nn.Module):
    """
    Self-Attention Multi-Instance Learning (MIL) Model.
    
    1. Global Context / CLS Token (Index 0): Whole-protein ESM-2 sequence embedding.
    2. Instance Tokens (Indices 1..N): Candidate 3D binding pockets from P2Rank.
    3. Multi-Head Self-Attention: Contextual interaction between protein and pockets.
    4. Transformer FFN + Residual LayerNorms.
    5. Classification Head: Projects updated CLS token to cofactor logits.
    """
    def __init__(self, feature_dim=1280, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.pocket_proj = nn.Linear(feature_dim, hidden_dim)
        self.protein_proj = nn.Linear(feature_dim, hidden_dim)
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, pocket_features, padding_mask, full_protein_feature=None):
        B, N, _ = pocket_features.size()
        
        pockets = self.pocket_proj(pocket_features)
        
        if full_protein_feature is not None:
            cls_token = self.protein_proj(full_protein_feature).unsqueeze(1)
            x = torch.cat([cls_token, pockets], dim=1)
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=padding_mask.device)
            full_mask = torch.cat([cls_mask, padding_mask], dim=1)
        else:
            x = pockets
            full_mask = padding_mask
            
        attn_out, attn_weights = self.self_attn(
            query=x, 
            key=x, 
            value=x, 
            key_padding_mask=full_mask
        )
        
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        
        if full_protein_feature is not None:
            cls_out = x[:, 0, :]
        else:
            cls_out = x.mean(dim=1)
            
        logits = self.classifier(cls_out)
        return logits, attn_weights


__all__ = [
    'TARGET_NAMES',
    'SelfAttentionMIL'
]
