import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Canonical target cofactors and their SMILES
COFACTORS = {
    'acetyl-CoA': r'CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@H]([C@H](O)[C@@H]1OP(=O)(O)O)n2cnc3c(N)ncnc23)[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)C',
    'ATP': r'Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O',
    'B12': r'CC1=CC2=C(C=C1C)[N+](=CN2)[C@@H]3[C@@H]([C@@H]([C@H](O3)CO)OP(=O)(O)O[C@H](C)CNC(=O)CC[C@@]4([C@H]([C@@H]5[C@]6([C@@]([C@@H](C(=N6)/C(=C\7/[C@@]([C@@H](/C(=C/C8=N/C(=C(\C4=N5)/C)/[C@H](C8(C)C)CCC(=N)[O-])/N7)CCC(=N)[O-])(C)CC(=O)N)/C)CCC(=N)[O-])(C)CC(=O)N)C)CC(=O)N)C)O.[C]#N.[Co+2]',
    'FAD': r'Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]4O[C@H](n5cnc6c(N)ncnc65)[C@H](O)[C@@H]4O)c2cc1C',
    'NAD': r'NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1'
}

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

def generate_ecfp4_fingerprints(radius=2, n_bits=1024):
    """Generates 1024-bit Morgan ECFP4 fingerprints for all target cofactors."""
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
            # Fallback for complex organometallics
            mol = Chem.MolFromSmiles(smi, sanitize=False)
        if has_generator and mol is not None:
            arr = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.float32)
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return torch.tensor(np.stack(fps), dtype=torch.float32)


class LigandCrossAttentionMIL(nn.Module):
    """
    Ligand-Protein Cross-Attention Multi-Instance Learning Model.
    
    Architecture:
    1. Keys & Values: Global protein sequence context (Token 0) + Candidate P2Rank 3D pockets (Tokens 1..N).
    2. Queries: 5 ECFP4 Morgan fingerprints corresponding to candidate cofactors.
    3. Multi-Head Cross-Attention: Chemically guides the model to attend to matching binding pockets.
    4. Canonical Transformer FFN with ReLU & Residual LayerNorm.
    5. Linear Scorer Head yielding logit predictions for all 5 cofactor classes.
    """
    def __init__(self, feature_dim=1280, ecfp_dim=1024, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 1. Projections into shared latent space
        self.pocket_proj = nn.Sequential(
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
        
        # 2. Registered static ECFP4 Morgan fingerprints
        cofactor_fps = generate_ecfp4_fingerprints(n_bits=ecfp_dim) # [5, ecfp_dim]
        self.register_buffer('cofactor_fps', cofactor_fps)
        
        # 3. Multi-Head Cross-Attention (Q: Ligands [5, d], K/V: Protein Context + Pockets [N+1, d])
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 4. Canonical Transformer Feed-Forward Network
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 5. Final Scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, pocket_features, padding_mask, full_protein_feature=None):
        """
        Args:
            pocket_features: [B, N, 1280] (ESM-2 embeddings for N candidate pockets per protein)
            padding_mask: [B, N] (True for padded pockets to ignore)
            full_protein_feature: [B, 1280] (Global ESM-2 sequence embedding of the whole protein)
            
        Returns:
            logits: [B, 5] (Classification logits for all 5 cofactors)
            attn_weights: [B, 5, N+1] (Attention distribution over [Protein_Token, Pocket_1, ..., Pocket_N])
        """
        B, N, _ = pocket_features.size()
        
        # 1. Project pockets to Keys & Values
        k = self.pocket_proj(pocket_features) # [B, N, hidden_dim]
        v = k
        
        # 2. Prepend whole-protein sequence embedding as context token at index 0
        if full_protein_feature is not None:
            prot_tok = self.protein_proj(full_protein_feature).unsqueeze(1) # [B, 1, hidden_dim]
            k = torch.cat([prot_tok, k], dim=1) # [B, N+1, hidden_dim]
            v = torch.cat([prot_tok, v], dim=1) # [B, N+1, hidden_dim]
            
            # Position 0 is valid protein context (False in padding mask)
            prot_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            mask = torch.cat([prot_mask, padding_mask], dim=1) # [B, N+1]
        else:
            mask = padding_mask
            
        # 3. Project candidate cofactors as Query vectors
        q_ligands = self.ligand_proj(self.cofactor_fps).unsqueeze(0).expand(B, -1, -1) # [B, 5, hidden_dim]
        
        # 4. Multi-Head Cross-Attention
        attn_out, attn_weights = self.cross_attn(
            query=q_ligands, 
            key=k, 
            value=v, 
            key_padding_mask=mask
        ) # attn_out: [B, 5, hidden_dim], attn_weights: [B, 5, N+1]
        
        # 5. Residual connection + FFN + Norm
        out = self.norm1(attn_out + q_ligands)
        out = self.norm2(out + self.ffn(out))
        
        # 6. Evaluate score for each (Protein, Ligand) pair: [B, 5, 1] -> [B, 5]
        logits = self.scorer(out).squeeze(-1)
        
        return logits, attn_weights


class SelfAttentionMIL(nn.Module):
    """
    Self-Attention Multi-Instance Learning Model.
    
    Architecture:
    1. Global Context / CLS Token (Index 0): Whole protein sequence embedding projected into hidden_dim.
    2. Instance Tokens (Indices 1..N): Candidate P2Rank 3D pockets projected into hidden_dim.
    3. Multi-Head Self-Attention: Learns mutual contextual interactions between the whole protein
       and all candidate pockets (Keys, Queries, Values all from [CLS, Pocket_1, ..., Pocket_N]).
    4. Canonical Transformer Feed-Forward Network (FFN) with GELU activations and Residual LayerNorms.
    5. Classification Head: Projects the contextualized CLS / protein token (index 0)
       to logits for the 5 target cofactor classes.
    """
    def __init__(self, feature_dim=1280, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 1. Projections from ESM embedding dimensions into hidden latent space
        self.pocket_proj = nn.Linear(feature_dim, hidden_dim)
        self.protein_proj = nn.Linear(feature_dim, hidden_dim)
        
        # 2. Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 3. Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 4. Canonical Transformer Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 5. Classifier Head acting on the updated CLS token
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, pocket_features, padding_mask, full_protein_feature=None):
        """
        Args:
            pocket_features: [B, N, 1280] (ESM-2 embeddings for N candidate pockets per protein)
            padding_mask: [B, N] (True for padded pockets to ignore)
            full_protein_feature: [B, 1280] (Global ESM-2 sequence embedding of the whole protein)
            
        Returns:
            logits: [B, 5] (Classification logits for all 5 cofactors)
            attn_weights: [B, N+1, N+1] or [B, N, N] (Self-attention weight matrix)
        """
        B, N, _ = pocket_features.size()
        
        # 1. Project candidate pockets
        pockets = self.pocket_proj(pocket_features) # [B, N, hidden_dim]
        
        # 2. Prepend whole-protein sequence embedding as CLS token at index 0
        if full_protein_feature is not None:
            cls_token = self.protein_proj(full_protein_feature).unsqueeze(1) # [B, 1, hidden_dim]
            x = torch.cat([cls_token, pockets], dim=1) # [B, N+1, hidden_dim]
            
            # Position 0 is valid protein context (False in padding mask)
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=padding_mask.device)
            full_mask = torch.cat([cls_mask, padding_mask], dim=1) # [B, N+1]
        else:
            x = pockets
            full_mask = padding_mask
            
        # 3. Multi-Head Self-Attention
        attn_out, attn_weights = self.self_attn(
            query=x, 
            key=x, 
            value=x, 
            key_padding_mask=full_mask
        ) # attn_out: [B, seq_len, hidden_dim], attn_weights: [B, seq_len, seq_len]
        
        # 4. Residual connection 1 + LayerNorm 1
        x = self.norm1(x + attn_out)
        
        # 5. Feed-Forward Network + Residual connection 2 + LayerNorm 2
        x = self.norm2(x + self.ffn(x))
        
        # 6. Classification from updated CLS token at index 0 (or mean pooling if no protein token)
        if full_protein_feature is not None:
            cls_out = x[:, 0, :] # [B, hidden_dim]
        else:
            cls_out = x.mean(dim=1) # [B, hidden_dim]
            
        logits = self.classifier(cls_out) # [B, 5]
        
        return logits, attn_weights


class SequenceMLPClassifier(nn.Module):
    """
    Pure sequence baseline model.
    Classifies protein cofactor specificity directly from its global ESM-2 sequence embedding (1280-dim),
    without candidate 3D pockets and without chemical ligand queries.
    """
    def __init__(self, in_features=1280, hidden_dim=256, num_classes=5, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, 1280] or [B, N, 1280] full protein sequence embedding.
        Returns:
            logits: [B, 5]
        """
        if x.ndim == 3:
            x = x.mean(dim=1)
        return self.net(x)


__all__ = [
    'COFACTORS',
    'TARGET_NAMES',
    'generate_ecfp4_fingerprints',
    'LigandCrossAttentionMIL',
    'SelfAttentionMIL',
    'SequenceMLPClassifier',
]


