import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Canonical target cofactors and their SMILES
COFACTORS = {
    'acetyl-CoA': 'CC(C)(COP(=O)(O)OP(=O)(O)OCC1C(C(C(O1)N2C=NC3=C2N=CN=C3N)O)OP(=O)(O)O)C(C(=O)NCCC(=O)NCCSC(=O)C)O',
    'ATP': 'C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N',
    'B12': 'CC1=CC2=C(C=C1C)N(C=N2)C3C(C(C(O3)CO)OP(=O)([O-])OC(C)CNC(=O)CCC4(C(C5C6(C(C(C(=N6)C(=C7C(C(C(=N7)C=C8C(C(C(=N8)C(=C4[N-]5)C)CCC(=O)N)(C)C)CCC(=O)N)(C)CC(=O)N)C)CCC(=O)N)(C)CC(=O)N)C)CC(=O)N)C)O.[Co+3]',
    'FAD': 'CC1=CC2=C(C=C1C)N(C3=NC(=O)NC(=O)C3=N2)CC(C(C(COP(=O)(O)OP(=O)(O)OCC4C(C(C(O4)N5C=NC6=C5N=CN=C6N)O)O)O)O)O',
    'NAD': 'C1=CC(=C[N+](=C1)C2C(C(C(O2)COP(=O)(O)OP(=O)(O)OCC3C(C(C(O3)N4C=NC5=C4N=CN=C5N)O)O)O)O)C(=O)N'
}

TARGET_NAMES = ['acetyl-CoA', 'ATP', 'B12', 'FAD', 'NAD']

def generate_ecfp4_fingerprints(radius=2, n_bits=1024):
    """Generates 1024-bit Morgan ECFP4 fingerprints for all target cofactors."""
    fps = []
    for name in TARGET_NAMES:
        smi = COFACTORS[name]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Fallback for complex organometallics
            mol = Chem.MolFromSmiles(smi, sanitize=False)
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
    4. Canonical Transformer FFN with GELU & Residual LayerNorm.
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
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.protein_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.ligand_proj = nn.Sequential(
            nn.Linear(ecfp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 5. Final Scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
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
