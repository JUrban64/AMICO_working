import torch
import torch.nn as nn
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


class LigandCrossAttentionMIL(nn.Module):
    """
    Ligand-Protein Cross-Attention Multi-Instance Learning Model.
    
    1. Keys & Values: Global protein context (token 0) + P2Rank 3D pockets (tokens 1..N).
    2. Queries: 5 ECFP4 chemical cofactor fingerprints.
    3. Multi-Head Cross-Attention: Chemically guided attention between ligands and pockets.
    4. Transformer FFN + Residual LayerNorms.
    5. Linear Scorer: Output logits for each cofactor class.
    """
    def __init__(self, feature_dim=1280, ecfp_dim=2048, hidden_dim=256, num_heads=4, num_classes=5, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Shared latent space projections
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
        
        # ECFP4 cofactor fingerprints
        cofactor_fps = generate_ecfp4_fingerprints(n_bits=ecfp_dim)
        self.register_buffer('cofactor_fps', cofactor_fps)
        
        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # FFN block
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, pocket_features, padding_mask, full_protein_feature=None):
        B, N, _ = pocket_features.size()
        
        # Keys & Values from pockets
        k = self.pocket_proj(pocket_features)
        v = k
        
        # Prepend whole protein embedding as token 0
        if full_protein_feature is not None:
            prot_tok = self.protein_proj(full_protein_feature).unsqueeze(1)
            k = torch.cat([prot_tok, k], dim=1)
            v = torch.cat([prot_tok, v], dim=1)
            prot_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            mask = torch.cat([prot_mask, padding_mask], dim=1)
        else:
            mask = padding_mask
            
        # Queries from cofactors
        q_ligands = self.ligand_proj(self.cofactor_fps).unsqueeze(0).expand(B, -1, -1)
        
        # Cross-Attention
        attn_out, attn_weights = self.cross_attn(
            query=q_ligands, 
            key=k, 
            value=v, 
            key_padding_mask=mask
        )
        
        # Residual + FFN
        out = self.norm1(attn_out + q_ligands)
        out = self.norm2(out + self.ffn(out))
        
        logits = self.scorer(out).squeeze(-1)
        return logits, attn_weights


__all__ = [
    'COFACTORS',
    'TARGET_NAMES',
    'generate_ecfp4_fingerprints',
    'LigandCrossAttentionMIL',
]
