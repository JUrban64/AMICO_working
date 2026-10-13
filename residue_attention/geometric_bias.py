import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Pairwise3DDistanceBias(nn.Module):
    """
    3D Geometric Distance Attention Bias for Residues.
    ==================================================
    Computes a spatial attention bias matrix based on 3D Euclidean distances between
    C-alpha residue coordinates. In structural biology, residues within 4-10 Angstroms
    frequently form catalytic dyads, hydrogen-bonding networks, or binding surfaces.
    
    This module supports:
    1. 'linear_decay': -gamma_h * D_ij (with learnable positive decay slope gamma_h per head).
    2. 'rbf_gaussian': Multi-kernel RBF expansion across distance bins (e.g. 0-20 A) projected to heads.
    3. 'exponential': -exp(D_ij / sigma_h) soft spatial decay.
    
    Additionally, it handles the 'Token 0' (global protein context):
    Since Token 0 does not possess a single 3D coordinate, the interaction between Token 0
    and residues is parameterized by learnable context bias parameters.
    """
    def __init__(self, num_heads=4, mode='rbf_gaussian', num_rbf=16, max_distance=25.0):
        super().__init__()
        self.num_heads = num_heads
        self.mode = mode
        self.max_distance = max_distance
        
        # Learnable bias for Token 0 (Global Protein Context) <-> Residues
        self.cls_to_res_bias = nn.Parameter(torch.zeros(num_heads))
        self.res_to_cls_bias = nn.Parameter(torch.zeros(num_heads))
        self.cls_to_cls_bias = nn.Parameter(torch.zeros(num_heads))
        
        if mode == 'rbf_gaussian':
            self.num_rbf = num_rbf
            # Centers evenly spaced from 0 to max_distance
            centers = torch.linspace(0.0, max_distance, num_rbf)
            # Width based on spacing
            spacing = max_distance / (num_rbf - 1)
            sigma = spacing / 2.0
            self.register_buffer('centers', centers)
            self.register_buffer('inv_two_sigma_sq', torch.tensor(1.0 / (2.0 * sigma ** 2)))
            # Linear projection from RBF basis to num_heads attention bias logits
            self.rbf_proj = nn.Linear(num_rbf, num_heads, bias=False)
            # Initialize with negative slope: closer residues get higher attention logits
            with torch.no_grad():
                weights = torch.linspace(1.0, -2.0, num_rbf).unsqueeze(0).repeat(num_heads, 1)
                self.rbf_proj.weight.copy_(weights)
                
        elif mode == 'linear_decay':
            # Learnable gamma > 0 per head (initialized to ~0.05 per Angstrom)
            self.log_gamma = nn.Parameter(torch.full((num_heads,), math.log(0.05)))
            
        elif mode == 'exponential':
            self.log_scale = nn.Parameter(torch.full((num_heads,), math.log(8.0)))
        else:
            raise ValueError(f"Unknown 3D bias mode: {mode}. Choose 'rbf_gaussian', 'linear_decay', or 'exponential'.")

    def forward(self, coords, has_cls_token=True):
        """
        Args:
            coords: [B, M, 3] tensor of C-alpha coordinates in Angstroms.
            has_cls_token: bool, whether Token 0 is the global protein context token.
            
        Returns:
            bias: [B, num_heads, S, S] tensor of attention logits biases, where:
                  S = M + 1 if has_cls_token else M.
        """
        B, M, _ = coords.size()
        
        # 1. Pairwise Euclidean Distance Matrix [B, M, M]
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, M, M, 3]
        dist = torch.norm(diff, p=2, dim=-1)  # [B, M, M]
        
        # 2. Compute residue-residue bias [B, num_heads, M, M]
        if self.mode == 'rbf_gaussian':
            # dist: [B, M, M, 1], centers: [1, 1, 1, num_rbf]
            dist_exp = dist.unsqueeze(-1)
            rbf = torch.exp(-((dist_exp - self.centers) ** 2) * self.inv_two_sigma_sq)  # [B, M, M, num_rbf]
            res_bias = self.rbf_proj(rbf)  # [B, M, M, num_heads]
            res_bias = res_bias.permute(0, 3, 1, 2)  # [B, num_heads, M, M]
            
        elif self.mode == 'linear_decay':
            gamma = torch.exp(self.log_gamma).view(1, self.num_heads, 1, 1)  # [1, H, 1, 1]
            dist_exp = dist.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [B, H, M, M]
            res_bias = -gamma * dist_exp
            
        elif self.mode == 'exponential':
            scale = torch.exp(self.log_scale).view(1, self.num_heads, 1, 1)
            dist_exp = dist.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            res_bias = -torch.exp(dist_exp / scale)

        if not has_cls_token:
            return res_bias
            
        # 3. Assemble full bias matrix including Token 0 (CLS) -> [B, num_heads, M + 1, M + 1]
        full_bias = torch.zeros(B, self.num_heads, M + 1, M + 1, dtype=coords.dtype, device=coords.device)
        
        # CLS <-> CLS
        full_bias[:, :, 0, 0] = self.cls_to_cls_bias.view(1, self.num_heads)
        
        # CLS -> Residues (Row 0, Cols 1..M)
        full_bias[:, :, 0, 1:] = self.cls_to_res_bias.view(1, self.num_heads, 1).expand(B, -1, M)
        
        # Residues -> CLS (Rows 1..M, Col 0)
        full_bias[:, :, 1:, 0] = self.res_to_cls_bias.view(1, self.num_heads, 1).expand(B, -1, M)
        
        # Residues <-> Residues (Rows 1..M, Cols 1..M)
        full_bias[:, :, 1:, 1:] = res_bias
        
        return full_bias
