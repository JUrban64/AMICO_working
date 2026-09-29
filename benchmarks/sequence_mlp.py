import torch
import torch.nn as nn

class SequenceMLPClassifier(nn.Module):
    """
    Čistě sekvenční baseline model pro AMICO.
    Klasifikuje specificitu kofaktoru proteinu přímo z jeho globálního
    ESM-2 sequence embeddingu (1280-dim), bez použití 3D kapes a bez chemických ligandových queries.
    """
    def __init__(self, in_features=1280, hidden_dim=256, num_classes=5, dropout=0.3):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

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
            x: [B, 1280] nebo [B, N, 1280] (pokud je předán bag, zprůměruje se na [B, 1280]).
        Returns:
            logits: [B, num_classes] (Logity pro 5 kofaktorových tříd).
        """
        if x.ndim == 3:
            x = x.mean(dim=1)
        return self.net(x)

__all__ = ['SequenceMLPClassifier']
