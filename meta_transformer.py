import torch
import torch.nn as nn
from torch.nn import functional as F

class MetaTransformer(nn.Module):
    """
    Kleine Transformer-Architektur für das Meta-Ensembling.
    Robust gegen fehlende 'meta'-Einträge in der Config:
    - d_token   : Dimension der Token-Einbettung (Default = 64)
    - n_blocks  : Anzahl Transformer-Blöcke      (Default = 2)
    - d_ff      : Größe des Feed-Forward-Layers  (Default = 256)
    - dropout   : Dropout-Rate                  (Default = 0.10)
    """

    def __init__(self, cfg: dict, n_models: int):
        super().__init__()

        # ---- Konfig auslesen mit Fallbacks ----
        meta_cfg  = cfg.get("meta", {})
        d_token   = meta_cfg.get("d_token",   64)
        n_blocks  = meta_cfg.get("n_blocks",   2)
        d_ff      = meta_cfg.get("d_ff",     256)
        dropout   = meta_cfg.get("dropout", 0.10)

        self.embed = nn.Linear(n_models, d_token)

        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.TransformerEncoderLayer(
                d_model   = d_token,
                nhead     = 4,
                dim_feedforward = d_ff,
                dropout   = dropout,
                batch_first = True,
            ))
        self.transformer = nn.Sequential(*blocks)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x, labels=None):
        """
        x : Tensor [B, n_models]  – gestapelte Basis-Vorhersagen
        """
        # [B, 1, n_models]  → [B, 1, d_token]
        tok = self.embed(x.unsqueeze(1))
        # [B, 1, d_token]   → [B, 1, d_token]
        out = self.transformer(tok)
        # [B, 1, 1]         → [B]
        logits = self.head(out).squeeze(-1).squeeze(-1)
        if labels is None:
            return {"logits": logits}
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        return {"loss": loss, "logits": logits}
