import torch, torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MetaTransformer(nn.Module):
    def __init__(self, cfg, n_models: int):
        super().__init__()
        d_token = cfg["meta"]["d_token"]
        self.input_proj = nn.Linear(1, d_token)
        enc_layer = TransformerEncoderLayer(
            d_model=d_token, nhead=4,
            dim_feedforward=4*d_token,
            dropout=cfg["meta"]["attn_dropout"])
        self.encoder = TransformerEncoder(enc_layer,
                                          num_layers=cfg["meta"]["n_blocks"])
        self.head = nn.Sequential(
            nn.Linear(d_token, cfg["meta"]["hidden"]),
            nn.ReLU(),
            nn.Linear(cfg["meta"]["hidden"], 1))
        self.n_models = n_models

    def forward(self, model_preds, labels=None):
        x = model_preds.unsqueeze(-1)             # [B, K, 1]
        x = self.input_proj(x)                    # [B, K, d]
        x = x.permute(1, 0, 2)                    # [K, B, d]
        enc = self.encoder(x).mean(dim=0)         # [B, d]
        logits = self.head(enc).squeeze(-1)       # [B]
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels)
        return {"logits": logits, "loss": loss}
