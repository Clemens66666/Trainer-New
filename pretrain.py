from pathlib import Path
from rtdl import FTTransformer
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import types
from rtdl.modules import FTTransformer as _FTT
import torch

from utils.dataset import SSLWindowDataset 
# ───────────────────────────────────────────────────────────
# Konfiguration
CFG = dict(
    seq_len      = 24,
    batch        = 128,
    epochs       = 20,
    n_blocks     = 4,
    lr           = 1e-3,
    weight_decay = 1e-2,
)

CSV  = Path("./data/longtrend.csv")
COLS = ["Open", "high", "low", "Close", "Volume"]

# Dataset & DataLoader (kein custom collate_fn, es sei denn Du brauchst target)
ds     = SSLWindowDataset(CSV, COLS, seq_len=CFG["seq_len"])
loader = DataLoader(ds,
                    batch_size=CFG["batch"],
                    shuffle=True,
                    num_workers=0,
                    collate_fn=None)  # Default Collate

import types, torch, torch.nn as nn
from rtdl import FTTransformer
# … weitere imports (Path, DataLoader, SSLWindowDataset, …) …

# ── Modell ohne Zusatz‑Args bauen ───────────────────────────
ft = FTTransformer.make_default(
    n_num_features=len(COLS),
    cat_cardinalities=(),
    d_out=len(COLS) * CFG["seq_len"],   # wird gleich ignoriert
    n_blocks=CFG["n_blocks"],
)

import torch.nn as nn, types

# ➜ Class‑Token deaktivieren
ft.cls_token = nn.Identity()

# ➜ eigenes Forward – benutzt feature_tokenizer, Transformer,
#   **keine** cls‑Token‑Logik und kein to_out‑Layer.
def _forward_no_cls(self, x_num, x_cat=None):
    # 1. Numerische Tokens (kategorial = None)
    x = self.feature_tokenizer(x_num, x_cat)   # evtl. >3 D
    # 2. Auf 3 D reduzieren  (B, n_tokens, d)
    if x.ndim > 3:
        b, d = x.size(0), x.size(-1)
        x    = x.view(b, -1, d)
    elif x.ndim == 2:          # (B, d) → (B, 1, d)
        x = x.unsqueeze(1)
    # 3. Transformer‑Blöcke
    x = self.transformer(x)    # bleibt 3 D
    # 4. Rückgabe als flacher Vektor pro Sample
    return x.reshape(x.size(0), -1)   # (B, n_tokens·d)

# Monkey‑Patch binden
ft.forward = types.MethodType(_forward_no_cls, ft)

# 3) Optimizer & Loss
opt  = optim.AdamW(ft.parameters(), lr=CFG["lr"])
crit = nn.MSELoss()

ft.train()

# ───────────────────────────────────────────────────────────
for epoch in range(CFG["epochs"]):
    total_loss = 0.0

    for batch in loader:
        # 1) Batch entpacken
        if isinstance(batch, dict):
            x = batch["x_num"]
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        # 2) In Tensor & Float umwandeln
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.float()

        # 3) Alle überschüssigen Singleton‑Dims entfernen, sodass x.dim() == 3
        #    (Batch, Seq_len, Num_features)
        for dim in range(x.dim() - 1, 0, -1):
            if x.size(dim) == 1:
                x = x.squeeze(dim)
        # Falls wir zu wenige Dims haben (eher selten), eine am Token‑Level nachschieben
        if x.dim() < 3:
            x = x.unsqueeze(1)

        # 4) Rekonstruktions‑Target vorbereiten
        tgt = x.view(x.size(0), -1)

        # 5) Training‑Step
        opt.zero_grad()
        out  = ft(x, None)        # x_cat=None, da keine Kategorialen
        loss = crit(out, tgt)
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(loader.dataset)
    print(f"E{epoch+1:02d}  SSL-MSE {avg_loss:.5f}")

# Am Ende: Weights speichern
torch.save(ft.state_dict(), Path("ft_ssl.pt"))
print("🟢 Self‑Supervised Weights gespeichert → ft_ssl.pt")
