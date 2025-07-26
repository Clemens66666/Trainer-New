import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# ───────────────────────── numerische Batches ──────────────────────────
def numeric_collate(batch):
    """
    Erwartet Items mit Keys:
        • 'x_num'   – Tensor (seq_len, F)      oder ndarray
        • 'labels'  – Skalar‑Float
    Gibt Batch‑Dict zurück, das der Trainer direkt nutzen kann.
    """
    # Tensors sicherstellen
    x_list = [torch.as_tensor(b["x_num"]) for b in batch]  # (L, F)
    x_pad  = pad_sequence(x_list, batch_first=True)        # (B, L_max, F)

    y = torch.as_tensor(
            [b.get("labels", b["label"]) for b in batch],
            dtype=torch.float32
        )
    return {"x_num": x_pad.float(), "labels": y}


# ───────────────────────── Stack‑Pred‑Meta ─────────────────────────────
def meta_collate(batch):
    preds  = torch.tensor([x["preds"]  for x in batch], dtype=torch.float32)
    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.float32)
    return {"model_preds": preds, "labels": labels}


# ───────────────────────── Rolling‑Window Helper ───────────────────────
SEQ_LEN = 24          # Default‑Länge: 24 Kerzen (1 Tag bei H‑Daten)

def make_sequences(arr: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    arr  shape (N, F)  →  (N‑seq_len+1, seq_len, F)
    baut überlappende Sliding‑Windows; extrem schnell via Stride‑Tricks.
    """
    if seq_len <= 1:
        return arr[:, None, :]
    assert arr.ndim == 2, "erwarte 2‑D‑Array (N, F)"
    # neues Shape + Strides so anpassen, dass jedes Fenster nur View‑Speicher ist
    from numpy.lib.stride_tricks import sliding_window_view
    seq = sliding_window_view(arr, window_shape=(seq_len, arr.shape[1]))
    return seq.reshape(-1, seq_len, arr.shape[1])
