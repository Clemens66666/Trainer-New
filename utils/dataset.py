# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path
from utils.features import enrich
from utils.collate   import make_sequences

class LongTrendDataset:
    """Supervised Zeitfenster → Label 0/1"""
    def __init__(self,
                 csv_path: Path | str,
                 numerical_cols: list[str],
                 seq_len: int = 24):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        # 1) CSV laden + Feature-Enrichment
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = enrich(df)

        # fehlende Labels = 0 (Negativ)
        if "label" in df.columns:
            df["label"].fillna(0.0, inplace=True)
        else:
            raise KeyError("Spalte 'label' fehlt im Eingabe-CSV!")

        X_num = df[numerical_cols].to_numpy(dtype=np.float32)
        y     = df["label"].to_numpy(dtype=np.float32)

        self.X_seq = make_sequences(X_num,  seq_len)
        self.y_seq = y[seq_len - 1:]

        assert len(self.X_seq) == len(self.y_seq)

    def __len__(self):  return len(self.X_seq)
    def __getitem__(self, idx):
        return {"x_num": self.X_seq[idx], "label": self.y_seq[idx]}


# ──────────────────────────────────────────────────────────────
class SSLWindowDataset:
    """
    Self-Supervised Dataset:
      Input = Fenster (seq_len, F)
      Target = denselben Vektor rekonstruieren
    """
    def __init__(self, csv_path: Path | str,
                 numerical_cols: list[str],
                 seq_len: int = 24):
        df = enrich(pd.read_csv(csv_path, parse_dates=["timestamp"]))
        X = df[numerical_cols].to_numpy(dtype=np.float32)
        self.X_seq = make_sequences(X, seq_len)

    def __len__(self): return len(self.X_seq)
    def __getitem__(self, idx):
        x = self.X_seq[idx]
        return {"x_num": x, "target": x}   # rekonstruiere Input
