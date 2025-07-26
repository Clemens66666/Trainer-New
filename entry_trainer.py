# entry_trainer.py  – Entry‑Setup (Hoch / Tief)
from pathlib import Path
from hybrid_longtrend_trainer import HybridLongTrendTrainer

class EntryTrainer(HybridLongTrendTrainer):
    CSV_FILE    = Path("data/longtrend_entry.csv")
    TARGET_COL  = "label_entry"
    RUN_NAME    = "entry"               # Ordner: models/entry/ …

    # ────────── nur Load‑Data überschreiben ──────────
    def load_data(self):
        num_cols = [
            "Open","High","Low","Close","Volume",
            "sma_10","ema_20","rsi_14","hour_sin","hour_cos"
        ]
        seq_len = 24                               # passt zu 1‑Min‑Bars
        ds = self._dataset_cls(                    # nutzt dieselbe Klasse
            csv_path       = self.CSV_FILE,
            numerical_cols = num_cols,
            seq_len        = seq_len,
            target_col     = self.TARGET_COL
        )
        # Train/Val‑Split identisch
        tr_idx, va_idx = self._split_fn(ds.X_seq, ds.y_seq, seq_len)
        self.X_train, self.y_train = ds.X_seq[tr_idx], ds.y_seq[tr_idx]
        self.X_val,   self.y_val   = ds.X_seq[va_idx], ds.y_seq[va_idx]
        return self.X_train, self.y_train
