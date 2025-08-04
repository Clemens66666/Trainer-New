# exit_trainer.py – Exit‑Risk (Draw‑Down)
from pathlib import Path
from hybrid_longtrend_trainer import HybridLongTrendTrainer

class ExitTrainer(HybridLongTrendTrainer):
    CSV_FILE    = Path("data/longtrend_exit.csv")
    TARGET_COL  = "label_exit"
    RUN_NAME    = "exit"                # Ordner: models/exit/ …

    def load_data(self):
        num_cols = [
            "Open","High","Low","Close","Volume",
            "sma_10","ema_20","rsi_14","hour_sin","hour_cos"
        ]
        seq_len = 12                               # 12‑Stunden‑Bars
        ds = self._dataset_cls(
            csv_path       = self.CSV_FILE,
            numerical_cols = num_cols,
            seq_len        = seq_len,
            target_col     = self.TARGET_COL
        )
        tr_idx, va_idx = self._split_fn(ds.X_seq, ds.y_seq, seq_len)
        self.X_train, self.y_train = ds.X_seq[tr_idx], ds.y_seq[tr_idx]
        self.X_val,   self.y_val   = ds.X_seq[va_idx], ds.y_seq[va_idx]
        return self.X_train, self.y_train
