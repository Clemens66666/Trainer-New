# trainers/hybrid_longtrend_trainer.py
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

import gc, optuna, psutil, math
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost  as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
# ganz oben in hybrid_longtrend_trainer.py ➜ Import‑Block ergänzen
from transformers.trainer_utils import IntervalStrategy

from rtdl import FTTransformer
from peft import LoraConfig, get_peft_model

from utils.dataset   import LongTrendDataset
from utils.collate   import numeric_collate, meta_collate
from utils.features  import enrich
from .meta_transformer import MetaTransformer
from .base import BaseTrainer
# ganz oben einmalig ergänzen (falls noch nicht vorhanden):
from transformers import EarlyStoppingCallback, IntervalStrategy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ───────────────────────── helper‑datasets ─────────────────────────
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):          return len(self.y)
    def __getitem__(self, idx): return {"x_num": self.X[idx], "label": self.y[idx]}

# ───── Focal‑Loss (binary) ───────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.bce   = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward(self, logits, targets):
        bce  = self.bce(logits, targets)
        prob = torch.sigmoid(logits)
        pt   = torch.where(targets == 1, prob, 1 - prob)        # p_t
        focal = (1 - pt) ** self.gamma * bce
        return focal.mean()

class FTWrapped(nn.Module):
    """
    Binary‑Wrapper
      • BCEWithLogits + optionale Class‑Imbalance‑Gewichte
      • Label‑Smoothing  (eps)
      • Focal‑Loss‑Faktor (gamma)  – wenn gamma > 0
    """
    def __init__(
        self,
        ft_base: nn.Module,
        pos_weight: torch.Tensor | None = None,
        label_smooth_eps: float = 0.0,
        focal_gamma: float | None = None,          # ← NEU
    ):
        super().__init__()
        self.ft   = ft_base
        self.eps  = label_smooth_eps
        self.gamma= focal_gamma
        self.bce  = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, x_num, labels: torch.Tensor | None = None):
        logits = self.ft(x_num, None).squeeze(-1)        # [B]
        loss   = None

        if labels is not None:
            # ── Label‑Smoothing ───────────────────────────────────────────
            if self.eps > 0.0:
                labels = labels * (1.0 - self.eps) + 0.5 * self.eps

            bce_loss = self.bce(logits, labels)

            # ── Focal‑Modulation (optional) ───────────────────────────────
            if self.gamma and self.gamma > 0:
                p_t = torch.sigmoid(logits).detach()
                focal_factor = (1.0 - p_t).pow(self.gamma)
                bce_loss = focal_factor * bce_loss

            loss = bce_loss.mean()

        return {"logits": logits, "loss": loss}


class SimpleCNN(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        # Eingabe­form:  x  = [B, n_feat, seq_len]
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, 32, kernel_size=3, padding=1),  # 32 Filter
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                          # -> [B,32,1]
            nn.Flatten(),                                     # -> [B,32]
            nn.Linear(32, 1)                                  # -> [B,1]
        )

    def forward(self, x):                                     # x: [B, n_feat, L]
        return self.net(x).squeeze(-1)                        # -> [B]


class MetaDataset(Dataset):
    def __init__(self, preds, y): self.preds, self.y = preds, y
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return {"preds":self.preds[i],"labels":self.y[i]}

# ──────────────────────── Trainer‑Klasse ───────────────────────────
# ───────────────────────── HybridLongTrendTrainer ──────────────────────────
class HybridLongTrendTrainer(BaseTrainer):
# ═══════════════════════════  Daten laden & Val‑Split  ═════════════════════
# ═════════════════════════════════════════════════════════════════════
#  Daten laden  (ohne Trial‑Abhängigkeit)
# ═════════════════════════════════════════════════════════════════════
# ───────────────────────────── Daten laden ──────────────────────────────
    def load_data(self):
        num_cols = [
            "Open", "high", "low", "Close", "Volume",
            "sma_10", "ema_20", "rsi_14", "hour_sin", "hour_cos"
        ]
        # feste Länge aus der YAML
        seq_len = self.cfg["training"].get("seq_len", 24)

        ds = LongTrendDataset(
            csv_path       = f"{self.cfg['data']['raw_dir']}/{self.cfg['data']['longtrend_file']}",
            numerical_cols = num_cols,
            seq_len        = seq_len
        )

        # ── TimeSeriesSplit mit größerem Gap + 5 % Positives ────────────────
        def last_fold_with_enough_pos(X, y, n_splits=5, min_frac=0.05):
            tss   = TimeSeriesSplit(n_splits=n_splits, gap=2*seq_len)
            folds = list(tss.split(X))
            for tr, va in reversed(folds):
                if y[va].mean() >= min_frac:
                    return tr, va
            return folds[-1]

        tr_idx, va_idx = last_fold_with_enough_pos(
            ds.X_seq, ds.y_seq, n_splits=5, min_frac=0.05
        )
        self.X_train, self.y_train = ds.X_seq[tr_idx], ds.y_seq[tr_idx]
        self.X_val,   self.y_val   = ds.X_seq[va_idx], ds.y_seq[va_idx]

        return self.X_train, self.y_train

# ═══════════════════════════ Utility 3‑D → 2‑D ════════════════════════════
    @staticmethod
    def _flat(X3d: np.ndarray) -> np.ndarray:
        return X3d.reshape(len(X3d), -1)

    def build_features(self, X):  return X               # Identity‑Hook
    def train_final(self,*_,**__): return self.meta      # liefert Meta‑Modell

    # ───────────────────── Mini‑Batch‑LogLoss (smoothing aware) ─────────────────────
    @staticmethod
    def _batch_log_loss(model, X, y, batch_size: int = 1024) -> float:
        """Val‑Loss identisch zur Trainings‑Loss, auch wenn FTWrapped geändert wurde."""
        model.eval()
        loss_list = []

        # ⇣ passende Loss‑Funktion finden oder rekonstruieren
        if hasattr(model, "bce"):
            base_loss = model.bce          # neue Namensvariante
            eps       = getattr(model, "eps", 0.0)
        elif hasattr(model, "crit"):
            base_loss = model.crit         # alte Variante
            eps       = getattr(model, "eps", 0.0)
        else:
            # Fallback – einfaches BCE (ohne focal)
            pos_weight = torch.tensor([(len(y) - y.sum()) / (y.sum() + 1e-6)])
            base_loss  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
            eps        = 0.0

        with torch.no_grad():
            for i in range(0, len(y), batch_size):
                xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
                lb = torch.tensor(y[i:i+batch_size], dtype=torch.float32)

                if eps > 0:
                    lb = lb * (1.0 - eps) + 0.5 * eps

                logits = model(xb)["logits"]
                loss   = base_loss(logits, lb).mean()
                loss_list.append(loss.cpu())

        return torch.stack(loss_list).mean().item()


# ════════════════════════ Basis‑Modelle ═══════════════════════════════════
    def _train_rf(self, X, y):
        Xf, rng, models = self._flat(X), np.random.default_rng(42), []
        for _ in range(3):
            idx = rng.choice(len(Xf), len(Xf), replace=True)
            m   = RandomForestClassifier(**self.cfg["model"]["rf_params"])
            m.fit(Xf[idx], y[idx]); models.append(m)
        return models

    def _train_lgb(self, X, y):
        Xf, rng, models = self._flat(X), np.random.default_rng(0), []
        for _ in range(3):
            idx = rng.choice(len(Xf), len(Xf), replace=True)
            d   = lgb.Dataset(Xf[idx], label=y[idx])
            m   = lgb.train(dict(objective="binary", learning_rate=0.05,
                                 num_leaves=64, metric="binary_logloss"),
                            d, 300)
            models.append(m)
        return models

    def _train_xgb(self, X, y):
        Xf, rng, models = self._flat(X), np.random.default_rng(1), []
        for _ in range(3):
            idx = rng.choice(len(Xf), len(Xf), replace=True)
            d   = xgb.DMatrix(Xf[idx], label=y[idx])
            m   = xgb.train(dict(objective="binary:logistic", eta=0.05,
                                 max_depth=6, eval_metric="logloss"),
                            d, 300)
            models.append(m)
        return models

    # -------- 1‑D‑CNN Basismodell --------
    def _train_cnn(self, X: np.ndarray, y: np.ndarray):
        """
        X: ndarray  [N, seq_len, n_feat]
        y: ndarray  [N]
        """
        mdl = SimpleCNN(n_feat=X.shape[2]).to(device)

        # →  Torch-Tensors auf den richtigen Device
        X_t = torch.tensor(X, dtype=torch.float32, device=device).permute(0, 2, 1)  # [N, n_feat, L]
        y_t = torch.tensor(y, dtype=torch.float32, device=device).view(-1)          # [N]  (‼️ SQUEEZE)

        crit = nn.BCEWithLogitsLoss()
        opt  = optim.Adam(mdl.parameters(), lr=1e-3)

        mdl.train()
        for _ in range(10):
            opt.zero_grad()
            logits = mdl(X_t)                 # [N]
            loss   = crit(logits, y_t)
            loss.backward()
            opt.step()

        #  für spätere CPU-Inference wieder zurück-schieben
        return mdl.cpu()
        # ───────────────────────────────
    # RAM‑schonende Inferenz für FT
    # ───────────────────────────────
    def _predict_ft_logits(self, X_num: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Berechnet Logits des FT‑Transformers stückweise, um Speicher zu sparen.

        Parameters
        ----------
        X_num : ndarray  [N, 240]  flach ⇒ 24 × 10 numerische Features
        batch_size : int  Größe der Mini‑Batches für die Inferenz
        """
        self.ft.eval()                          # kein Dropout
        preds = []

        with torch.no_grad():                   # Autograd ausschalten
            for i in range(0, len(X_num), batch_size):
                x_batch = torch.tensor(
                    X_num[i : i + batch_size],
                    dtype=torch.float32,
                    device=self.device         # self.device ist cpu | cuda
                )
                logits = self.ft(x_batch, None).squeeze(-1)  # [B]
                preds.append(logits.cpu().numpy())

        return np.concatenate(preds, axis=0)

# ═════════════════════ FT‑Transformer Helfer ══════════════════════════════
        # ══════════════════  FT‑Transformer Fabrik  ════════════════════════
    def _make_ft(self, n_features: int, n_blocks: int) -> FTTransformer:
        """
        Erstellt einen FT‑Transformer und setzt anschließend höhere Dropout‑
        Raten auf Attention‑ sowie FFN‑Ebene (0.2).  Das geht, weil
        rtdl.FTTransformer.make_default die Layer als Dict speichert.
        """
        ft = FTTransformer.make_default(       # ohne Dropout‑Args
            n_num_features    = n_features,
            cat_cardinalities = (),
            d_out             = 1,
            n_blocks          = n_blocks
        )

        # ──  Attention & FFN‑Dropout pro Block anpassen  ──────────────
        for blk in ft.transformer.blocks:               # type: ignore[attr-defined]
            blk['attention'].dropout.p = 0.2            # Attention‑Dropout
            blk['ffn'].dropout.p       = 0.2            # FFN‑Dropout

        return ft

    def _add_lora(self, ft, n_blocks, rank=4):
        last2 = range(n_blocks-2, n_blocks)
        target = [f"blocks.{i}.{p}"
                  for i in last2
                  for p in ["attention.W_q","attention.W_k","attention.W_v",
                            "attention.W_out","ffn.linear_first",
                            "ffn.linear_second"]]
        return get_peft_model(
            ft,
            LoraConfig(r=rank,
                       lora_alpha=8*rank,
                       lora_dropout=0.05,
                       target_modules=target)
        )

        # ───────────────────── FT + Optuna Objective ──────────────────────────────
    # ───────────────────────── FT‑Optuna‑Objective ─────────────────────────
    # ═════════════════════════════════════════════════════════════════════
#  FT‑Optuna‑Objective  (inkl. seq_len‑Search, Early‑Stopping, Focal Loss)
# ═════════════════════════════════════════════════════════════════════
    # ───────────────────────── Optuna‑Objective ─────────────────────────────
    def _ft_objective(self, trial, Xf, y):
        hp = {
            "n_blocks": trial.suggest_int("n_blocks", 2, 6),
            "lr"      : trial.suggest_float("lr", 1e-5, 3e-4, log=True),
        }

        pos_weight = torch.tensor([(len(y) - y.sum()) / (y.sum() + 1e-6)])
        ft   = self._make_ft(Xf.shape[1], hp["n_blocks"])

        # kleiner LoRA‑Rank
        last2  = range(hp["n_blocks"] - 2, hp["n_blocks"])
        target = [f"blocks.{i}.{p}"
                for i in last2
                for p in ["attention.W_q", "attention.W_k", "attention.W_v",
                            "attention.W_out", "ffn.linear_first",
                            "ffn.linear_second"]]
        ft = get_peft_model(ft, LoraConfig(r=4, lora_alpha=16,
                                       lora_dropout=0.05,
                                       target_modules=target))

        model = FTWrapped(ft,
                        pos_weight    = pos_weight,
                        label_smooth_eps = 0.10,
                        focal_gamma      = 2.0)

        split = int(0.8 * len(y))
        ds_train = NumpyDataset(Xf[:split],  y[:split])
        ds_val   = NumpyDataset(Xf[split:], y[split:])

        args = TrainingArguments(
            output_dir                  = "opt_ft_tmp",
            per_device_train_batch_size = 32,
            num_train_epochs            = 6,
            learning_rate               = hp["lr"],
            weight_decay                = 1e-2,
            fp16                        = True,
            logging_steps               = 50,
            report_to                   = [],      # keine externen Logger
        )
        # ► Early‑Stopping‑Settings  (neu: eval_strategy statt evaluation_strategy)
        args.metric_for_best_model  = "eval_loss"
        args.load_best_model_at_end = True
        args.eval_strategy          = IntervalStrategy.EPOCH  

        trainer = Trainer(
            model         = model,
            args          = args,
            train_dataset = ds_train,
            eval_dataset  = ds_val,
            data_collator = numeric_collate,
        )
        from transformers import EarlyStoppingCallback
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
        trainer.train()

        val_loss = self._batch_log_loss(model, Xf[split:], y[split:])

        del trainer, model, ft, ds_train, ds_val
        torch.cuda.empty_cache(); gc.collect()
        return val_loss

    # Aufräum‑Callback nach jedem Trial
    def _cleanup_callback(self, study, trial):
        import shutil, glob
        for ckpt in glob.glob("opt_ft_tmp/checkpoint-*"):
            shutil.rmtree(ckpt, ignore_errors=True)
        torch.cuda.empty_cache(); gc.collect()


    # ───────────────────────── FT‑Training (Optuna + Final‑Fit) ──────────────────
    def _train_ft(self, X, y):  
        Xf = self._flat(X)                              # (N, L*F)

        # ---------- Optuna‑Suche -------------------------------------------------
        ft_study = optuna.create_study(direction="minimize")
        ft_study.optimize(
            lambda t: self._ft_objective(t, Xf, y),
            n_trials          = self.cfg["optuna"]["n_trials"],
            callbacks         = [self._cleanup_callback],
            show_progress_bar = False,
        )

        best = ft_study.best_params
        print("🟢  FT‑best:", best)

        # ---------- Final‑Fit mit besten Parametern -----------------------------
        pos_weight = torch.tensor([(len(y) - y.sum()) / (y.sum() + 1e-6)])

        ft = self._make_ft(Xf.shape[1], best["n_blocks"])

        last2  = range(best["n_blocks"] - 2, best["n_blocks"])
        target = [f"blocks.{i}.{p}"
                for i in last2
                for p in ["attention.W_q", "attention.W_k", "attention.W_v",
                            "attention.W_out", "ffn.linear_first",
                            "ffn.linear_second"]]
        ft = get_peft_model(ft, LoraConfig(r=4, lora_alpha=16,
                                        lora_dropout=0.05,
                                        target_modules=target))

        model = FTWrapped(ft,
                        pos_weight       = pos_weight,
                        label_smooth_eps = 0.10,
                        focal_gamma      = 2.0)

        # ── Optional: SSL‑Weights vor dem Fine‑Tuning laden ────────────────
        from pathlib import Path

        if getattr(self, "ssl_weights", None) and Path(self.ssl_weights).exists():
            print(f"▶  Lade SSL-Weights von {self.ssl_weights}")
            # model.ft ist dein PEFT‑Model, strict=False ignoriert fehlende Keys
            model.ft.load_state_dict(torch.load(self.ssl_weights), strict=False)

        final_ds   = NumpyDataset(Xf, y)

        final_args = TrainingArguments(
            output_dir                  = "ft_final",
            per_device_train_batch_size = 32,
            num_train_epochs            = 8,
            learning_rate               = best["lr"],
            weight_decay                = 1e-2,
            fp16                        = True,
            logging_steps               = 50,
            report_to                   = [],
        )

        Trainer(model=model, args=final_args,
                train_dataset=final_ds,
                data_collator=numeric_collate).train()

        torch.cuda.empty_cache(); gc.collect()
        return model

# ═══════════════  Meta‑Stacking / Optimieren  ═════════════════════════════
    def optimize(self, X, y):
        X_flat      = self._flat(X)

        # ① Basismodelle trainieren
        self.rf_list  = self._train_rf(X, y)
        self.ft       = self._train_ft(X, y)           # ← nutzt intern GPU über HF-Trainer
        self.lgb_list = self._train_lgb(X, y)
        self.xgb_list = self._train_xgb(X, y)
        self.cnn      = self._train_cnn(X, y)

        # ② Vorhersagen **für den Val-Fold** erzeugen
        X_val_flat = self._flat(self.X_val)

        def mean_preds(models, Xtab, kind):
            if kind == "rf":
                return np.mean([m.predict_proba(Xtab)[:, 1] for m in models], axis=0)
            if kind == "lgb":
                return np.mean([m.predict(Xtab) for m in models], axis=0)
            if kind == "xgb":
                return np.mean([m.predict(xgb.DMatrix(Xtab)) for m in models], axis=0)

        preds_val = np.stack([
            mean_preds(self.rf_list,  X_val_flat, "rf"),
            mean_preds(self.lgb_list, X_val_flat, "lgb"),
            mean_preds(self.xgb_list, X_val_flat, "xgb"),
            # ─ FT (gestückelt, spart RAM) ────────────────────────────────
            torch.sigmoid(
                torch.tensor(
                    self._predict_ft_logits(X_val_flat),
                    dtype=torch.float32
                )
            ).numpy(),
            # CNN – permutieren bleibt gleich
            torch.sigmoid(
                self.cnn(
                    torch.tensor(self.X_val, dtype=torch.float32, device=self.device).permute(0, 2, 1)
                )
            ).detach().cpu().numpy()
        ], axis=1).astype(np.float32)

        # ③ Meta‑Modell NUR auf Val‑Fold trainieren (reduziert Leakage / Overfitting)
        meta_ds   = MetaDataset(preds_val, self.y_val)
        meta_args = TrainingArguments(
            output_dir                  = "meta_runs",
            per_device_train_batch_size = 128,
            num_train_epochs            = 10,
            learning_rate               = 1e-3,
            fp16                        = torch.cuda.is_available(),
            logging_steps               = 50,
            report_to                   = []   # ← FIX: '=' eingefügt
        )


        self.meta = MetaTransformer(self.cfg, n_models=preds_val.shape[1]).to(device)
        Trainer(model=self.meta,
                args=meta_args,
                train_dataset=meta_ds,
                data_collator=meta_collate).train()

        class Dummy: best_params = {}
        return Dummy()

# ═════════════════════ Speichern ══════════════════════════════════════════
    def save_model(self, _):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.ft.ft.state_dict(), self.model_dir / "ft.pt")
        torch.save(self.ft.state_dict(),    self.model_dir / "ft_wrap.pt")
        torch.save(self.cnn.state_dict(),   self.model_dir / "cnn.pt")
        torch.save(self.meta.state_dict(),  self.model_dir / "meta.pt")

        import joblib
        joblib.dump(self.rf_list,  self.model_dir / "rf_list.pkl")
        joblib.dump(self.lgb_list, self.model_dir / "lgb_list.pkl")
        joblib.dump(self.xgb_list, self.model_dir / "xgb_list.pkl")
        print(f"✅  Modelle gespeichert in {self.model_dir}")