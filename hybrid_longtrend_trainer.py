# trainers/hybrid_longtrend_trainer.py
# ============================================================================
# HybridLongTrendTrainer  –  CPU-freundliches Ensemble (RF, LGB, XGB, FT, CNN)
# mit Optuna-Tuning, Focal-Loss, Regime-Features & Temperature-Scaling
# ============================================================================

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import gc, math, joblib, optuna, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from transformers import (Trainer, TrainingArguments, EarlyStoppingCallback,
                          IntervalStrategy)
from rtdl import FTTransformer
from peft import LoraConfig, get_peft_model

# ─── Projekt-Imports ────────────────────────────────────────────────────────
from utils.dataset   import LongTrendDataset
from utils.collate   import numeric_collate, meta_collate
from utils.features  import enrich
from .meta_transformer import MetaTransformer
from .base           import BaseTrainer
from lightgbm import early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation


# ========================================================================== #
# Helper-Datasets
# ========================================================================== #
class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y
    def __len__(self):          return len(self.y)
    def __getitem__(self, idx): return {"x_num": self.X[idx], "label": self.y[idx]}

class MetaDataset(Dataset):
    """
    Liefert Stacking-Input
      preds  : [n_models]
      regime : [n_regime] (optional)
      labels : Float 0/1
    """
    def __init__(self, preds, labels, regime=None):
        assert len(preds) == len(labels)
        self.preds  = preds.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.regime = regime.astype(np.float32) if regime is not None else None
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {"preds": self.preds[idx], "labels": self.labels[idx]}
        if self.regime is not None: item["regime"] = self.regime[idx]
        return item

# ========================================================================== #
# Regime-Features
# ========================================================================== #
from ta.volatility import AverageTrueRange, BollingerBands
def extract_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    bb  = BollingerBands(df["close"], window=20, window_dev=2)
    feat = pd.DataFrame(index=df.index)
    feat["atr"] = atr.average_true_range()
    feat["bbp"] = bb.bollinger_pband()
    return feat.ffill().fillna(0)

# ========================================================================== #
# Verluste & Kalibrierung
# ========================================================================== #
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma, self.bce = gamma, nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=pos_weight)
    def forward(self, logits, targets):
        bce  = self.bce(logits, targets)
        prob = torch.sigmoid(logits)
        pt   = torch.where(targets == 1, prob, 1 - prob)
        return ((1 - pt) ** self.gamma * bce).mean()

class TemperatureScaler(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
    def forward(self, x): return self.model(x) / self.temperature

@torch.no_grad()
def calibrate_temperature(model, val_loader):
    device = next(model.parameters()).device
    scaler = TemperatureScaler(model).to(device)
    opt = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    def _loss():
        opt.zero_grad()
        logits = torch.cat([scaler(x.to(device)) for x, _ in val_loader])
        labels = torch.cat([y.to(device) for _, y in val_loader])
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(), labels.float())
        loss.backward(); return loss
    opt.step(_loss)
    return scaler.temperature.item()
# ---- neu: kleiner Wrapper für FT-Logits (für Temperature-Scaling)
class _FTLogits(nn.Module):
    def __init__(self, ft_model: nn.Module):
        super().__init__()
        self.ft_model = ft_model
    def forward(self, x):
        out = self.ft_model(x)
        return out["logits"] if isinstance(out, dict) else out
# ========================================================================== #
# Einfache 1-D-CNN - konfigurierbar
# ========================================================================== #
class SimpleCNN(nn.Module):
    def __init__(self, n_feat: int, n_filters: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, n_filters, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_filters, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# ========================================================================== #
# Optuna-Utility
# ========================================================================== #
def run_optuna_and_save(objective_fn, n_trials: int, study_name: str, save_dir: Path):
    import optuna, joblib, time
    optuna.logging.set_verbosity(optuna.logging.INFO)

    def cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        print(f"[{time.strftime('%H:%M:%S')}] {study.study_name} "
              f"trial#{trial.number} value={trial.value} params={trial.params}", flush=True)

    study = optuna.create_study(direction="minimize", study_name=study_name)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True, callbacks=[cb])

    pkl = save_dir / f"{study_name}.pkl"
    joblib.dump(study, pkl)
    study.trials_dataframe().to_csv(save_dir / f"{study_name}_trials.csv", index=False)
    return study

class FTWrapped(nn.Module):
    """
    Wrapper für rtdl.FTTransformer:
      - gibt {"logits": logits, "loss": loss} zurück
      - unterstützt optional Focal-Loss und Label-Smoothing
    """
    def __init__(self, ft_base: nn.Module,
                 pos_weight: torch.Tensor | None = None,
                 label_smooth_eps: float = 0.0,
                 focal_gamma: float | None = None):
        super().__init__()
        self.ft = ft_base
        self.eps = label_smooth_eps
        self.gamma = focal_gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, x_num, labels: torch.Tensor | None = None):
        # rtdl.FTTransformer erwartet (x_num, x_cat=None)
        logits = self.ft(x_num, None).squeeze(-1)
        loss = None
        if labels is not None:
            if self.eps > 0.0:
                labels = labels * (1.0 - self.eps) + 0.5 * self.eps
            bce = self.bce(logits, labels)
            if self.gamma and self.gamma > 0:
                with torch.no_grad():
                    p = torch.sigmoid(logits)
                    pt = torch.where(labels == 1, p, 1 - p)
                bce = (1 - pt).pow(self.gamma) * bce
            loss = bce.mean()
        return {"logits": logits, "loss": loss}


# ========================================================================== #
# HybridLongTrendTrainer
# ========================================================================== #
class HybridLongTrendTrainer(BaseTrainer):

    # ------------------------------------------------------------------ init
    def __init__(self, cfg_path: str):
        super().__init__(cfg_path)
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stamp         = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir= Path(self.cfg.get("output_root", "models")) \
                        / f"{self.cfg.get('exp_name','hybrid')}_{stamp}"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- load_data
    def load_data(self):
        seq_len  = self.cfg["training"].get("seq_len", 24)
        num_cols = self.cfg["data"]["numerical_cols"]
        ds = LongTrendDataset(
            csv_path       = f"{self.cfg['data']['raw_dir']}/"
                             f"{self.cfg['data']['longtrend_file']}",
            numerical_cols = num_cols,
            seq_len        = seq_len
        )
        # TimeSeriesSplit – letzter Fold mit ≥ 5 % Positiv
        def pick_fold(X, y, splits=5, min_frac=0.05):
            tss = TimeSeriesSplit(splits, gap=2*seq_len)
            for tr, va in reversed(list(tss.split(X))):
                if y[va].mean() >= min_frac: return tr, va
            return list(tss.split(X))[-1]
        tr, va = pick_fold(ds.X_seq, ds.y_seq)
        self.X_train, self.y_train = ds.X_seq[tr], ds.y_seq[tr]
        self.X_val,   self.y_val   = ds.X_seq[va], ds.y_seq[va]
        return self.X_train, self.y_train

    # -------------------------------------------------- Basis-Model-Trainer
    def _train_rf(self, X, y):
        Xf = self._flat(X)
        def obj(t): return self._rf_objective(t, Xf, y)
        best = run_optuna_and_save(obj, self.cfg["optuna"]["n_trials"],
                                   "rf_study", self.model_dir).best_params
        models, rng = [], np.random.default_rng(42)
        for _ in range(3):
            idx = rng.choice(len(Xf), len(Xf), replace=True)
            m   = RandomForestClassifier(**best); m.fit(Xf[idx], y[idx])
            models.append(m)
        return models

    def _train_lgb(self, X, y):
        Xf = self._flat(X)
        best = run_optuna_and_save(
            lambda t: self._lgb_objective(t, Xf, y),
            self.cfg["optuna"]["n_trials"], "lgb_study", self.model_dir
        ).best_params
        d   = lgb.Dataset(Xf, label=y)
        return [lgb.train({**best,"objective":"binary","metric":"binary_logloss"},
                          d, 500)]

    def _train_xgb(self, X, y):
        Xf = self._flat(X)
        best = run_optuna_and_save(
            lambda t: self._xgb_objective(t, Xf, y),
            self.cfg["optuna"]["n_trials"], "xgb_study", self.model_dir
        ).best_params
        d = xgb.DMatrix(Xf, label=y)
        return [xgb.train({**best,"objective":"binary:logistic",
                           "eval_metric":"logloss"}, d, 500)]

    def _train_cnn(self, X, y):
        best = run_optuna_and_save(
            lambda t: self._cnn_objective(t),  # nutzt self.X_train intern
            self.cfg["optuna"]["n_trials"], "cnn_study", self.model_dir
        ).best_params
        mdl = SimpleCNN(X.shape[2], best["n_filters"]).to(self.device)
        opt = optim.Adam(mdl.parameters(), lr=best["lr"])
        crit= nn.BCEWithLogitsLoss()
        Xt  = torch.tensor(X, dtype=torch.float32,
                           device=self.device).permute(0,2,1)
        yt  = torch.tensor(y, dtype=torch.float32, device=self.device)
        mdl.train()
        for _ in range(10):
            opt.zero_grad(); loss = crit(mdl(Xt).squeeze(), yt)
            loss.backward(); opt.step()
        return mdl.cpu()

    # ---------------------------------------------------- FT-Transformer
    def _train_ft(self, X, y):
        Xf = self._flat(X)
        study = run_optuna_and_save(
            lambda t: self._ft_objective(t, Xf, y),
            self.cfg["optuna"]["n_trials"], "ft_study", self.model_dir)
        hp = study.best_params
        ft = FTTransformer.make_default(
            n_num_features=Xf.shape[1], cat_cardinalities=(), d_out=1,
            n_blocks=hp["n_blocks"])
        ft = get_peft_model(
            ft, LoraConfig(r=4, lora_alpha=16, lora_dropout=0.05,
                           target_modules=["ffn.linear_first"]))
        posw = torch.tensor([(len(y)-y.sum())/(y.sum()+1e-6)], device=self.device)
        model= FTWrapped(ft, posw, label_smooth_eps=0.1,
                         focal_gamma=self.cfg["training"]["focal_gamma"])
        ds   = NumpyDataset(Xf, y)
        Trainer(model, TrainingArguments(
            output_dir=f"{self.model_dir}/ft_final",
            per_device_train_batch_size=32, num_train_epochs=8,
            learning_rate=hp["lr"], no_cuda=False, fp16=False,
            save_strategy="no", eval_strategy="no",
            report_to=[]
        ), train_dataset=ds, data_collator=numeric_collate).train()
        return model.cpu()

    # ----------------------------------------------------- Meta-Training
    def _train_meta(self, preds_val, regime_val):
        obj = lambda t: self._meta_objective(
            t, preds_val, self.y_val, regime_val)
        best = run_optuna_and_save(
            obj, self.cfg["optuna"]["n_trials"],
            "meta_study", self.model_dir).best_params

        inp_dim = preds_val.shape[1] + (regime_val.shape[1] if
                                        regime_val is not None else 0)
        meta = MetaTransformer(input_dim=inp_dim,
                               d_token=best["d_token"],
                               dropout=best["dropout"]).to(self.device)
        optim= torch.optim.Adam(meta.parameters(), lr=best["lr"])
        crit = nn.BCEWithLogitsLoss()
        Xval = preds_val if regime_val is None else \
               np.hstack([preds_val, regime_val])
        ds   = TensorDataset(torch.tensor(Xval, dtype=torch.float32),
                             torch.tensor(self.y_val, dtype=torch.float32))
        dl   = DataLoader(ds, batch_size=256, shuffle=True)
        meta.train()
        for _ in range(10):
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss   = crit(meta(xb).squeeze(), yb)
                optim.zero_grad(); loss.backward(); optim.step()
        return meta.cpu()

    # ----------------------------------------------------------- optimize
    def optimize(self, X, y):

        print("\n=== RF: Optuna + Final-Training ===", flush=True)
        self.rf_list  = self._train_rf(X, y)
        print("=== RF: fertig ===", flush=True)

        print("\n=== LGB: Optuna startet ===", flush=True)
        self.lgb_list = self._train_lgb(X, y)
        print("=== LGB: fertig ===", flush=True)

        print("\n=== XGB: Optuna startet ===", flush=True)
        self.xgb_list = self._train_xgb(X, y)
        print("=== XGB: fertig ===", flush=True)

        print("\n=== CNN: Optuna startet ===", flush=True)
        self.cnn      = self._train_cnn(X, y)
        print("=== CNN: fertig ===", flush=True)

        print("\n=== FT: Optuna startet ===", flush=True)
        self.ft       = self._train_ft(X, y)
        print("=== FT: fertig ===", flush=True)


        # Vorhersagen für Val-Fold
        X_val_flat = self._flat(self.X_val)
        preds_val  = np.vstack([
            np.mean([m.predict_proba(X_val_flat)[:,1] for m in self.rf_list],0),
            np.mean([m.predict(X_val_flat)          for m in self.lgb_list],0),
            np.mean([m.predict(xgb.DMatrix(X_val_flat)) for m in self.xgb_list],0),
            torch.sigmoid(
                self.ft(torch.tensor(X_val_flat, dtype=torch.float32))["logits"]
            ).squeeze().numpy(),

            torch.sigmoid(self.cnn(torch.tensor(self.X_val,
                          dtype=torch.float32).permute(0,2,1))).numpy()
        ]).T

        # Regime-Features (falls aktiviert)
        regime_val = None
        if self.cfg["meta"]["use_regime"]:
            cols = self.cfg["data"]["numerical_cols"]
            try:
                i_high  = cols.index("high")
                i_low   = cols.index("low")
                i_close = cols.index("Close")
                high_seq  = self.X_val[:, :, i_high]   # [N, L]
                low_seq   = self.X_val[:,  :, i_low]   # [N, L]
                close_seq = self.X_val[:, :, i_close]  # [N, L]

                # ATR-Proxy: mean(true range) über Sequenz
                tr = np.maximum(high_seq, np.roll(close_seq, 1, axis=1)) \
                     - np.minimum(low_seq,  np.roll(close_seq, 1, axis=1))
                tr[:, 0] = (high_seq[:, 0] - low_seq[:, 0])
                atr = tr.mean(axis=1)  # [N]

                # Bollinger-%B (letzte 20 Bars, falls vorhanden)
                w = min(20, close_seq.shape[1])
                c_last = close_seq[:, -w:]     # [N, w]
                ma  = c_last.mean(axis=1)
                sd  = c_last.std(axis=1) + 1e-9
                upper = ma + 2*sd
                lower = ma - 2*sd
                bbp = (close_seq[:, -1] - lower) / (upper - lower + 1e-9)

                regime_val = np.stack([atr, bbp], axis=1).astype(np.float32)  # [N,2]
            except ValueError:
                print("⚠️  Regime-Features: high/low/Close nicht in numerical_cols gefunden – skip.")
                regime_val = None

        # Meta-Stack trainieren+
        self.meta = self._train_meta(preds_val, regime_val)

        # Temperature-Scaling (FT)
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val_flat, dtype=torch.float32),
                          torch.tensor(self.y_val,   dtype=torch.float32)),
            batch_size=512, shuffle=False
        )
        T = calibrate_temperature(_FTLogits(self.ft), val_loader)
        torch.save({"temperature": T}, self.model_dir / "temp_scaler.pt")

    # ------------------------------------------------------------- save
    def save_model(self, *_):
        self.model_dir.mkdir(exist_ok=True)
        torch.save(self.ft.state_dict(),   self.model_dir / "ft.pt")
        torch.save(self.cnn.state_dict(),  self.model_dir / "cnn.pt")
        torch.save(self.meta.state_dict(), self.model_dir / "meta.pt")
        joblib.dump(self.rf_list,  self.model_dir / "rf_list.pkl")
        joblib.dump(self.lgb_list, self.model_dir / "lgb_list.pkl")
        joblib.dump(self.xgb_list, self.model_dir / "xgb_list.pkl")
        print(f"✅ Modelle gespeichert → {self.model_dir}")

    # ----------------------------------------------------- Hilfs-Methoden
    @staticmethod
    def _flat(X): return X.reshape(len(X), -1)

        # ---------------------------------------------------------------- build_features
    def build_features(self, X):
        """
        Placeholder – hier könntest du später Feature-Engineering einbauen.
        Aktuell: Identity-Funktion.
        """
        return X

    # ---------------------------------------------------------------- train_final
    def train_final(self, *_, **__):
        """
        Gibt das Meta-Modell zurück, damit BaseTrainer.run()
        es nach dem Training abspeichern kann.
        """
        return self.meta

    # ------------------------ Optuna-Objectives (gekürzt im Chat) -------
    # ───── Random-Forest – Optuna-Objective ─────────────────────────
    def _rf_objective(self, trial, Xf: np.ndarray, y: np.ndarray):
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import log_loss

        X_flat = Xf
        
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_flat, y, test_size=0.2, shuffle=False
        )

        params = {
            "n_estimators":      trial.suggest_int("n_estimators",      50, 400),
            "max_depth":         trial.suggest_int("max_depth",          3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split",  2, 10),
            "max_features":      trial.suggest_categorical(
                                    "max_features", ["sqrt", "log2", None]),
            "n_jobs":     -1,
            "random_state": 42,
        }
        model = RandomForestClassifier(**params)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        return log_loss(y_va, proba)

    # ───── LightGBM – Optuna-Objective ──────────────────────────────
    def _lgb_objective(self, trial, Xf: np.ndarray, y: np.ndarray):
        import lightgbm as lgb
        from sklearn.metrics import log_loss
        from sklearn.model_selection import train_test_split

        # Split
        X_tr, X_va, y_tr, y_va = train_test_split(Xf, y, test_size=0.2, shuffle=False)

        # Optuna-Suche inkl. fixer Rundenanzahl
        params = {
            "objective":        "binary",
            "metric":           "binary_logloss",
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 15, 150),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "verbosity":        -1,
        }
        n_rounds = trial.suggest_int("num_boost_round", 150, 800)  # <— statt early stopping

        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)

        model = lgb.train(params, dtr, num_boost_round=n_rounds, valid_sets=[dva])
        preds = model.predict(X_va)  # nutzt alle n_rounds
        return log_loss(y_va, preds)

    # ───── XGBoost – Optuna-Objective ───────────────────────────────
    def _xgb_objective(self, trial, Xf: np.ndarray, y: np.ndarray):
        from sklearn.model_selection import train_test_split
        import xgboost as xgb
        from sklearn.metrics import log_loss

        X_flat = Xf
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_flat, y, test_size=0.2, shuffle=False
        )

        params = {
            "objective":        "binary:logistic",
            "eval_metric":      "logloss",
            "eta":              trial.suggest_float("eta", 0.01, 0.2, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "lambda":           trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)
        model = xgb.train(
            params, dtr, num_boost_round=500,
            evals=[(dva, "val")], early_stopping_rounds=30,
            verbose_eval=10
        )
        preds = model.predict(dva, iteration_range=(0, model.best_iteration))
        return log_loss(y_va, preds)

    # ───── Simple-CNN – Optuna-Objective ────────────────────────────
    def _cnn_objective(self, trial):
        import torch, torch.nn as nn
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import log_loss

        lr        = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        n_filters = trial.suggest_int("n_filters", 16, 64)

        model = SimpleCNN(n_feat=self.X_train.shape[2],
                        n_filters=n_filters).to(self.device)
        crit  = nn.BCEWithLogitsLoss()
        opt   = torch.optim.Adam(model.parameters(), lr=lr)

        X_tr, X_va, y_tr, y_va = train_test_split(
            self.X_train, self.y_train, test_size=0.2, shuffle=False
        )
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32,
                            device=self.device).permute(0, 2, 1)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=self.device)

        model.train()
        for _ in range(5):
            opt.zero_grad()
            loss = crit(model(X_tr_t).squeeze(), y_tr_t)
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            X_va_t = torch.tensor(X_va, dtype=torch.float32,
                                  device=self.device).permute(0, 2, 1)
            logits = model(X_va_t).squeeze().cpu().numpy()
            probs  = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
        return log_loss(y_va, probs)

    # ───────────────── FT-Transformer Optuna-Objective ──────────────────
    def _ft_objective(self, trial, Xf: np.ndarray, y: np.ndarray) -> float:
        """
        Optuna-Ziel­funktion:
          • lr        – Lernrate
          • n_blocks  – Transformer-Blöcke
        liefert den Validierungs-LogLoss (je kleiner, desto besser)
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import Trainer, TrainingArguments

        # 1) Hyperparameter-Vorschläge
        hp = {
            "lr":       trial.suggest_float("lr", 1e-6, 5e-4, log=True),
            "n_blocks": trial.suggest_int(  "n_blocks", 2,     6),
        }

        # 2) FT-Backbone + LoRA erstellen
        base_ft = FTTransformer.make_default(
            n_num_features    = Xf.shape[1],
            cat_cardinalities = (),
            d_out             = 1,
            n_blocks          = hp["n_blocks"]
        )
        peft_ft = get_peft_model(
            base_ft,
            LoraConfig(r=4, lora_alpha=16, lora_dropout=0.05,
                       target_modules=["ffn.linear_first"])
        )

        # 3) Focal/BCE-Wrapper mit pos_weight
        pos_weight = torch.tensor(
            [(len(y) - y.sum()) / (y.sum() + 1e-6)],
            device=self.device
        )
        gamma = 0.0
        if self.cfg.get("loss", {}).get("use_focal", False):
            gamma = float(self.cfg.get("loss", {}).get("focal_gamma", 0.0))
        model = FTWrapped(
            ft_base          = peft_ft,
            pos_weight       = pos_weight,
            label_smooth_eps = 0.10,
            focal_gamma      = gamma
        ).to(self.device)

        # 4) 80/20-Split für Optuna-Val
        split = int(0.8 * len(Xf))
        X_tr, X_va = Xf[:split], Xf[split:]
        y_tr, y_va = y [:split], y [split:]

        ds_tr = NumpyDataset(X_tr, y_tr)
        ds_va = NumpyDataset(X_va, y_va)

        # 5) HF-Trainer-Setup
        args = TrainingArguments(
            output_dir             = f"{self.model_dir}/opt_ft_tmp",
            per_device_train_batch_size = 32,
            per_device_eval_batch_size  = 32,
            num_train_epochs       = self.cfg["training"]["ft_optuna_epochs"],
            learning_rate          = hp["lr"],
            weight_decay           = 1e-2,
            no_cuda                = (self.device.type == "cpu"),
            fp16                   = False,
            logging_steps          = 50,
            report_to              = [],
            eval_strategy          = IntervalStrategy.EPOCH,
            save_strategy          = IntervalStrategy.NO,
            load_best_model_at_end = True,
            metric_for_best_model  = "eval_loss",
        )
        trainer = Trainer(
            model         = model,
            args          = args,
            train_dataset = ds_tr,
            eval_dataset  = ds_va,
            data_collator = numeric_collate,
            compute_metrics= None,
        )
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience = 2)
        )
        trainer.train()

        # 6) Eval-Loss des besten Checkpoints ermitteln
        eval_res = trainer.evaluate(ds_va)
        val_loss = float(eval_res["eval_loss"])

        # Aufräumen (RAM & GPU-Cache)
        del trainer, model, peft_ft, base_ft, ds_tr, ds_va
        torch.cuda.empty_cache(); gc.collect()

        return val_loss

    # ───── Meta-Transformer – Optuna-Objective (mit Regime) ─────────
    def _meta_objective(self, trial,
                        preds_train: np.ndarray,
                        y_train:    np.ndarray,
                        regime_train: np.ndarray | None = None):
        import numpy as np
        import torch, torch.nn as nn
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import log_loss
        from torch.utils.data import TensorDataset, DataLoader

        d_token = trial.suggest_int("d_token", 16, 128)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        lr      = trial.suggest_float("lr", 1e-4, 1e-3, log=True)

        X_tr, X_va, y_tr, y_va = train_test_split(
            preds_train, y_train, test_size=0.2, shuffle=False
        )
        if regime_train is not None:
            R_tr, R_va, _, _ = train_test_split(
                regime_train, y_train, test_size=0.2, shuffle=False)
            X_tr = np.hstack([X_tr, R_tr])
            X_va = np.hstack([X_va, R_va])

        input_dim  = X_tr.shape[1]
        meta       = MetaTransformer(input_dim=input_dim,
                                    d_token=d_token,
                                    dropout=dropout).to(self.device)
        optim      = torch.optim.Adam(meta.parameters(), lr=lr)
        crit       = nn.BCEWithLogitsLoss()

        ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                        torch.tensor(y_tr, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=256, shuffle=True)

        meta.train()
        for _ in range(5):
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = crit(meta(xb).squeeze(), yb)
                optim.zero_grad(); loss.backward(); optim.step()

        meta.eval()
        with torch.no_grad():
            prob_val = torch.sigmoid(
                meta(torch.tensor(X_va, dtype=torch.float32,
                                device=self.device))
            ).cpu().numpy()
        return log_loss(y_va, prob_val)
