# trainers/hybrid_longtrend_trainer.py
# ============================================================================
# HybridLongTrendTrainer  –  CPU-freundliches Ensemble (RF, LGB, XGB, FT, CNN)
# mit Optuna-Tuning, Focal-Loss, Regime-Features & Temperature-Scaling
# ============================================================================

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os, gc, math, joblib, optuna, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
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
        # konsistente Typen (HF-Trainer + Torch)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return {"x_num": self.X[idx], "labels": self.y[idx]}

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

def calibrate_temperature(model, val_loader):
    device = next(model.parameters()).device
    scaler = TemperatureScaler(model).to(device)
    opt = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    # Wir optimieren NUR die Temperatur, nicht das Modell:
    for p in scaler.model.parameters():
        p.requires_grad = False
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    def _loss():
        opt.zero_grad()
        total, n = 0.0, 0
        for x, y in val_loader:
            x = x.to(device); y = y.to(device).float()
            logits = scaler(x).squeeze()
            total += bce(logits, y)
            n += y.numel()
        loss = total / max(1, n)
        loss.backward()
        return loss
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
    # Robust gegen pathologische Objectives: 0.0/NaN/Inf dürfen nie "best" werden
    def _safe_objective(trial):
        val = objective_fn(trial)
        if val is None or not np.isfinite(val) or float(val) <= 0.0:
            return float("inf")
        return float(val)
    study.optimize(_safe_objective, n_trials=n_trials, show_progress_bar=True, callbacks=[cb])

    pkl = save_dir / f"{study_name}.pkl"
    joblib.dump(study, pkl)
    study.trials_dataframe().to_csv(save_dir / f"{study_name}_trials.csv", index=False)
    return study

# ---- Kleine Hülle, damit BaseTrainer.run() immer ein "study.best_params" bekommt
class _SimpleStudy:
    def __init__(self, best_params: dict):
        self.best_params = best_params

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
        # CPU-Threads für RF/LGB/XGB (konfigurierbar: hardware.cpu_threads)
        self.cpu_threads = int(self.cfg.get("hardware", {}).get(
            "cpu_threads",
            max(1, (os.cpu_count() or 4)//2)
        ))
        # Auf der GPU etwas Headroom lassen (reduziert Freezes)
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(0.9)
            except Exception:
                pass

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

    def _predict_ft_batched(self, model, X, batch_size=64):
        dev = next(model.parameters()).device
        outs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=dev)
                with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
                    logits = model(xb)["logits"]
                outs.append(torch.sigmoid(logits).float().cpu().numpy())
        return np.concatenate(outs).astype(np.float32)


    # -------------------------------------------------- Basis-Model-Trainer
    def _train_rf(self, X, y):
        Xf = self._flat(X)
        def obj(t): return self._rf_objective(t, Xf, y)
        best = run_optuna_and_save(obj, self.cfg["optuna"]["n_trials"],
                                   "rf_study", self.model_dir).best_params
        # Merke Best-Params, damit wir sie am Ende als study.best_params zurückgeben können
        self.best_rf_params = dict(best)
        # Full-fit Ensemble (konfigurierbar, default=2 statt 3 → schneller)
        rf_n_models = int(self.cfg.get("rf", {}).get("n_full_models", 3))
        rf_max_samples = self.cfg.get("rf", {}).get("max_samples", None)  # z.B. 0.7 für 70%
        models, rng = [], np.random.default_rng(42)
        for _ in range(rf_n_models):
            idx = rng.choice(len(Xf), len(Xf), replace=True)
            m   = RandomForestClassifier(
                    **best,
                    n_jobs=self.cpu_threads,
                    bootstrap=True,
                    max_samples=rf_max_samples
                )
            # <— fehlte: trainieren und in Ensemble aufnehmen
            m.fit(Xf[idx], y[idx])
            models.append(m)
        # OOF (K=5)
        tss = TimeSeriesSplit(n_splits=5)
        oof = np.zeros(len(Xf), dtype=np.float32)
        for tr, va in tss.split(Xf):
            # WICHTIG: n_jobs setzen, sonst 1 Thread → sehr langsam
            m = RandomForestClassifier(
                    **best,
                    n_jobs=self.cpu_threads,
                    bootstrap=True,
                    max_samples=rf_max_samples
                )
            m.fit(Xf[tr], y[tr])
            oof[va] = m.predict_proba(Xf[va])[:,1]
        # Val-Preds
        Xv = self._flat(self.X_val)
        # Safety: falls models leer wäre (sollte jetzt nicht mehr passieren)
        if len(models) == 0:
            m_full = RandomForestClassifier(**best, n_jobs=self.cpu_threads)
            m_full.fit(Xf, y)
            models = [m_full]
        val_preds = np.mean([m.predict_proba(Xv)[:,1] for m in models], axis=0).astype(np.float32)
        return models, oof, val_preds

    def _train_lgb(self, X, y):
        Xf = self._flat(X)
        best = run_optuna_and_save(
            lambda t: self._lgb_objective(t, Xf, y),
            self.cfg["optuna"]["n_trials"], "lgb_study", self.model_dir
        ).best_params
        self.best_lgb_params = dict(best)
        # Full-fit
        d   = lgb.Dataset(Xf, label=y)
        models = [lgb.train({**best,"objective":"binary","metric":"binary_logloss"}, d, 500)]
        # OOF
        tss = TimeSeriesSplit(n_splits=5)
        oof = np.zeros(len(Xf), dtype=np.float32)
        for tr, va in tss.split(Xf):
            dt = lgb.Dataset(Xf[tr], label=y[tr])
            mv = lgb.train({**best,"objective":"binary","metric":"binary_logloss"}, dt, 500)
            oof[va] = mv.predict(Xf[va]).astype(np.float32)
        # Val
        Xv = self._flat(self.X_val)
        val_preds = np.mean([m.predict(Xv) for m in models], axis=0).astype(np.float32)
        return models, oof, val_preds


    def _train_xgb(self, X, y):
        Xf = self._flat(X)
        best = run_optuna_and_save(
            lambda t: self._xgb_objective(t, Xf, y),
            self.cfg["optuna"]["n_trials"], "xgb_study", self.model_dir
        ).best_params
        params = {**best, "objective":"binary:logistic", "eval_metric":"logloss"}
        if "lambda_l2" in params:  # Optuna-Name → XGB-Param
            params["lambda"] = params.pop("lambda_l2")
        self.best_xgb_params = dict(best)
        # Full-fit
        d = xgb.DMatrix(Xf, label=y)
        model = xgb.train(params, d, 500)
        models = [model]
        # OOF
        tss = TimeSeriesSplit(n_splits=5)
        oof = np.zeros(len(Xf), dtype=np.float32)
        for tr, va in tss.split(Xf):
            mt = xgb.train(params, xgb.DMatrix(Xf[tr], label=y[tr]), 500)
            oof[va] = mt.predict(xgb.DMatrix(Xf[va])).astype(np.float32)
        # Val
        Xv = self._flat(self.X_val)
        val_preds = np.mean([m.predict(xgb.DMatrix(Xv)) for m in models], axis=0).astype(np.float32)
        return models, oof, val_preds

    def _train_cnn(self, X, y):
        best = run_optuna_and_save(
            lambda t: self._cnn_objective(t),  # nutzt self.X_train intern
            self.cfg["optuna"]["n_trials"], "cnn_study", self.model_dir
        ).best_params
        self.best_cnn_params = dict(best)
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
        # OOF
        tss = TimeSeriesSplit(n_splits=5)
        oof = np.zeros(len(X), dtype=np.float32)
        for tr, va in tss.split(X):
            m = SimpleCNN(X.shape[2], best["n_filters"]).to(self.device)
            o = optim.Adam(m.parameters(), lr=best["lr"])
            m.train()
            Xtr = torch.tensor(X[tr], dtype=torch.float32, device=self.device).permute(0,2,1)
            ytr = torch.tensor(y[tr], dtype=torch.float32, device=self.device)
            for _ in range(5):
                o.zero_grad(); L = crit(m(Xtr).squeeze(), ytr); L.backward(); o.step()
            m.eval()
            Xva = torch.tensor(X[va], dtype=torch.float32, device=self.device).permute(0,2,1)
            oof[va] = torch.sigmoid(m(Xva).squeeze()).detach().cpu().numpy().astype(np.float32)
        # Val
        self.cnn = mdl.cpu()
        Xv = torch.tensor(self.X_val, dtype=torch.float32).permute(0,2,1)
        val_preds = torch.sigmoid(self.cnn(Xv)).detach().cpu().numpy().astype(np.float32)
        return self.cnn, oof, val_preds

    # ---------------------------------------------------- FT-Transformer
    def _train_ft(self, X, y):
        Xf = self._flat(X)
        study = run_optuna_and_save(
            lambda t: self._ft_objective(t, Xf, y),
            self.cfg["optuna"]["n_trials"], "ft_study", self.model_dir)
        hp = study.best_params
        self.best_ft_params = dict(hp)
        ft = FTTransformer.make_default(
            n_num_features=Xf.shape[1], cat_cardinalities=(), d_out=1,
            n_blocks=hp["n_blocks"])
        ft = get_peft_model(
            ft, LoraConfig(r=4, lora_alpha=16, lora_dropout=0.05,
                           target_modules=["ffn.linear_first"]))
        posw = torch.tensor([(len(y)-y.sum())/(y.sum()+1e-6)], device=self.device)
        # Konsistente Config: Focal-Loss-Settings kommen aus cfg["loss"]
        use_focal = self.cfg.get("loss", {}).get("use_focal", False)
        gamma = float(self.cfg.get("loss", {}).get("focal_gamma", 0.0)) if use_focal else 0.0
        model = FTWrapped(
            ft, pos_weight=posw, label_smooth_eps=0.1, focal_gamma=gamma
        )
        ds   = NumpyDataset(Xf, y)
        Trainer(model, TrainingArguments(
            output_dir=f"{self.model_dir}/ft_final",
            per_device_train_batch_size=16, num_train_epochs=8,
            learning_rate=hp["lr"],
            no_cuda=(self.device.type == "cpu"),
            fp16=False,
            save_strategy="no", eval_strategy="no",
            report_to=[]
        ), train_dataset=ds, data_collator=numeric_collate).train()
        model = model.cpu()
        # OOF
        tss = TimeSeriesSplit(n_splits=5)
        oof = np.zeros(len(Xf), dtype=np.float32)
        for tr, va in tss.split(Xf):
            ft = FTTransformer.make_default(
                n_num_features=Xf.shape[1], cat_cardinalities=(), d_out=1, n_blocks=hp["n_blocks"])
            ft = get_peft_model(ft, LoraConfig(r=4, lora_alpha=16, lora_dropout=0.05,
                                               target_modules=["ffn.linear_first"]))
            posw = torch.tensor([(len(y[tr])-y[tr].sum())/(y[tr].sum()+1e-6)], device=self.device)
            m = FTWrapped(ft, pos_weight=posw, label_smooth_eps=0.1, focal_gamma=gamma).to(self.device)
            Trainer(m, TrainingArguments(output_dir=f"{self.model_dir}/ft_oof",
                   per_device_train_batch_size=16, num_train_epochs=4, learning_rate=hp["lr"],
                   no_cuda=(self.device.type == "cpu"), fp16=True, save_strategy="no", eval_strategy="no", report_to=[]),
                   train_dataset=NumpyDataset(Xf[tr], y[tr]), data_collator=numeric_collate).train()
            oof[va] = self._predict_ft_batched(m, Xf[va], batch_size=64)
        # Val
        val_preds = self._predict_ft_batched(model, self._flat(self.X_val), batch_size=64)
        return model, oof, val_preds

    # ----------------------------------------------------- Meta-Training (robust)
    def _build_meta_inputs(self, preds_oof: np.ndarray, preds_val: np.ndarray, L: int = 12):
        """
        Baut History- und Kontext-Features für den sequenziellen Meta-Stack:
          - History: letzte L Schritte je Basismodell
          - Kontext: rollierender LogLoss im Train (aus OOF) und optional Regime-Features
        """
        K = preds_oof.shape[1]
        regime_tr = None
        regime_va = None
        if self.cfg["meta"].get("use_regime", False):
            cols = self.cfg["data"]["numerical_cols"]
            def regime_from_seq(X_seq):
                iH, iL, iC = cols.index("high"), cols.index("low"), cols.index("Close")
                high, low, close = X_seq[:,:,iH], X_seq[:,:,iL], X_seq[:,:,iC]
                tr = np.maximum(high, np.roll(close,1,axis=1)) - np.minimum(low, np.roll(close,1,axis=1))
                tr[:,0] = (high[:,0] - low[:,0])
                atr = tr.mean(axis=1)
                w = min(20, close.shape[1])
                c_last = close[:,-w:]; ma = c_last.mean(axis=1); sd = c_last.std(axis=1)+1e-9
                upper, lower = ma+2*sd, ma-2*sd
                bbp = (close[:,-1]-lower)/(upper-lower+1e-9)
                return np.stack([atr, bbp], axis=1).astype(np.float32)
            regime_tr = regime_from_seq(self.X_train)
            regime_va = regime_from_seq(self.X_val)

        # Rolling-Performance (nur Train,  aus OOF)
        roll_perf_tr = []
        eps = 1e-7
        for j in range(K):
            p = np.clip(preds_oof[:, j], eps, 1-eps)
            y = self.y_train
            ll = -(y*np.log(p) + (1-y)*np.log(1-p))
            rp = pd.Series(ll).rolling(L, min_periods=1).mean().to_numpy().astype(np.float32)
            roll_perf_tr.append(rp)
        roll_perf_tr = np.stack(roll_perf_tr, axis=1)  # [Ntr, K]

        # History-Sequenzen
        def make_hist(preds):
            N = len(preds); hist = np.zeros((N, L, K), dtype=np.float32)
            for t in range(N):
                s = max(0, t-L+1); window = preds[s:t+1]
                hist[t, -len(window):, :] = window
            return hist
        H_tr = make_hist(preds_oof)
        H_va = make_hist(preds_val)

        # Kontext (Regime + roll. Perf)
        C_tr = roll_perf_tr if regime_tr is None else np.hstack([roll_perf_tr, regime_tr])
        C_va = np.zeros((len(preds_val), C_tr.shape[1]), dtype=np.float32)
        if regime_va is not None:
            pad = C_tr.shape[1] - regime_va.shape[1]
            C_va = np.hstack([np.zeros((len(preds_val), pad), dtype=np.float32), regime_va])

        return (H_tr, C_tr, self.y_train.astype(np.float32)), \
               (H_va, C_va, self.y_val.astype(np.float32))

    def _meta_objective_seq(self, trial, H_tr, C_tr, y_tr, H_va, C_va, y_va):
        # kleine, schnelle Suche
        d_model = trial.suggest_int("d_token", 32, 128)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        lr      = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        n_heads = self._pick_n_heads(d_model, max_heads=4)
        n_layers= 1

        K = H_tr.shape[2]
        meta = MetaMoE(K=K, L=H_tr.shape[1], ctx_dim=C_tr.shape[1],
                       d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                       dropout=dropout).to(self.device)
        opt  = torch.optim.Adam(meta.parameters(), lr=lr, weight_decay=5e-4)
        use_amp = (self.device.type == "cuda")
        scaler  = GradScaler(enabled=use_amp)
        # Zero-Copy: vermeidet RAM-Verdopplung (Torch teilt Speicher mit NumPy)
        H_t = torch.from_numpy(H_tr) if isinstance(H_tr, np.ndarray) else H_tr
        C_t = torch.from_numpy(C_tr) if isinstance(C_tr, np.ndarray) else C_tr
        y_t = torch.from_numpy(y_tr) if isinstance(y_tr, np.ndarray) else y_tr
        ds = TensorDataset(H_t, C_t, y_t)
        # Kleinere physische Batch, um Peak-Speicher zu reduzieren
        dl = DataLoader(
            ds,
            batch_size=64,           # vorher 128
            shuffle=True,
            num_workers=0,           # keine Extra-Kopien durch Worker
            pin_memory=False,
            drop_last=True
        )

        eps = 0.02
        ent_w = self.cfg["meta"].get("entropy_weight", 1e-3)
        tv_w  = self.cfg["meta"].get("tv_weight", 1e-3)
        meta.train()
        for _ in range(5):
            for H, C, yb in dl:
                H = H.to(self.device); C = C.to(self.device); yb = yb.to(self.device)
                with autocast(enabled=use_amp):
                    w, p_now = meta(H, C)                    # [B,K], [B,K]
                    p_base = H[:, -1, :]                     # echte Basis-Preds
                    alpha  = self.cfg.get("meta", {}).get("alpha_base_mix", 1.0)
                    p_mix  = alpha * p_base + (1.0 - alpha) * p_now
                    p_hat  = (w * p_mix).sum(dim=1)
                    p_hat  = torch.nan_to_num(p_hat, nan=0.5).clamp(1e-6, 1-1e-6)
                    y_s    = torch.nan_to_num(yb*(1-eps) + 0.5*eps, nan=0.5).clamp(1e-6, 1-1e-6)
                    ent    = -(w * (w.clamp_min(1e-8)).log()).sum(dim=1).mean()
                    tv     = meta.tv_penalty().mean()
                # BCE auf Wahrscheinlichkeiten **außerhalb** von autocast, stabil in FP32
                bce = nn.functional.binary_cross_entropy(p_hat.float(), y_s.float())
                loss = bce - ent_w*ent + tv_w*tv
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        # ===== gebatchte Validierung, um VRAM-Peaks zu vermeiden =====
        meta.eval()
        with torch.no_grad():
            Hv = torch.from_numpy(H_va).to(torch.float32)
            Cv = torch.from_numpy(C_va).to(torch.float32)
            ds_va = TensorDataset(Hv, Cv)
            dl_va = DataLoader(ds_va, batch_size=1024, shuffle=False)
            preds = []
            for H, C in dl_va:
                H = H.to(self.device); C = C.to(self.device)
                with autocast(enabled=use_amp):
                    w, p_now = meta(H, C)
                    p_base   = H[:, -1, :]
                    alpha    = self.cfg.get("meta", {}).get("alpha_base_mix", 1.0)
                    p_mix    = alpha * p_base + (1.0 - alpha) * p_now
                    p_hat    = (w * p_mix).sum(dim=1)
                preds.append(p_hat.float().cpu().numpy())
            p_hat = np.clip(np.nan_to_num(np.concatenate(preds), nan=0.5), 1e-7, 1-1e-7)
        # Aufräumen nach Trial
        del Hv, Cv, ds_va, dl_va
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return log_loss(y_va, p_hat)

    def _train_meta(self, preds_oof: np.ndarray, preds_val: np.ndarray):
        (H_tr, C_tr, y_tr), (H_va, C_va, y_va) = self._build_meta_inputs(preds_oof, preds_val)
        best = run_optuna_and_save(
            lambda t: self._meta_objective_seq(t, H_tr, C_tr, y_tr, H_va, C_va, y_va),
            self.cfg["optuna"]["n_trials"], "meta_study", self.model_dir
        ).best_params
        # Merken für Rückgabe
        self.best_meta_params = dict(best)
        d_model = best["d_token"]; dropout = best["dropout"]; n_heads = self._pick_n_heads(d_model, max_heads=4)
        meta = MetaMoE(K=H_tr.shape[2], L=H_tr.shape[1], ctx_dim=C_tr.shape[1],
                       d_model=d_model, n_heads=n_heads, n_layers=1,
                       dropout=dropout).to(self.device)
        opt  = torch.optim.Adam(meta.parameters(), lr=best["lr"], weight_decay=5e-4)
        # Zero-Copy + kleinere physische Batch
        H_t = torch.from_numpy(H_tr) if isinstance(H_tr, np.ndarray) else H_tr
        C_t = torch.from_numpy(C_tr) if isinstance(C_tr, np.ndarray) else C_tr
        y_t = torch.from_numpy(y_tr) if isinstance(y_tr, np.ndarray) else y_tr
        ds = TensorDataset(H_t, C_t, y_t)
        dl = DataLoader(
            ds,
            batch_size=64,       # vorher 128
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        eps  = 0.02
        ent_w = self.cfg["meta"].get("entropy_weight", 1e-3)
        tv_w  = self.cfg["meta"].get("tv_weight", 1e-3)
        use_amp = (self.device.type == "cuda")
        scaler  = GradScaler(enabled=use_amp)
        meta.train()
        for _ in range(10):
            for H, C, yb in dl:
                H = H.to(self.device); C = C.to(self.device); yb = yb.to(self.device)
                with autocast(enabled=use_amp):
                    w, p_now = meta(H, C)
                    p_base = H[:, -1, :]
                    alpha  = self.cfg.get("meta", {}).get("alpha_base_mix", 1.0)
                    p_mix  = alpha * p_base + (1.0 - alpha) * p_now
                    p_hat  = (w * p_mix).sum(dim=1)
                    p_hat  = torch.nan_to_num(p_hat, nan=0.5).clamp(1e-6, 1-1e-6)
                    y_s    = torch.nan_to_num(yb*(1-eps) + 0.5*eps, nan=0.5).clamp(1e-6, 1-1e-6)
                    ent    = -(w * (w.clamp_min(1e-8)).log()).sum(dim=1).mean()
                    tv     = meta.tv_penalty().mean()
                # BCE auf Wahrscheinlichkeiten **außerhalb** von autocast, stabil in FP32
                bce = nn.functional.binary_cross_entropy(p_hat.float(), y_s.float())
                loss = bce - ent_w*ent + tv_w*tv
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        return meta.cpu(), self.best_meta_params

    # ----------------------------------------------------------- optimize
    def optimize(self, X, y):

        print("\n=== RF: Optuna + Final-Training ===", flush=True)
        self.rf_list,  rf_oof,  rf_val  = self._train_rf(X, y)
        print("=== RF: fertig ===", flush=True)

        print("\n=== LGB: Optuna startet ===", flush=True)
        self.lgb_list, lgb_oof, lgb_val = self._train_lgb(X, y)
        print("=== LGB: fertig ===", flush=True)

        print("\n=== XGB: Optuna startet ===", flush=True)
        self.xgb_list, xgb_oof, xgb_val = self._train_xgb(X, y)
        print("=== XGB: fertig ===", flush=True)

        print("\n=== CNN: Optuna startet ===", flush=True)
        self.cnn,      cnn_oof, cnn_val = self._train_cnn(X, y)
        print("=== CNN: fertig ===", flush=True)

        print("\n=== FT: Optuna startet ===", flush=True)
        self.ft,        ft_oof,  ft_val = self._train_ft(X, y)
        print("=== FT: fertig ===", flush=True)


        # Stacking-Matrizen: OOF (Train) & VAL (Val-Fold) — defensiv clippen
        preds_oof = np.vstack([rf_oof, lgb_oof, xgb_oof, ft_oof, cnn_oof]).T
        preds_val = np.vstack([rf_val, lgb_val, xgb_val, ft_val, cnn_val]).T
        preds_oof = np.clip(np.nan_to_num(preds_oof, nan=0.5), 1e-7, 1-1e-7).astype(np.float32)
        preds_val = np.clip(np.nan_to_num(preds_val, nan=0.5), 1e-7, 1-1e-7).astype(np.float32)



        # Meta-Stack trainieren (OOF + VAL → sequenzieller Meta-Encoder)
        try:
            # _train_meta liefert (meta_model, best_meta_params)
            self.meta, self.best_meta_params = self._train_meta(preds_oof, preds_val)
        except Exception as e:
            print(f"⚠️  Meta-Training fehlgeschlagen: {e}. Fallback auf einfache LogReg.")
            # Fallback, damit das Ensemble *immer* fertig gebaut wird:
            from sklearn.linear_model import LogisticRegression
            Xtr = preds_oof
            Xva = preds_val
            lr  = LogisticRegression(max_iter=1000)
            lr.fit(Xtr, self.y_train)
            # kleiner Wrapper, damit save_model funktioniert
            class _MetaFallback(nn.Module):
                def __init__(self, sk):
                    super().__init__(); self.sk = sk
                def forward(self, x):
                    with torch.no_grad():
                        p = torch.tensor(self.sk.predict_proba(x.cpu().numpy())[:,1])
                    return p
            self.meta = _MetaFallback(lr).cpu()
            # Markiere Fallback-Params, damit base.run() nicht crasht
            self.best_meta_params = {"type": "logreg_fallback"}

        # Temperature-Scaling (FT)
        X_val_flat = self._flat(self.X_val)
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val_flat, dtype=torch.float32),
                          torch.tensor(self.y_val,   dtype=torch.float32)),
            batch_size=328, shuffle=False
        )
        T = calibrate_temperature(_FTLogits(self.ft), val_loader)
        torch.save({"temperature": T}, self.model_dir / "temp_scaler.pt")

        # ---- Gib eine Study-ähnliche Struktur zurück, damit BaseTrainer.run() nicht auf None läuft
        combined_best = {
            "rf":  getattr(self, "best_rf_params",  {}),
            "lgb": getattr(self, "best_lgb_params", {}),
            "xgb": getattr(self, "best_xgb_params", {}),
            "cnn": getattr(self, "best_cnn_params", {}),
            "ft":  getattr(self, "best_ft_params",  {}),
            "meta":getattr(self, "best_meta_params",{}),
        }
        return _SimpleStudy(combined_best)

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
    @staticmethod
    def _pick_n_heads(d_model: int, max_heads: int = 8) -> int:
        """
        Wählt die größte Potenz-von-2 (1,2,4,8) <= max_heads, die d_model teilt.
        Gewährleistet: d_model % n_heads == 0 (Pflicht für MultiheadAttention).
        """
        for h in (8, 4, 2, 1):
            if h <= max_heads and d_model % h == 0:
                return h
        return 1
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
            X_va_t = torch.tensor(X_va, dtype=torch.float32, device=self.device).permute(0, 2, 1)
            logits_t = model(X_va_t).squeeze()                 # Tensor, kein zweiter Forward-Pass
            probs = torch.sigmoid(logits_t).cpu().numpy()      # stabile Sigmoid in Torch
            probs = np.clip(probs, 1e-7, 1 - 1e-7)             # gegen 0/1 clampen
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
        import gc

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
            per_device_train_batch_size = 16,
            per_device_eval_batch_size  = 32,
            num_train_epochs       = self.cfg["training"]["ft_optuna_epochs"],
            learning_rate          = hp["lr"],
            weight_decay           = 1e-2,
            no_cuda                = (self.device.type == "cpu"),
            fp16                   = False,
            logging_steps          = 50,
            report_to              = [],
            eval_strategy          = IntervalStrategy.EPOCH,
            save_strategy          = IntervalStrategy.EPOCH,
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
        val_loss = float(eval_res.get("eval_loss", float("inf")))
        # Safety: nie 0/NaN/Inf als "guten" Wert zurückgeben
        if (not np.isfinite(val_loss)) or (val_loss <= 0.0):
            val_loss = float("inf")

        # Aufräumen (RAM & GPU-Cache)
        del trainer, model, peft_ft, base_ft, ds_tr, ds_va
        torch.cuda.empty_cache(); gc.collect()

        return val_loss

 # ========================================================================== #
 # MetaMoEclassTrainer
 # ========================================================================== #
class MetaMoE(nn.Module):
    def __init__(self, K:int, L:int, ctx_dim:int, d_model:int=96,
                 n_heads:int=2, n_layers:int=1, dropout:float=0.2):
        super().__init__()
        self.K, self.L = K, L
        self.input_proj = nn.Linear(K, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               dim_feedforward=4*d_model, dropout=dropout,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ctx_proj = nn.Linear(ctx_dim, d_model)
        # Gewichts-Kopf (Softmax über K)
        self.w_head = nn.Linear(d_model, K)
        # Per-Modell-Probs Kopf (sigmoid) – „auch eigenständige Vorhersagen“
        self.p_head = nn.Linear(d_model, K)
        self.dropout = nn.Dropout(dropout)
        self._last_weights = None  # für TV-Penalty

    def forward(self, hist: torch.Tensor, ctx: torch.Tensor):
        """
        hist: [B, L, K]  – Pred-Historie der K Modelle (zuletzt rechts)
        ctx : [B, C]     – Kontext (Regime, Rolling-Perf)
        """
        x = self.input_proj(hist)                      # [B,L,d]
        x = self.encoder(x)                            # [B,L,d]
        x_last = x[:, -1, :]                           # letzter Schritt
        x_fuse = x_last + self.ctx_proj(ctx)           # einfache Fusion
        x_fuse = self.dropout(x_fuse)
        w = nn.functional.softmax(self.w_head(x_fuse), dim=-1)  # [B,K]
        p_now = torch.sigmoid(self.p_head(x_fuse))              # [B,K]
        # TV speichern: weights entlang Sequenz (approx: letzten beiden Schritte)
        self._last_weights = w.detach()
        return w, p_now

    def tv_penalty(self):
        # einfache Approximation: Glätte innerhalb Batch (Lauf-Batch als t,t-1)
        if self._last_weights is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        # L1-Norm gegenüber Batch-Shift (surrogate)
        w = self._last_weights
        return (w[1:] - w[:-1]).abs().mean() if w.size(0) > 1 else torch.tensor(0.0, device=w.device)

# --- Ende MetaMoE -----------------------------------------------------------