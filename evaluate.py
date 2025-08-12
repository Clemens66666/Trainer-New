#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate.py — Ensemble-/Komponenten-Evaluierung für HybridLongTrendTrainer

Fähigkeiten
-----------
- DA/MDA (Directional Accuracy; MDA mit neutraler Klasse via Band um 0.5)
- Pesaran–Timmermann (DA-z-Test) auf aktiven Phasen (ohne "flat")
- Kalibrierung: Brier-Score, Expected Calibration Error (ECE), Reliability-Plot
- Trading-Metriken einer einfachen Schwellen-Strategie:
  Win-Rate, Profit-Factor, Profit/Loss-Ratio, Expectancy (APPT)
- Komponenten-Report (RF, LGB, XGB, FT, CNN), "only-X" und "leave-one-out"
- Robust, wenn einzelne Komponenten fehlen
- Optional: MetaMoE-Ensemble, wenn rekonstruierbar

Aufrufbeispiel (PowerShell)
---------------------------
python .\\evaluate.py `
  --model-dir models\\hybrid_longtrend_20250810_232923 `
  --test-file data\\longtrend.csv `
  --plots
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)

# ML libs
import joblib
import torch
import xgboost as xgb
import lightgbm as lgb

# ---- Projekt-Imports (Trainer, Features, Regime) ----
# MetaMoE & SimpleCNN leben (bei dir) im Hybrid-Trainer-File
from trainers.hybrid_longtrend_trainer import SimpleCNN, FTWrapped, extract_regime_features
# MetaMoE kann in neueren Versionen im gleichen Modul definiert sein:
try:
    from trainers.hybrid_longtrend_trainer import MetaMoE  # neues Meta
except Exception:
    MetaMoE = None

# --------------------------------------------------------------------
# Hilfsfunktionen: Laden, Preprocessing
# --------------------------------------------------------------------
def load_yaml_cfg(model_dir: Path) -> Dict:
    import yaml
    cfg_path = model_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def safe_joblib_or_pickle(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

def read_test_data(path: Path) -> pd.DataFrame:
    """
    Liest CSV/TXT. Erwartet mind. timestamp  OHLC oder price.
    """
    df = pd.read_csv(path)
    # unify timestamp
    ts_candidates = [c for c in df.columns if c.lower() in ("timestamp","time","datetime","date")]
    if not ts_candidates:
        raise ValueError("Keine Zeitspalte gefunden (timestamp/time/datetime/date).")
    df = df.rename(columns={ts_candidates[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    # unify price/ohlc
    cols = {c.lower(): c for c in df.columns}
    if "price" not in cols:
        # map close→price wenn möglich
        close_col = None
        for k in ("close","Close","last"):
            if k in df.columns:
                close_col = k; break
        if close_col is None:
            raise ValueError("Keine 'price' oder 'close' Spalte gefunden.")
        df["price"] = pd.to_numeric(df[close_col], errors="coerce")
    else:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # OHLC harmonisieren (optional)
    for want, cands in [("open", ["open","Open"]),
                        ("high", ["high","High"]),
                        ("low",  ["low","Low"]),
                        ("Close",["Close","close","last","price"])]:
        for c in cands:
            if c in df.columns:
                df[want] = pd.to_numeric(df[c], errors="coerce")
                break
    df = df.dropna(subset=["price"])
    return df

def make_trend_side_dc(price: pd.Series,
                       dc_thres: float = 0.5,
                       windows: Tuple[int,...] = (5,15,30),
                       tau: int = 1) -> pd.Series:
    """
    Primitive Directional-Change-Phasen -> Mehrheits-Scan -> trend_side ∈ {-1,0,1}
    """
    p = pd.to_numeric(price, errors="coerce").astype(float)
    last_ext = p.iloc[0]
    phase = np.zeros(len(p), dtype=int)
    direction = 0
    for i, val in enumerate(p):
        move = (val / last_ext - 1.0) * 100.0
        if direction >= 0 and move <= -dc_thres:
            direction = -1; last_ext = val
        elif direction <= 0 and move >=  dc_thres:
            direction =  1; last_ext = val
        phase[i] = direction
    dc = pd.Series(phase, index=price.index)
    scans = []
    for w in windows:
        scans.append(dc.rolling(w, min_periods=w).apply(lambda a: np.sign(a.sum()) if abs(a.sum())>=tau else 0.0))
    scans = pd.concat(scans, axis=1).fillna(0.0)
    smean = scans.mean(axis=1)
    side = smean.apply(lambda x: 1 if x>0 else (-1 if x<0 else 0)).astype(int)
    side.name = "trend_side"
    return side

# --------------------------------------------------------------------
# Modell-Lader
# --------------------------------------------------------------------
def try_load_rf(model_dir: Path):
    for name in ["rf_list.pkl","rf.pkl","rf_list.joblib"]:
        p = model_dir / name
        obj = safe_joblib_or_pickle(p)
        if obj:
            return obj
    return []

def try_load_lgb(model_dir: Path):
    for name in ["lgb_list.pkl","lgb.pkl","lgb_list.joblib"]:
        p = model_dir / name
        obj = safe_joblib_or_pickle(p)
        if obj:
            return obj
    return []

def try_load_xgb(model_dir: Path):
    for name in ["xgb_list.pkl","xgb.pkl","xgb_list.joblib"]:
        p = model_dir / name
        obj = safe_joblib_or_pickle(p)
        if obj:
            return obj
    return []

def try_load_ft(model_dir: Path, n_features: int) -> Optional[FTWrapped]:
    """
    Rekonstruiert FTWrapped  lädt state_dict aus ft.pt.
    Versucht n_blocks aus ft_study.pkl zu lesen, sonst Default.
    """
    ft_state_path = model_dir / "ft.pt"
    if not ft_state_path.exists():
        return None
    # n_blocks bestimmen
    n_blocks = 4
    study = safe_joblib_or_pickle(model_dir / "ft_study.pkl")
    if study and hasattr(study, "best_params"):
        bp = study.best_params
        if "n_blocks" in bp:
            n_blocks = int(bp["n_blocks"])
    # Backbone bauen
    from rtdl import FTTransformer
    from peft import LoraConfig, get_peft_model
    base = FTTransformer.make_default(
        n_num_features=n_features, cat_cardinalities=(), d_out=1, n_blocks=n_blocks
    )
    base = get_peft_model(base, LoraConfig(r=4, lora_alpha=16, lora_dropout=0.05,
                                           target_modules=["ffn.linear_first"]))
    model = FTWrapped(base, pos_weight=None, label_smooth_eps=0.0, focal_gamma=0.0)
    sd = torch.load(ft_state_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

def try_load_cnn(model_dir: Path, n_feat_guess: int) -> Optional[SimpleCNN]:
    path = model_dir / "cnn.pt"
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu")
    # state kann ein reines state_dict sein
    if isinstance(state, dict) and any(k.startswith("net.0.weight") for k in state.keys()):
        w = state["net.0.weight"]  # [n_filters, n_feat, 3]
        n_filters = w.shape[0]
        n_feat = w.shape[1]
        m = SimpleCNN(n_feat=int(n_feat), n_filters=int(n_filters))
        m.load_state_dict(state)
        m.eval()
        return m
    # falls komplett gespeichertes Modell (selten)
    if isinstance(state, SimpleCNN):
        state.eval()
        return state
    # Fallback
    m = SimpleCNN(n_feat=n_feat_guess, n_filters=32)
    try:
        m.load_state_dict(state, strict=False)
        m.eval()
        return m
    except Exception:
        return None

def try_load_meta(model_dir: Path, K: int, ctx_dim: int) -> Optional[torch.nn.Module]:
    """
    Baut MetaMoE gemäß meta_study.pkl (d_token/dropout); L ist flexibel.
    Rückgabe None, wenn Meta nicht rekonstruierbar.
    """
    meta_path = model_dir / "meta.pt"
    if not meta_path.exists() or MetaMoE is None:
        return None
    d_model = 96
    dropout = 0.3
    study = safe_joblib_or_pickle(model_dir / "meta_study.pkl")
    if study and hasattr(study, "best_params"):
        bp = study.best_params
        d_model = int(bp.get("d_token", d_model))
        dropout = float(bp.get("dropout", dropout))
    # n_heads so wählen, dass d_model teilbar ist (8/4/2/1)
    def pick_heads(d):
        for h in (8,4,2,1):
            if d % h == 0:
                return h
        return 1
    n_heads = pick_heads(d_model)
    meta = MetaMoE(K=K, L=32, ctx_dim=ctx_dim, d_model=d_model,
                   n_heads=n_heads, n_layers=1, dropout=dropout)
    try:
        sd = torch.load(meta_path, map_location="cpu")
        meta.load_state_dict(sd, strict=False)
        meta.eval()
        return meta
    except Exception:
        return None

# --------------------------------------------------------------------
# Inferenz pro Komponente
# --------------------------------------------------------------------
def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def predict_components(
    seq: np.ndarray,              # (N, L, F)
    rf_list, lgb_list, xgb_list,
    ft_model: Optional[FTWrapped],
    cnn_model: Optional[SimpleCNN],
    temperature_T: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Gibt Dict mit Komponentenvorhersagen (Wkeit für UP) pro Zeitindex zurück.
    """
    N, L, F = seq.shape
    X_flat = seq.reshape(N, L*F)

    out: Dict[str, np.ndarray] = {}

    # RF
    if rf_list:
        preds = [m.predict_proba(X_flat)[:,1] for m in rf_list]
        out["rf"] = np.mean(np.vstack(preds), axis=0)
    # LGB
    if lgb_list:
        preds = [m.predict(X_flat) for m in lgb_list]
        out["lgb"] = np.mean(np.vstack(preds), axis=0)
    # XGB
    if xgb_list:
        dm = xgb.DMatrix(X_flat)
        preds = [m.predict(dm) for m in xgb_list]
        out["xgb"] = np.mean(np.vstack(preds), axis=0)

    # FT (logits -> optional Temperature-Scaling)
    if ft_model is not None:
        with torch.no_grad():
            xb = torch.from_numpy(X_flat.astype(np.float32))
            out_logits = ft_model(xb)
            if isinstance(out_logits, dict):
                out_logits = out_logits["logits"].squeeze()
            logits = out_logits.cpu().numpy().ravel()
        if temperature_T and temperature_T > 0:
            logits = logits / float(temperature_T)
        out["ft"] = sigmoid_np(logits)

    # CNN: (N, L, F) -> (N, F, L)
    if cnn_model is not None:
        with torch.no_grad():
            xb = torch.from_numpy(seq.astype(np.float32)).permute(0,2,1)
            logits = cnn_model(xb).cpu().numpy().ravel()
        out["cnn"] = sigmoid_np(logits)

    return out

# --------------------------------------------------------------------
# Meta-Ensemble (falls vorhanden) / Equal-Weight-Avg
# --------------------------------------------------------------------
def build_meta_inputs_from_component_stream(
    comp: Dict[str, np.ndarray],
    regime_df: Optional[pd.DataFrame],
    L: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baut History H:[N,L,K] und Kontext C:[N,ctx] (nur Regime-Features).
    Rolling-Performance lassen wir (wie im Val-Pfad) weg.
    """
    keys = sorted(comp.keys())
    K = len(keys)
    N = len(next(iter(comp.values())))
    P = np.stack([comp[k] for k in keys], axis=1)  # [N,K]

    # History
    H = np.zeros((N, L, K), dtype=np.float32)
    for t in range(N):
        # Fenster der letzten L Zeitpunkte bis inkl. t
        s = max(0, t - L + 1)
        e = t + 1
        window = P[s:e]
        H[t, -len(window):, :] = window

    # Kontext (ATR, %B), falls gegeben
    if regime_df is not None and not regime_df.empty:
        reg = regime_df[["atr","bbp"]].to_numpy(dtype=np.float32)
        reg = reg[-N:]  # Align auf Länge N
        C = reg
    else:
        C = np.zeros((N, 2), dtype=np.float32)
    return H, C

def run_meta_or_average(
    comp: Dict[str, np.ndarray],
    meta: Optional[torch.nn.Module],
    regime_df: Optional[pd.DataFrame],
    L: int = 32
) -> np.ndarray:
    keys = sorted(comp.keys())
    if not keys:
        return np.array([])
    # Equal-weight Avg als Default
    eq_avg = np.mean(np.stack([comp[k] for k in keys], axis=1), axis=1)
    if meta is None:
        return eq_avg
    # Meta bauen & vorwärts
    H, C = build_meta_inputs_from_component_stream(comp, regime_df, L=L)
    with torch.no_grad():
        Ht = torch.from_numpy(H)
        Ct = torch.from_numpy(C)
        w, p_now = meta(Ht, Ct)        # [N,K], [N,K]
        P_base = Ht[:, -1, :]          # echte Basis-Preds der letzten Zeile
        alpha = 1.0                    # nur Basis (wie Val-Pfad)
        P_mix = alpha * P_base  (1.0 - alpha) * p_now
        p_hat = (w * P_mix).sum(dim=1).cpu().numpy().ravel()
    # safety
    p_hat = np.clip(np.nan_to_num(p_hat, nan=0.5), 1e-7, 1-1e-7)
    return p_hat

# --------------------------------------------------------------------
# Metriken: DA/MDA, PT-Test, Kalibrierung, Trading
# --------------------------------------------------------------------
def da_mda_metrics(trend_side: np.ndarray,
                   p_up: np.ndarray,
                   mda_band: float = 0.05) -> Dict[str, float]:
    """
    trend_side ∈ {-1,0,1}, p_up ∈ [0,1].
    """
    trend_side = trend_side.astype(int)
    p_up = p_up.astype(float)
    pred_bin = (p_up >= 0.5).astype(int)
    true_bin = (trend_side == 1).astype(int)

    out = {}
    out["da_all"] = float(accuracy_score(true_bin, pred_bin))

    act_mask = trend_side != 0
    if act_mask.any():
        out["da_active"] = float(accuracy_score(true_bin[act_mask], pred_bin[act_mask]))
    else:
        out["da_active"] = float("nan")

    # MDA: 3 Klassen via Band um 0.5
    pred_dir = np.where(p_up > 0.5 + mda_band, 1,
                 np.where(p_up < 0.5 - mda_band, -1, 0))
    out["mda_3class"] = float((pred_dir == trend_side).mean())
    return out

def pesaran_timmermann_da_z(
    trend_side: np.ndarray,
    p_up: np.ndarray
) -> Dict[str, float]:
    """
    DA-z-Test nach Anatolyev (2005, eq. 2.1-2.2): z = sqrt(T)*(A-B)/sqrt((1-mx^2)(1-my^2))
    mit x=Vorzeichen(Prognose), y=Vorzeichen(Realität); nur aktive Phasen {-1,1}.
    """
    # Prognose-Sign: 1 wenn p>=0.5, sonst -1; neutrals ignorieren
    y = trend_side.astype(int)
    mask = y != 0
    if mask.sum() < 5:
        return {"pt_z": float("nan"), "pt_p": float("nan")}
    y = y[mask]
    x = np.where(p_up[mask] >= 0.5, 1, -1)

    # Komponenten
    T = len(y)
    sx = x
    sy = y
    A = np.mean(sx * sy)
    mx = np.mean(sx)
    my = np.mean(sy)
    B = mx * my
    denom = math.sqrt(max(1e-12, (1 - mx**2) * (1 - my**2)))
    z = math.sqrt(T) * (A - B) / denom
    from math import erf, sqrt
    # two-sided p = 2*(1 - Phi(|z|)), Phi(z)=0.5*(1+erf(z/sqrt(2)))
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2))))
    # numerisch sauber einklammern
    p = max(0.0, min(1.0, p))
    return {"pt_z": float(z), "pt_p": float(p)}

def ece_score(y_true: np.ndarray, p_up: np.ndarray, n_bins: int = 10) -> Tuple[float, Dict]:
    """
    Expected Calibration Error (ECE) mit gleichbreiten Bins in [0,1].
    """
    y = y_true.astype(int)
    p = p_up.astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    # Bin-Index [0..n_bins-1]
    idx = np.digitize(p, bins, right=False) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    bins_out = []
    N = len(p)
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            bins_out.append({"bin": b, "count": 0, "conf": np.nan, "acc": np.nan})
            continue
        conf = float(np.mean(p[mask]))
        acc  = float(np.mean(y[mask]))
        w = np.mean(mask)
        ece += w * abs(acc - conf)
        bins_out.append({"bin": b, "count": int(mask.sum()), "conf": conf, "acc": acc})
    return float(ece), {"bins": bins_out}

def trading_metrics(
    prices: np.ndarray,            # Close/Price Serie (aligned mit p_up index  1 bar vor)
    p_up: np.ndarray,
    long_th: float = 0.6,
    short_th: float = 0.4,
    hold: int = 1
) -> Dict[str, float]:
    """
    Einfache Schwellen-Strategie: long wenn p>=long_th, short wenn p<=short_th, sonst flat.
    Haltedauer 'hold' Bars. Returns auf log-Returns (stabiler).
    """
    p = p_up
    signal = np.where(p >= long_th, 1, np.where(p <= short_th, -1, 0)).astype(int)
    # log-returns
    pr = np.asarray(prices, dtype=float)
    ret = np.diff(np.log(pr))
    # align: signal_t wirkt auf ret_{t} (nächste Bar nach Signal-Entstehung)
    sig = signal[:-hold] if hold == 1 else np.convolve(signal, np.ones(hold, dtype=int), "valid")
    rets_used = ret[:len(sig)]
    pnl = sig * rets_used

    if len(pnl) == 0:
        return {k: float("nan") for k in [
            "win_rate","profit_factor","pl_ratio","expectancy","avg_win","avg_loss","trades"
        ]}

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = float((pnl > 0).mean())
    sum_win = float(wins.sum()) if len(wins) else 0.0
    sum_loss = float(np.abs(losses.sum())) if len(losses) else 0.0
    profit_factor = float(sum_win / sum_loss) if sum_loss > 0 else float("inf")

    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(np.abs(losses.mean())) if len(losses) else 0.0
    loss_rate = 1.0 - win_rate
    expectancy = float(win_rate * avg_win - loss_rate * avg_loss)

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "pl_ratio": (avg_win / avg_loss) if avg_loss > 0 else float("inf"),
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "trades": int((signal != 0).sum())
    }

# --------------------------------------------------------------------
# Haupt-Evaluierung
# --------------------------------------------------------------------
def evaluate(
    model_dir: Path,
    test_file: Path,
    plots: bool = False,
    mda_band: float = 0.05,
    long_th: float = 0.6,
    short_th: float = 0.4,
    hold: int = 1
) -> Dict:
    cfg = load_yaml_cfg(model_dir)
    seq_len = int(cfg.get("training", {}).get("seq_len", 24))
    num_cols = cfg.get("data", {}).get("numerical_cols", ["open","high","low","Close","volume"])

    # Daten laden
    df = read_test_data(test_file)

    # Trend-Seite & Label
    side = make_trend_side_dc(df["price"],
                              dc_thres=float(cfg.get("dc_thres", 0.5)),
                              windows=tuple(cfg.get("w_list", [5,15,30])),
                              tau=int(cfg.get("tau", 1)))
    df["trend_side"] = side
    df["label"] = (side == 1).astype(int)

    # Regime-Features (optional)
    regime_df = None
    if bool(cfg.get("meta", {}).get("use_regime", True)):
        # extract_regime_features erwartet Spalten: high/low/close
        tmp = df.copy()
        if "close" not in tmp.columns:
            tmp["close"] = tmp["Close"] if "Close" in tmp.columns else tmp["price"]
        regime_df = extract_regime_features(tmp).reindex(df.index)

    # Feature-Matrix auf Trainingsspalten (falls vorhanden)
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
    feat = df[num_cols].dropna()
    # Sequenzen bilden
    X = feat.to_numpy(dtype=np.float32)
    if len(X) <= seq_len:
        raise ValueError("Zu wenig Daten für Sequenzen.")
    seq = np.stack([X[i-seq_len:i] for i in range(seq_len, len(X))])        # (N,L,F)
    # Targets und Trend auf gleiche Länge schneiden
    y_true = df["label"].to_numpy(dtype=int)[seq_len:]
    trend_side = df["trend_side"].to_numpy(dtype=int)[seq_len:]
    # Preise für Trading-Metrik (align zu p_up)
    price_aligned = df["price"].to_numpy(dtype=float)[seq_len-1:]  # damit ret_{t} zur p_{t} passt

    # Komponenten laden
    rf_list  = try_load_rf(model_dir)
    lgb_list = try_load_lgb(model_dir)
    xgb_list = try_load_xgb(model_dir)
    ft_model = try_load_ft(model_dir, n_features=seq.shape[1]*seq.shape[2])
    cnn_model= try_load_cnn(model_dir, n_feat_guess=seq.shape[2])

    # Temperature (nur FT)
    T = 1.0
    tfile = model_dir / "temp_scaler.pt"
    if tfile.exists():
        try:
            T = float(torch.load(tfile, map_location="cpu").get("temperature", 1.0))
        except Exception:
            T = 1.0

    comp = predict_components(seq, rf_list, lgb_list, xgb_list, ft_model, cnn_model, temperature_T=T)

    # Meta (nur wenn rekonstruierbar)
    ctx_dim = 2 if regime_df is not None else 0
    meta = try_load_meta(model_dir, K=len(comp), ctx_dim=ctx_dim) if len(comp) >= 1 else None

    # Ensemble-Vorhersage
    p_meta = run_meta_or_average(comp, meta, regime_df)
    # Metrik-Sammelobjekt
    metrics: Dict[str, float] = {}

    # ---------------- Metriken gesamt (Meta/Avg) ----------------
    def full_metric_block(p_up: np.ndarray, tag: str) -> Dict:
        out = {}
        # klassisch  Kalibrierung
        out[f"{tag}_accuracy"] = float(accuracy_score(y_true, (p_up>=0.5).astype(int)))
        try: out[f"{tag}_roc_auc"] = float(roc_auc_score(y_true, p_up))
        except Exception: out[f"{tag}_roc_auc"] = float("nan")
        try: out[f"{tag}_brier"] = float(brier_score_loss(y_true, p_up))
        except Exception: out[f"{tag}_brier"] = float("nan")
        ece, bins = ece_score(y_true, p_up, n_bins=10)
        out[f"{tag}_ece"] = float(ece)

        # DA/MDA  PT
        out.update({f"{tag}_{k}": v for k,v in da_mda_metrics(trend_side, p_up, mda_band=mda_band).items()})
        out.update({f"{tag}_{k}": v for k,v in pesaran_timmermann_da_z(trend_side, p_up).items()})

        # Trading
        out.update({f"{tag}_{k}": v for k,v in trading_metrics(price_aligned, p_up, long_th, short_th, hold).items()})
        return out

    # ====== ENSEMBLE (Meta) ======
    # Guard: nur auswerten, wenn Vorhersagen vorhanden und Längen passen
    def _len_ok(p):
        try:
            return (p is not None) and (len(p) == len(y_true)) and (len(p) > 0)
        except Exception:
            return False

    if _len_ok(p_meta):
        metrics.update(full_metric_block(p_meta, "ensemble"))
    else:
        print(f"⚠️  Skipping ensemble metrics: predictions len="
              f"{0 if p_meta is None else len(p_meta)} vs labels len={len(y_true)}")


    # ---------------- Komponenten-Report ----------------
    # ====== Basis-Modelle ======
    for name, preds in comp.items():
        if _len_ok(preds):
            metrics.update(full_metric_block(preds, f"only_{name}"))
        else:
            print(f"⚠️  Skipping component '{name}': len={0 if preds is None else len(preds)}")

    # ---------------- Ablation: leave-one-out (Equal-Weight) ---
        # ====== Ablationen (only-X / leave-one-out) ======
    if len(comp) >= 2:
        keys = sorted(comp.keys())
        stack = np.stack([comp[k] for k in keys], axis=1)   # [N,K]
        for i, k in enumerate(keys):
            p_loo = np.mean(np.delete(stack, i, axis=1), axis=1)
            metrics.update(full_metric_block(p_loo, f"loo_drop_{k}"))

    # ---------------- Plots (optional) -------------------
    out_dir = model_dir / "eval_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    if plots:
        import matplotlib.pyplot as plt
        from sklearn.calibration import CalibrationDisplay

    # ── Reliability / Calibration (Ensemble) – nur wenn Predictions verfügbar
    if plots:
        from sklearn.calibration import CalibrationDisplay
        if (p_meta is not None) and (len(p_meta) == len(y_true)) and (len(p_meta) > 0):
            plt.figure()
            CalibrationDisplay.from_predictions(y_true, p_meta, n_bins=10)
            plt.title("Reliability Diagram – Ensemble")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "reliability_ensemble.png"), dpi=120)
            plt.close()
        else:
            print(
                f"⚠️  Skipping calibration plot: "
                f"predictions len={(0 if p_meta is None else len(p_meta))} "
                f"vs labels len={len(y_true)}"
            )

        # ROC optional – wenn AUC berechenbar
        try:
            from sklearn.metrics import RocCurveDisplay
            fig = plt.figure(figsize=(4,4))
            RocCurveDisplay.from_predictions(y_true, p_meta)
            plt.title("ROC (Ensemble)")
            plt.tight_layout()
            plt.savefig(out_dir / "roc_ensemble.png")
            plt.close(fig)
        except Exception:
            pass

    # JSON raus
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Kurz-Print
    print("\n=== Ensemble (Equal-Weight oder Meta) ===")
    for k in ["ensemble_accuracy","ensemble_roc_auc","ensemble_brier","ensemble_ece",
              "ensemble_da_all","ensemble_da_active","ensemble_mda_3class",
              "ensemble_pt_z","ensemble_pt_p",
              "ensemble_win_rate","ensemble_profit_factor","ensemble_pl_ratio","ensemble_expectancy"]:
        if k in metrics:
            print(f"{k:>24s}: {metrics[k]:.6f}")

    return {"metrics": metrics, "out_dir": str(out_dir)}

# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate HybridLongTrend models (components  ensemble)")
    p.add_argument("--model-dir", required=True, help="Pfad zum Modellverzeichnis")
    p.add_argument("--test-file", required=True, help="CSV/TXT mit timestamp & price/Close ( optional OHLCV)")
    p.add_argument("--plots", action="store_true", help="Speichere Reliability/ROC Plots")
    p.add_argument("--mda-band", type=float, default=0.05, help="Bandbreite um 0.5 für MDA neutrale Klasse")
    p.add_argument("--long-thresh", type=float, default=0.6, help="Long-Schwelle p_up")
    p.add_argument("--short-thresh", type=float, default=0.4, help="Short-Schwelle p_up")
    p.add_argument("--hold", type=int, default=1, help="Haltedauer (Bars) der einfachen Strategie")
    return p.parse_args()

def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    test_file = Path(args.test_file)
    evaluate(
        model_dir=model_dir,
        test_file=test_file,
        plots=args.plots,
        mda_band=args.mda_band,
        long_th=float(args.long_thresh),
        short_th=float(args.short_thresh),
        hold=int(args.hold),
    )

if __name__ == "__main__":
    main()
