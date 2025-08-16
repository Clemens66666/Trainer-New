#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_v3.py
-------------
Robuster Streaming-Predictor für das Hybrid-Ensemble (RF/LGB/XGB/FT/CNN) + MetaMoE.

Änderungen ggü. v2:
- Temperature-Scaling für FT-Logits (lädt temp_scaler.pt und teilt Logits durch T)
- Regime-Features (ATR(14), Bollinger-%B(20)) werden (wenn vorhanden) in den Meta-Kontext C integriert
- Meta-Kontext vollständig und konsistent (alle Basis-Outputs + Zeitsinus/cos, Regime, ggf. Padding)
- Stabiler Fallback: Wenn Meta ~0.5-konstant wird, Rückfall auf Basis-Mittel
- Konsistente Trimming-Logik: Nach Fenstern Längenangleichung und nur getrimmte Arrays weiterverwenden
- Ausführliche Logs (Shapes, geladene Modelle, Meta-ctx_dim-Klärung, Temp-Scaling, etc.)

Nutzung (Beispiel):
python predict_v3.py \
  --ea-tick-file data/ticks.csv --freq 5min --seq-len 24 --label-horizon-min 5 \
  --model-dir models/exp_123 --output out/preds.csv --use-meta 1 --use-regime 1 \
  --precision auto --device auto --verbose

Outputs:
- CSV-Streaming in --output mit Spalten: Time, proba_final, proba_meta, proba_rf, proba_lgb, proba_xgb, proba_ft, proba_cnn, atr, bbp
"""

from __future__ import annotations
import argparse, os, sys, json, gc, math, time, contextlib, traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# ============================ TRAIN META / LOCKING =========================
from pathlib import Path
import json

# Feste Trainingsreihenfolge der Basismodelle (siehe Trainer-Stacking)  # rf,lgb,xgb,ft,cnn
TRAIN_BASE_ORDER = ["rf", "lgb", "xgb", "ft", "cnn"]  # entspricht dem OOF-Stack im Trainer :contentReference[oaicite:1]{index=1}

def _safe_read_json(p: Path) -> dict:
    try:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}

def _infer_tree_n_features(model_dir: Path) -> int | None:
    # Versuche RF zuerst (sklearn: n_features_in_), dann LGB/XGB (num_feature)
    import pickle
    try:
        rf_p = model_dir / "rf_list.pkl"
        if rf_p.exists():
            with open(rf_p, "rb") as f:
                rf_list = pickle.load(f)
            for m in (rf_list or []):
                n = getattr(m, "n_features_in_", None)
                if isinstance(n, int) and n > 0:
                    return n
    except Exception:
        pass
    try:
        lgb_p = model_dir / "lgb_list.pkl"
        if lgb_p.exists():
            with open(lgb_p, "rb") as f:
                lgb_list = pickle.load(f)
            for m in (lgb_list or []):
                booster = getattr(m, "booster_", None) or getattr(m, "booster", None)
                if booster is not None:
                    n = getattr(booster, "num_feature", None)
                    if isinstance(n, int) and n > 0:
                        return n
    except Exception:
        pass
    try:
        xgb_p = model_dir / "xgb_list.pkl"
        if xgb_p.exists():
            with open(xgb_p, "rb") as f:
                xgb_list = pickle.load(f)
            for m in (xgb_list or []):
                n = getattr(getattr(m, "get_booster", lambda: None)(), "num_features", None)
                if isinstance(n, int) and n > 0:
                    return n
    except Exception:
        pass
    return None

def _infer_cnn_in_chans(model_dir: Path, best_params: dict) -> int | None:
    # i) aus best_params.json
    cnn_bp = (best_params or {}).get("cnn", {}) or {}
    for k in ("in_chans", "n_in", "n_feat"):
        v = cnn_bp.get(k, None)
        if isinstance(v, int) and v > 0:
            return v
    # ii) aus Gewichten: erste Conv-Gewichte aus cnn.pt (shape [C_out, C_in, K])
    try:
        sd = torch.load(model_dir / "cnn.pt", map_location="cpu")
        if isinstance(sd, dict):
            for k, w in sd.items():
                if isinstance(w, torch.Tensor) and w.ndim == 3:
                    return int(w.shape[1])
    except Exception:
        pass
    return None

def load_train_signature(model_dir: Path, log) -> dict:
    """
    Lädt train_meta.json/feature_list.json, fällt auf best_params.json zurück
    und versucht seq_len robust aus (Tree.n_features_in / cnn_in) zu rekonstruieren.
    """
    sig = {}
    meta = {}
    for name in ("train_meta.json", "train_signature.json", "data_meta.json", "signature.json"):
        meta = _safe_read_json(model_dir / name)
        if meta:
            break
    best_params = _safe_read_json(model_dir / "best_params.json")

    # Basis aus train_meta (falls vorhanden)
    for k in ("freq", "seq_len", "label_horizon_min", "feature_list", "base_order"):
        if k in meta:
            sig[k] = meta[k]

    # feature_list.json überschreibt ggf. (expliziter Dump der Liste)
    feat_path = model_dir / "feature_list.json"
    if feat_path.exists():
        try:
            with open(feat_path, "r", encoding="utf-8") as f:
                sig["feature_list"] = json.load(f)
        except Exception:
            pass

    # seq_len notfalls rekonstruieren: n_features_tree = L * F_cnn
    if "seq_len" not in sig or not isinstance(sig["seq_len"], int):
        n_tree = _infer_tree_n_features(model_dir)
        n_cnn  = _infer_cnn_in_chans(model_dir, best_params)
        if isinstance(n_tree, int) and isinstance(n_cnn, int) and n_cnn > 0 and (n_tree % n_cnn == 0):
            sig["seq_len"] = int(n_tree // n_cnn)

    # Fallback: Base-Order aus Trainer-Konvention
    if "base_order" not in sig:
        sig["base_order"] = TRAIN_BASE_ORDER

    return sig

def enforce_and_lock_params(args, sig: dict, log):
    """
    Erzwingt die Trainings-Settings. Abweichungen sind *verboten*.
    """
    mismatches = []

    def _cmp(name, val):
        if val is None: 
            return
        curr = getattr(args, name, None)
        if curr is None:
            setattr(args, name, val)
            return
        # Zahlen konsistent casten
        if name in ("seq_len", "label_horizon_min"):
            curr = int(curr)
            val  = int(val)
        if str(curr) != str(val):
            mismatches.append(f"{name}={curr} (arg) ≠ {val} (train)")

        # Auf Trainingswert setzen, damit downstream *immer* korrekt ist
        setattr(args, name, val)

    _cmp("freq",               sig.get("freq", None))
    _cmp("seq_len",            sig.get("seq_len", None))
    _cmp("label_horizon_min",  sig.get("label_horizon_min", None))

    if mismatches:
        msg = "[PARAMETER-MISMATCH] " + ", ".join(mismatches) + \
              " — diese Parameter sind im Training fixiert und dürfen bei der Inferenz nicht abweichen."
        log.error(msg)
        raise SystemExit(msg)

def override_feature_order_from_signature(model_dir: Path, sig: dict, log):
    """
    Überschreibt die *globale* CNN-Featureliste in diesem Modul, falls persistiert.
    Dadurch erhalten CNN *und* Trees (Flatten) exakt die Trainingskanäle/-reihenfolge.
    """
    global FEATURE_ORDER_CNN
    fl = sig.get("feature_list", None)

    # separate Datei hat Priorität (falls vorhanden)
    feat_path = model_dir / "feature_list.json"
    if feat_path.exists():
        try:
            with open(feat_path, "r", encoding="utf-8") as f:
                fl = json.load(f)
        except Exception:
            pass

    if isinstance(fl, list) and len(fl) > 0:
        FEATURE_ORDER_CNN = list(fl)
        log.info(f"[Features] geladene Trainings-Featureliste: {FEATURE_ORDER_CNN} (len={len(FEATURE_ORDER_CNN)})")
    else:
        log.warning("[Features] Keine persistierte Featureliste gefunden → benutze Modul-Default. (Bitte im Trainer persistieren)")

# ============================ utils: device & precision ============================
def pick_device(device_arg="auto", log=None):
    import torch
    arg = str(device_arg or "auto").lower()
    if arg == "cpu":
        dev = "cpu"
    elif arg.startswith("cuda") or arg in ("gpu", "auto"):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    if log is not None:
        try:
            if dev == "cuda":
                try:
                    free, total = torch.cuda.mem_get_info()
                    free_gb = free / (1024**3)
                    total_gb = total / (1024**3)
                    log.info(f"device={device_arg} → cuda | free={free_gb:.1f}GB / total={total_gb:.1f}GB")
                except Exception:
                    log.info(f"device={device_arg} → cuda")
            else:
                log.info(f"device={device_arg} → cpu")
        except Exception:
            pass
    return dev


def resolve_amp_dtype(precision, device: str):
    import torch
    p = str(precision or "auto").lower()

    if p in ("none", "fp32", "float32"):
        return torch.float32

    if device.startswith("cuda"):
        if p in ("bf16", "bfloat16"):
            return torch.bfloat16
        if p in ("fp16", "float16"):
            return torch.float16
        # auto
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        try:
            cc = torch.cuda.get_device_capability()
            if isinstance(cc, (tuple, list)) and len(cc) >= 1 and cc[0] >= 8:  # Ampere+
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16  # broadest support
    # CPU: float32 (es gibt kein echtes AMP auf CPU)
    return torch.float32


def freq_to_minutes(freq: str) -> int:
    s = str(freq or "").strip().lower()
    # z. B. "5min", "5m"
    if s.endswith("min"):
        n = s[:-3].strip()
        return max(1, int(n)) if n.isdigit() else 1
    if s.endswith("m"):
        n = s[:-1].strip()
        return max(1, int(n)) if n.isdigit() else 1
    # z. B. "1h", "2h"
    if s.endswith("h"):
        n = s[:-1].strip()
        return max(1, int(n)) * 60 if n.isdigit() else 60
    # Tage als Fallback
    if s in ("d", "1d", "day", "daily"):
        return 1440
    # Default
    return 5

# --- NEW: load cnn normalization stats ---
def load_cnn_norm(model_dir: str, expected_features: Sequence[str], log=None):
    """
    Lädt Normalisierungs-Stats für den CNN-Pfad.
    Unterstützt beide Varianten:
      - {"feature_list":[...], "mu":[...], "sd":[...]}  (dein aktuelles File)
      - {"feature_list":[...], "mean":[...], "std":[...]} (ältere Variante)
    Gibt Dict {"mean": np.ndarray(F,), "std": np.ndarray(F,)} zurück oder None.
    """
    import json
    from pathlib import Path

    p = Path(model_dir) / "cnn_norm.json"
    if not p.exists():
        if log: log.info("[CNN][NORM] keine cnn_norm.json gefunden → ohne Norm.")
        return None

    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)

        feats = d.get("feature_list", None)
        if not isinstance(feats, list) or len(feats) == 0:
            raise ValueError("cnn_norm.json fehlt 'feature_list'.")

        # akzeptiere mu/sd oder mean/std
        mean = d.get("mu", d.get("mean", None))
        std  = d.get("sd", d.get("std", None))
        if mean is None or std is None:
            raise ValueError("cnn_norm.json braucht 'mu/sd' oder 'mean/std'.")

        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        std  = np.asarray(std,  dtype=np.float32).reshape(-1)

        if len(mean) != len(feats) or len(std) != len(feats):
            raise ValueError("cnn_norm.json: Längen von Stats und feature_list passen nicht.")

        # Reihenfolge prüfen – muss exakt dem Inferenz-Feature-Order entsprechen
        if list(expected_features) != list(feats):
            raise ValueError("cnn_norm.json: feature_list passt nicht zur Inferenz-Feature-Reihenfolge.")

        # numerisch robust (keine Division durch 0)
        std = np.where(std <= 0, 1.0, std).astype(np.float32)

        if log:
            log.info(f"[CNN][NORM] geladen: F={len(feats)} (mean/std)")
        return {"mean": mean, "std": std}

    except Exception as e:
        if log: log.warning(f"[CNN][NORM] laden fehlgeschlagen: {e} → ohne Norm weiter.")
        return None

# ============================ LOGGING ======================================
def get_logger(name: str, verbose: bool = False):
    import logging, sys, datetime
    class _Formatter(logging.Formatter):
        def format(self, record):
            t = datetime.datetime.now().strftime("%H:%M:%S")
            lvl = f"{record.levelname:<7}"
            return f"{t} | {lvl} | {record.name} | {record.getMessage()}"
    logger = logging.getLogger(name)
    level = logging.DEBUG if bool(verbose) else logging.INFO
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(level)
        h.setFormatter(_Formatter())
        logger.addHandler(h)
        logger.propagate = False
    else:
        for h in logger.handlers:
            h.setLevel(level)
    return logger


# ============================ GPU / AMP HELPERS ============================
def _bytes2human(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    if n <= 0 or not isinstance(n, (int, float)):
        return "0.0B"
    i = min(int(math.log(n, 1024)), len(units) - 1)
    p = 1024 ** i
    v = n / p
    return f"{v:.1f}{units[i]}"

def _gpu_mem(metric: str = "used") -> str:
    try:
        if not torch.cuda.is_available():
            return "0.0B"
        idx = 0
        used     = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        total    = torch.cuda.get_device_properties(idx).total_memory
        free     = max(total - reserved, 0)
        vals = {"used": used, "reserved": reserved, "free": free, "total": total}
        return _bytes2human(vals.get(metric, used))
    except Exception:
        return "0.0B"

def amp_dtype_from_arg(arg: str) -> Optional[torch.dtype]:
    m = (arg or "auto").lower()
    if m == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return None
    if m in ("fp32","float32","off","none"):
        return None
    if m in ("fp16","float16"):
        return torch.float16
    if m in ("bf16","bfloat16"):
        return torch.bfloat16
    return None

# ============================ IMPORTS TRAINER ==============================
# Wir versuchen, deine Trainer-Klassen zu importieren; ansonsten Fallback-Klassen.
try:
    from trainers.hybrid_longtrend_trainer import FTWrapped
except Exception:
    FTWrapped = None

try:
    from trainers.hybrid_longtrend_trainer import SimpleCNN as TrainerCNN
except Exception:
    TrainerCNN = None

# Meta-Klassensuche
MetaClassCandidates = []
try:
    from trainers.hybrid_longtrend_trainer import MetaMoE as _MetaMoE
    MetaClassCandidates.append(("trainer.MetaMoE", _MetaMoE))
except Exception:
    pass
try:
    from trainers.hybrid_longtrend_trainer import MetaTransformer as _MetaTr
    MetaClassCandidates.append(("trainer.MetaTransformer", _MetaTr))
except Exception:
    pass
try:
    from meta_transformer import MetaMoE as _MetaMoE2
    MetaClassCandidates.append(("meta_transformer.MetaMoE", _MetaMoE2))
except Exception:
    pass



# ============================ CNN-FEATURES =================================
FEATURE_ORDER_CNN = [
    "open", "high", "low", "close", "volume",
    "sma_10", "ema_20", "rsi_14",
    "hour_sin", "hour_cos",
]

def _add_time_feats(df_ohlc: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df_ohlc.index)
    hour = idx.hour.values.astype(np.float32)
    df_ohlc["hour_sin"] = np.sin(2*np.pi*hour/24.0).astype(np.float32)
    df_ohlc["hour_cos"] = np.cos(2*np.pi*hour/24.0).astype(np.float32)
    return df_ohlc

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = (roll_up / (roll_down + 1e-12)).fillna(0.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return (rsi / 100.0).astype(np.float32)

def enrich_for_cnn(ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    Erwartet OHLCV mit Index=Datetime und Spalten: open, high, low, close, volume
    Liefert DataFrame mit exakt FEATURE_ORDER_CNN in dieser Reihenfolge.
    """
    df = ohlc.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df["sma_10"] = df["close"].rolling(10, min_periods=1).mean().astype(np.float32)
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean().astype(np.float32)
    df["rsi_14"] = _rsi(df["close"], 14)
    df = _add_time_feats(df)
    df = df[FEATURE_ORDER_CNN].astype(np.float32)
    return df

def make_windows_from_features(df_feat: pd.DataFrame, seq_len: int, horizon_bars: int):
    """
    df_feat: index=Datetime, Spalten=Features (F)
    Return: X_seq [M,L,F], X_last [M,F], X_flat [M,L*F], emit_index [M]
    """
    if df_feat.empty:
        return np.empty((0, seq_len, 0), np.float32), np.empty((0, 0), np.float32), np.empty((0, 0), np.float32), df_feat.index[:0]

    arr = df_feat.to_numpy(dtype=np.float32, copy=False)  # [T,F]
    T, F = arr.shape
    L = int(seq_len)
    H = max(1, int(horizon_bars))

    if T < L + H:
        return np.empty((0, L, F), np.float32), np.empty((0, F), np.float32), np.empty((0, L*F), np.float32), df_feat.index[:0]

    ends = np.arange(L-1, T-H, dtype=np.int64)  # End-Index jedes Fensters
    M = len(ends)
    X_seq = np.empty((M, L, F), dtype=np.float32)
    for i, e in enumerate(ends):
        s = e - (L - 1)
        X_seq[i] = arr[s:e+1]

    X_last = X_seq[:, -1, :].copy()
    X_flat = X_seq.reshape(M, L*F)
    emit_index = df_feat.index[ends]
    return X_seq, X_last, X_flat, emit_index

# ==== COMPAT SHIMS (predict_v3) ============================================
# Diese Helfer schließen Lücken bei älteren Aufrufstellen in stream_predict.
# Sie werden nur definiert, wenn die Namen noch nicht existieren.
from pathlib import Path
import os
import numpy as np
import torch


# ============================ MODEL LOADER =================================
try:
    import joblib
except Exception:
    joblib = None

def _load_pickle_list(path: Path, name: str, log):
    if path.exists() and joblib is not None:
        try:
            obj = joblib.load(path)
            log.info(f"[{name}] {len(obj)} Modelle geladen.")
            return obj
        except Exception as e:
            log.warning(f"[{name}] Laden fehlgeschlagen ({path.name}): {e}")
    else:
        log.info(f"[{name}] keine Datei: {path.name}")
    return []

def load_rf_models(model_dir: Path, log):
    return _load_pickle_list(model_dir / "rf_list.pkl", "RF", log)

def load_lgb_models(model_dir: Path, log):
    return _load_pickle_list(model_dir / "lgb_list.pkl", "LGB", log)

def load_xgb_models(model_dir: Path, log):
    return _load_pickle_list(model_dir / "xgb_list.pkl", "XGB", log)

def _expected_features(model) -> Optional[int]:
    # sklearn
    if hasattr(model, "n_features_in_"):
        try: return int(model.n_features_in_)
        except Exception: pass
    # lightgbm booster
    try:
        import lightgbm as lgb  # noqa
        if isinstance(model, lgb.Booster):
            return int(model.num_feature())
    except Exception:
        pass
    # xgboost booster → oft nicht verfügbar
    try:
        import xgboost as xgb  # noqa
        if isinstance(model, xgb.Booster):
            # num_features() ist nicht zuverlässig verfügbar → None
            return None
    except Exception:
        pass
    return None

def _prepare_X_for_model(model, X_last: Optional[np.ndarray], X_flat: Optional[np.ndarray]) -> np.ndarray:
    exp = _expected_features(model)
    cand = [arr for arr in (X_flat, X_last) if arr is not None]  # bevorzugt flattened
    if not cand:
        raise RuntimeError("keine Eingaben für Tree-Modelle verfügbar")

    base = cand[0]
    if exp is None:
        return base

    # exakte Übereinstimmung?
    for arr in cand:
        if arr.shape[1] == exp:
            return arr

    # sonst pad/trim
    if base.shape[1] < exp:
        pad = np.zeros((base.shape[0], exp - base.shape[1]), dtype=base.dtype)
        return np.concatenate([base, pad], axis=1)
    else:
        return base[:, :exp]

# ============================ TREE-PROBAS (robust) =========================
from typing import List, Any, Optional

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _to_proba_01(y) -> np.ndarray:
    """bringt beliebige Scores stabil in [0,1]"""
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    # NaN/Inf abfangen
    arr = np.where(np.isfinite(arr), arr, 0.5)
    # Falls nicht schon [0,1], über Sigmoid
    if arr.min() < 0.0 or arr.max() > 1.0:
        arr = _sigmoid_np(arr)
    # hart clippen
    arr = np.clip(arr, 1e-7, 1.0 - 1e-7).astype(np.float32, copy=False)
    return arr

def _predict_trees(model_list: List[Any],
                   X_last: Optional[np.ndarray],
                   X_flat: Optional[np.ndarray],
                   name: str,
                   log) -> Optional[np.ndarray]:
    """
    Vereinheitlichte Proba-Vorhersage über eine Liste von RF/LGB/XGB/Sklearn-Modellen.
    Wählt automatisch passende Eingabe (X_flat bevorzugt), macht Pad/Trim und mittelt.
    """
    if not model_list:
        return None

    preds, any_success = [], False
    for m in model_list:
        try:
            Xin = _prepare_X_for_model(m, X_last, X_flat)

            # --- LightGBM Booster ---
            try:
                import lightgbm as lgb
                if isinstance(m, lgb.Booster):
                    y = m.predict(
                        Xin,
                        num_iteration=(getattr(m, "best_iteration", None) or None)
                    )
                    preds.append(_to_proba_01(y)); any_success = True; continue
            except Exception:
                pass

            # --- XGBoost Booster ---
            try:
                import xgboost as xgb
                if isinstance(m, xgb.Booster):
                    dm = xgb.DMatrix(Xin)
                    best_it = getattr(m, "best_iteration", None)
                    if best_it is not None:
                        y = m.predict(dm, iteration_range=(0, best_it + 1))
                    else:
                        y = m.predict(dm)
                    preds.append(_to_proba_01(y)); any_success = True; continue
            except Exception:
                pass

            # --- Sklearn-ähnliche Wrapper (inkl. xgb.sklearn, lgb.sklearn, RF etc.) ---
            if hasattr(m, "predict_proba"):
                proba = m.predict_proba(Xin)
                if isinstance(proba, (list, tuple)):
                    proba = np.asarray(proba)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    y = proba[:, 1]
                else:
                    y = proba.reshape(-1)
                preds.append(_to_proba_01(y)); any_success = True
            elif hasattr(m, "decision_function"):
                df = m.decision_function(Xin)
                preds.append(_to_proba_01(df)); any_success = True
            elif hasattr(m, "predict"):
                y = m.predict(Xin)
                preds.append(_to_proba_01(y)); any_success = True
            else:
                raise RuntimeError("Kein passender Vorhersage-Endpunkt gefunden")
        except Exception as ex:
            log.warning(f"[{name}] Vorhersage fehlgeschlagen: {ex}")

    if not any_success:
        return None

    # Modelle mitteln → (M,)
    return np.mean(np.stack(preds, axis=1), axis=1).astype(np.float32)

# --- Öffentliche Wrapper (Signaturen wie in stream_predict verwendet) -----
def predict_rf_batched(rf_models, X_flat_feat: np.ndarray, batch_size: int, log):
    return _predict_trees(rf_models, None, X_flat_feat, "RF", log) if rf_models else None

def predict_lgb_batched(lgb_models, X_flat_feat: np.ndarray, batch_size: int, log):
    return _predict_trees(lgb_models, None, X_flat_feat, "LGB", log) if lgb_models else None

def predict_xgb_batched(xgb_models, X_flat_feat: np.ndarray, batch_size: int, log):
    return _predict_trees(xgb_models, None, X_flat_feat, "XGB", log) if xgb_models else None
# ========================================================================== 

# ============================ FT (Lazy) + TempScaling ======================
# Sauberer, einmaliger FT-Block: kompatibel mit alten & neuen rtdl-Signaturen.

from pathlib import Path
import os, json
import numpy as np
import torch
import torch.nn as nn

# ---- Helpers ---------------------------------------------------------------

def _safe_get_ft_hp(model_dir, log=None):
    """Liest optionale Hyperparameter aus best_params.json (Schlüssel: ft oder flach)."""
    hp = {}
    bp = os.path.join(model_dir, "best_params.json")
    if os.path.isfile(bp):
        try:
            with open(bp, "r", encoding="utf-8") as f:
                data = json.load(f)
            hp = data.get("ft", data) or {}
            if log:
                pr = {k: hp.get(k) for k in ("n_blocks", "d_token", "n_heads", "dropout")}
                log.info(f"[FT][HP] aus best_params.json: {pr}")
        except Exception as e:
            if log: log.warning(f"[FT][HP] Konnte best_params.json nicht lesen: {e}")
    return hp

def _canonicalize_ft_state_dict(sd: dict, log=None) -> dict:
    """
    Mappt alte FT-Checkpoint-Keys ('transformer.*', 'feature_tokenizer.*', 'cls_token.*')
    auf das erwartete Schema 'ft.base_model.model.*', falls nötig.
    """
    if not isinstance(sd, dict):
        return sd
    has_expected = any(str(k).startswith("ft.base_model.model.") for k in sd.keys())
    has_old = any(str(k).startswith(("transformer.", "feature_tokenizer.", "cls_token.")) for k in sd.keys())
    if has_expected or not has_old:
        return sd
    new_sd = {}
    for k, v in sd.items():
        ks = str(k)
        if ks.startswith(("transformer.", "feature_tokenizer.", "cls_token.")):
            new_sd["ft.base_model.model." + ks] = v
        else:
            new_sd[ks] = v
    if log: log.info("[FT] Checkpoint-Keys kanonisiert → 'ft.base_model.model.*'")
    return new_sd

# ---- FT (Lazy) -------------------------------------------------------------
class _LazyFT(nn.Module):
    """
    Baut das FT-Transformer-Netz erst beim ersten Forward, wenn D (= #Features) feststeht.
    Gibt LOGITS zurück (ohne Sigmoid). Temperature-Scaling wird im Forward angewendet.
    """
    def __init__(self, model_dir: Path | str, state: dict, device: torch.device,
                 temp_T: float | None, log):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.state_raw = state or {}
        self.device    = device
        self.temp_T    = float(temp_T) if temp_T not in (None, 0) else None
        self.log       = log
        self.hp        = _safe_get_ft_hp(str(model_dir), log)
        self.net: nn.Module | None = None
        self._D: int | None = None

    def _build(self, D: int):
        if self.net is not None and self._D == D:
            return
        try:
            import rtdl
        except Exception as e:
            raise RuntimeError("rtdl muss installiert sein (für FTTransformer).") from e

        # HP (mit Defaults)
        n_blocks = int(self.hp.get("n_blocks", 3))
        d_token  = int(self.hp.get("d_token", 64))
        n_heads  = int(self.hp.get("n_heads", 8))
        dropout  = float(self.hp.get("dropout", 0.10))

        if self.log:
            self.log.info(f"[FT] Build (D={D}, n_blocks={n_blocks}, d_token={d_token}, "
                          f"n_heads={n_heads}, dropout={dropout})")

        # Zwei mögliche API-Signaturen von rtdl unterstützen
        try:
            # ältere/klassische Signatur
            net = rtdl.FTTransformer.make_default(
                n_num_features=D,           # <- alte API
                cat_cardinalities=(),       # keine Kategorien
                d_out=1,
                n_blocks=n_blocks
            )
        except TypeError:
            # neuere Signatur
            net = rtdl.FTTransformer.make_default(
                d_numerical=D,
                n_categories=None,
                d_token=d_token,
                n_layers=n_blocks,
                attention_n_heads=n_heads,
                d_out=1,
                dropout=dropout,
            )
        net = net.to(self.device)

        # State-Dict robust laden (alte Schlüssel ummappen)
        sd = _canonicalize_ft_state_dict(self.state_raw, self.log)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if self.log:
            self.log.info(f"[FT] state_dict geladen | missing={len(missing)} unexpected={len(unexpected)}")

        self.net = net.eval()
        self._D  = D

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: FloatTensor [B, D] → logits [B] (Temp-Scaling im Logitraum)
        Unterstützt rtdl-Modelle mit forward(x) oder forward(x_num, x_cat).
        """
        D = int(x.shape[1])
        if self.net is None or self._D != D:
            self._build(D)

        # rtdl erwartet je nach Version: net(x) ODER net(x_num, x_cat)
        try:
            y = self.net(x)  # [B, 1] oder [B]
        except TypeError:
            try:
                # Manche Implementationen akzeptieren None als x_cat
                y = self.net(x, None)
            except TypeError:
                # Fallback: explizit leeres kategorisches Tensor-Argument
                B = x.size(0)
                x_cat = torch.empty(B, 0, dtype=torch.long, device=x.device)
                y = self.net(x, x_cat)

        # auf 1D bringen
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = y.view(-1)

        if self.temp_T:
            y = y / self.temp_T
        return y

# ---- Loader + Batched Predict ---------------------------------------------

def load_ft(model_dir, device, log):
    """
    Lädt ft.pt (+ optional temp_scaler.pt) und gibt einen _LazyFT(nn.Module) zurück.
    """
    ft_path = os.path.join(model_dir, "ft.pt")
    if not os.path.isfile(ft_path):
        if log: log.warning("[FT] Kein ft.pt gefunden → FT deaktiviert.")
        return None

    st = torch.load(ft_path, map_location="cpu")
    if isinstance(st, nn.Module):
        # seltener Fall: komplettes Modell gespeichert
        if log: log.info("[FT] komplettes Modellobjekt geladen (kein Lazy-Build nötig).")
        return st.to(device).eval()

    temp_T = None
    ts_path = os.path.join(model_dir, "temp_scaler.pt")
    if os.path.isfile(ts_path):
        try:
            obj = torch.load(ts_path, map_location="cpu")
            if isinstance(obj, dict) and "T" in obj:
                temp_T = float(obj["T"])
            elif hasattr(obj, "T"):
                temp_T = float(obj.T)
            if log and temp_T:
                log.info(f"[FT] Temperature-Scaling aktiv: T={temp_T:.4f}")
        except Exception as e:
            if log: log.warning(f"[FT] temp_scaler.pt konnte nicht geladen werden: {e}")

    return _LazyFT(model_dir=model_dir, state=st, device=device, temp_T=temp_T, log=log)

@torch.no_grad()
def predict_ft_batched(ft_model: nn.Module, X_num: np.ndarray, device, batch_size: int,
                       temp_T: float | None, amp_dtype, log):
    """
    Führt die FT-Inferenz in Batches durch.
    - ft_model: _LazyFT oder beliebiges nn.Module, das Logits liefert
    - X_num: (N, D) float32
    - temp_T: optionaler zusätzlicher Temp-Override; wenn gesetzt, wirkt er zusätzlich
    """
    if ft_model is None or X_num is None or len(X_num) == 0:
        return None

    N = int(len(X_num))
    out = np.empty((N,), dtype=np.float32)
    use_amp = (amp_dtype is not None)

    i0 = 0
    while i0 < N:
        i1 = min(N, i0 + int(batch_size or 1024))
        xb = torch.as_tensor(X_num[i0:i1], dtype=torch.float32, device=device)
        with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available(), dtype=amp_dtype):
            logits = ft_model(xb)           # [b]
            if temp_T not in (None, 0):
                logits = logits / float(temp_T)
            probs = torch.sigmoid(logits).float().view(-1)
        out[i0:i1] = probs.detach().cpu().numpy()
        i0 = i1

    return out

# ============================ CNN – Loader & Predict =======================
class SimpleCNN(nn.Module):
    """Conv1d-Stack für Sequenzen; erwartet Input [B, C(=F), L]. Gibt LOGITS zurück."""
    def __init__(self, in_chans: int, n_filters: int = 64, kernel_size: int = 3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_chans, n_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_filters),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(n_filters, 1)

    def forward(self, x):  # [B, C, L]
        h = self.block1(x)
        h = self.block2(h).squeeze(-1)  # [B, n_filters]
        return self.head(h).squeeze(-1) # LOGITS [B]


# ============================ HELPERS (safe I/O) ============================

def _read_best_params_json(model_dir: Path, log):
    bp = (model_dir / "best_params.json")
    if not bp.exists():
        return {}
    try:
        import json
        d = json.loads(bp.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception as e:
        log.warning(f"[HP] best_params.json konnte nicht gelesen werden: {e}")
        return {}

def load_temp_scaler(model_dir: Path) -> Optional[float]:
    """Liest temp_scaler.pt (float T oder dict{'temperature': T})."""
    p = model_dir / "temp_scaler.pt"
    if not p.exists():
        return None
    try:
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, (int, float)):
            return float(obj)
        if isinstance(obj, dict):
            for k in ("T", "temperature", "temp", "scale"):
                if k in obj:
                    return float(obj[k])
    except Exception:
        pass
    return None

# ============================ CNN LOADING ===================================

def _infer_cnn_kernels_from_state(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """Versucht k1/k2 aus Gewichtsformen zu erraten (fallback 3/5)."""
    k1, k2 = 3, 5
    for k, w in sd.items():
        if not isinstance(w, torch.Tensor) or w.ndim != 3:
            continue
        if k.endswith("block1.0.weight"):
            k1 = int(w.shape[-1])
        elif k.endswith("block2.0.weight"):
            k2 = int(w.shape[-1])
    return k1, k2

def load_cnn_model(model_dir: Path, F_feat: int, device: torch.device, log) -> Optional[nn.Module]:
    """
    Baut das CNN so, dass es 1:1 zur Checkpoint-Architektur passt:
    - nimmt TrainerCNN, falls vorhanden
    - sonst eine kompatible Fallback-Class mit separaten k1/k2
    """
    sd_path = model_dir / "cnn.pt"
    if not sd_path.exists():
        log.info("[CNN] Kein cnn.pt gefunden → überspringe CNN.")
        return None

    sd = torch.load(sd_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # Hyperparameter aus best_params.json (falls vorhanden)
    hp_all = _read_best_params_json(model_dir, log) or {}
    hp_cnn = hp_all.get("cnn", {}) if isinstance(hp_all.get("cnn", {}), dict) else {}

    # n_filters aus HP oder aus StateDict ableiten
    n_filters = int(hp_cnn.get("n_filters")) if "n_filters" in hp_cnn else None
    if n_filters is None:
        # aus erster Conv ableiten
        for k, w in sd.items():
            if k.endswith("block1.0.weight") and isinstance(w, torch.Tensor) and w.ndim == 3:
                n_filters = int(w.shape[0])
                break
        if n_filters is None:
            n_filters = 32

    # Kernelgrößen bestimmen (HP > SD > Default)
    k1 = int(hp_cnn.get("k1")) if "k1" in hp_cnn else None
    k2 = int(hp_cnn.get("k2")) if "k2" in hp_cnn else None
    if k1 is None or k2 is None:
        _k1, _k2 = _infer_cnn_kernels_from_state(sd)
        k1 = k1 if k1 is not None else _k1
        k2 = k2 if k2 is not None else _k2

    dropout = float(hp_cnn.get("dropout", 0.0))
    dil2    = int(hp_cnn.get("dil2",    2))

    # Model instanzieren – bevorzugt das Trainer-Modell
    ModelClass = None
    if 'TrainerCNN' in globals() and (TrainerCNN is not None):
        ModelClass = TrainerCNN
        try:
            model = ModelClass(
                n_feat=F_feat, n_filters=n_filters, dropout=dropout,
                k1=k1, k2=k2, dil2=dil2
            )
        except TypeError:
            # Falls alte Signatur: fallback auf minimalen Aufruf
            model = ModelClass(F_feat, n_filters)
    else:
        # Lokaler Fallback mit k1/k2, damit die Gewichte passen
        class _SimpleCNNLike(nn.Module):
            def __init__(self, n_feat: int, n_filters: int, dropout: float, k1: int, k2: int, dil2: int):
                super().__init__()
                pad1 = (k1 - 1) // 2
                pad2 = (dil2 * (k2 - 1)) // 2
                self.block1 = nn.Sequential(
                    nn.Conv1d(n_feat, n_filters, kernel_size=k1, padding=pad1, bias=False),
                    nn.BatchNorm1d(n_filters),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                )
                self.block2 = nn.Sequential(
                    nn.Conv1d(n_filters, n_filters, kernel_size=k2, padding=pad2, dilation=dil2, bias=False),
                    nn.BatchNorm1d(n_filters),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                )
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(n_filters, 1),
                )
            def forward(self, x):  # x: [B, C, L]
                x1 = self.block1(x)
                y  = self.block2(x1)
                # defensiv: bei Längenabweichung residualfähig machen
                if y.size(-1) != x1.size(-1):
                    diff = x1.size(-1) - y.size(-1)
                    if diff > 0:
                        left = diff // 2
                        right = diff - left
                        y = nn.functional.pad(y, (left, right))
                    else:
                        y = y[..., :x1.size(-1)]
                z = x1 + y
                return self.head(z).squeeze(-1)  # logits

        model = _SimpleCNNLike(F_feat, n_filters, dropout, k1, k2, dil2)

    model = model.to(device)
    # Jetzt strikt laden – Keys/Shapes MÜSSEN passen, sonst Exception → wir haben k1/k2 already aligned
    model.load_state_dict(sd, strict=True)
    model.eval()

    log.info(f"[CNN][LOAD] Geladen aus 'cnn.pt' | n_filters={n_filters} | k1={k1}, k2={k2}, dil2={dil2} | expected_in={F_feat}")
    return model

def build_and_load_cnn(model_dir: Path, F_feat: int, device: torch.device, log):
    """Convenience-Wrapper – behält Signatur aus deinem main."""
    return load_cnn_model(model_dir, F_feat, device, log)

@torch.no_grad()
def predict_cnn_batched(cnn_model, X_seq_feat, device, batch_size, amp_dtype, log, cnn_norm=None):
    """
    X_seq_feat: np.ndarray [N, L, F] (F=10)
    gibt: np.ndarray [N] mit Probabilitäten
    """
    if cnn_model is None:  # robust
        return None

    cnn_model.eval()
    outs = []
    N = X_seq_feat.shape[0]
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = X_seq_feat[i:i+batch_size]  # [b, L, F]

            # --- NEW: optional standardisieren pro Feature (entsprechend Training) ---
            if cnn_norm is not None:
                mu = cnn_norm["mean"]   # [F]
                sd = cnn_norm["std"]    # [F]
                # numerisch robust
                sd_safe = np.where(sd==0.0, 1.0, sd)
                xb = (xb - mu.reshape(1,1,-1)) / sd_safe.reshape(1,1,-1)

            # CNN erwartet [b, C, L] → transpose
            xb_t = np.transpose(xb, (0, 2, 1)).astype(np.float32)
            xb_t = torch.from_numpy(xb_t).to(device=device, dtype=amp_dtype or torch.float32)

            with torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), dtype=amp_dtype) if amp_dtype else torch.no_grad():
                logits = cnn_model(xb_t)               # [b] oder [b,1]
                if logits.ndim > 1: logits = logits.squeeze(-1)
                pb = torch.sigmoid(logits).float()     # in (0,1)

            outs.append(pb.detach().cpu().numpy())

    return np.concatenate(outs, axis=0) if outs else None

# ============================ META INPUTS (H/C) =============================
# ============================ META INPUTS (H/C) =============================
def build_meta_inputs_from_base_preserve_H(
    base: Dict[str, Optional[np.ndarray]],   # {"rf": (M,), "lgb": (M,), ...}
    L: int,                                  # Sequenzlänge für Meta
    prev_tail: Optional[np.ndarray],         # (<= L-1, K) oder None
    emit_index,                              # DatetimeIndex/array/liste mit Länge ~ M
    target_ctx_dim: Optional[int],           # gewünschte ctx_dim oder None
    regime_dict: Optional[Dict[str, np.ndarray]] = None,  # {"atr":(M,), "bbp":(M,)}
    log=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Baut:
      H: (M, L, K) – Sliding-Window der Basis-Outputs (inkl. prev_tail),
      C: (M, ctx_dim) – 7D Kontext: [mean,std,entropy, atr, bbp, hour_sin, hour_cos],
      new_tail: (L-1, K) – letzter Kontext für den nächsten Chunk.

    Robust gegen:
      - fehlende/ungleiche Basis-Arrays (Trimmen auf gemeinsame M),
      - prev_tail mit falscher Shape (wird verworfen),
      - emit_index als (Date)Index/Series/List (konsequent to_numpy),
      - Längen-Mismatches (emit_index/regime werden auf M getrimmt/gepadded).
    """
    # gewünschte (Train-)Reihenfolge + evtl. weitere Keys hinten anhängen
    ORDER_DEFAULT = ["rf", "lgb", "xgb", "ft", "cnn"]
    keys = [k for k in ORDER_DEFAULT if (k in base and isinstance(base[k], np.ndarray))]
    extra = sorted([k for k in base.keys() if k not in keys and isinstance(base[k], np.ndarray)])
    keys.extend(extra)
    if not keys:
        return (np.empty((0, L, 0), np.float32),
                np.empty((0, 0),    np.float32),
                np.empty((0, 0),    np.float32))

    # gemeinsame Länge M bestimmen und trimmen
    lengths = [len(base[k]) for k in keys]
    M = int(min(lengths))
    if any(len(base[k]) != M for k in keys):
        if log: log.debug(f"[META] Länge uneinheitlich → trimme auf M={M}")
    cur = np.stack([base[k][:M] for k in keys], axis=1).astype(np.float32)  # (M, K)
    K = cur.shape[1]

    # prev_tail validieren/angleichen → (<= L-1, K)
    if not (isinstance(prev_tail, np.ndarray) and prev_tail.ndim == 2 and prev_tail.shape[1] == K):
        prev_tail = np.zeros((0, K), dtype=np.float32)
    if prev_tail.shape[0] > (L - 1):
        prev_tail = prev_tail[-(L-1):, :]

    # Sequenz zusammensetzen und H bauen
    seq = np.concatenate([prev_tail, cur], axis=0)  # (Ttot, K)
    Ttot = seq.shape[0]
    H_list = []
    for i in range(M):
        end = (Ttot - M) + i
        start = end - (L - 1)
        if start < 0:
            pad = np.repeat(seq[[0]], -start, axis=0)
            window = np.concatenate([pad, seq[0:end+1]], axis=0)
        else:
            window = seq[start:end+1]
        # defensiv sicherstellen
        if window.shape != (L, K):
            # harte Korrektur (croppen/padden) falls numerisch mal schief läuft
            if window.shape[0] < L:
                pad = np.repeat(window[[0]], L - window.shape[0], axis=0)
                window = np.concatenate([pad, window], axis=0)
            elif window.shape[0] > L:
                window = window[-L:, :]
            if window.shape[1] != K:
                if window.shape[1] < K:
                    pad = np.zeros((L, K - window.shape[1]), dtype=window.dtype)
                    window = np.concatenate([window, pad], axis=1)
                else:
                    window = window[:, :K]
        H_list.append(window.astype(np.float32))
    H = np.stack(H_list, axis=0).astype(np.float32)  # (M, L, K)

    # neuer Tail: die letzten L-1 Zeitschritte von seq
    new_tail_len = min(L - 1, Ttot)
    new_tail = seq[Ttot - new_tail_len:Ttot].astype(np.float32)

    # ===== Kontext C (7D): [mean,std,entropy, atr, bbp, hour_sin, hour_cos] =====
    last_row = H[:, -1, :]                                 # (M, K)
    mean_k = last_row.mean(axis=1, keepdims=True)
    std_k  = last_row.std(axis=1, keepdims=True)
    p = np.clip(last_row, 1e-6, 1.0 - 1e-6)
    ent = (-(p*np.log(p) + (1-p)*np.log(1-p))).mean(axis=1, keepdims=True)

    # Regime-Features robust extrahieren + auf M bringen
    def _get_reg(key: str) -> np.ndarray:
        if isinstance(regime_dict, dict) and key in regime_dict and isinstance(regime_dict[key], np.ndarray):
            arr = regime_dict[key].astype(np.float32)
        else:
            arr = np.zeros((0,), dtype=np.float32)
        if arr.shape[0] < M:
            # mit letzten Wert oder 0 auffüllen
            fill = arr[-1] if arr.shape[0] > 0 else 0.0
            arr = np.pad(arr, (0, M - arr.shape[0]), constant_values=fill)
        return arr[:M].reshape(-1, 1).astype(np.float32)

    atr = _get_reg("atr")
    bbp = _get_reg("bbp")

    # Zeitmerkmale robust aus emit_index (egal ob Index/Series/List)
    try:
        idx = pd.to_datetime(emit_index, errors="coerce")
        # hour/minute als numpy-float — kein .reshape auf Index-Objekten!
        hrs = getattr(idx, "hour", None)
        mins= getattr(idx, "minute", None)
        if hrs is None or mins is None:
            # falls 'idx' kein DatetimeIndex ist, als Serie interpretieren
            idx = pd.to_datetime(np.asarray(emit_index))
            hrs = idx.hour
            mins= idx.minute
        hours = (np.asarray(hrs,  dtype=np.float32) +
                 np.asarray(mins, dtype=np.float32)/60.0)
        if hours.shape[0] < M:
            fill = hours[-1] if hours.size > 0 else 0.0
            hours = np.pad(hours, (0, M - hours.shape[0]), constant_values=fill)
        hours = hours[:M]
    except Exception:
        hours = np.zeros((M,), dtype=np.float32)

    hour_sin = np.sin(2.0*np.pi*hours/24.0).astype(np.float32).reshape(-1, 1)
    hour_cos = np.cos(2.0*np.pi*hours/24.0).astype(np.float32).reshape(-1, 1)

    C = np.concatenate(
        [mean_k.astype(np.float32),
         std_k.astype(np.float32),
         ent.astype(np.float32),
         atr, bbp,
         hour_sin, hour_cos],
        axis=1
    )  # (M, 7)

    # Falls der Meta-Checkpoint eine feste ctx_dim erwartet → trimmen/padden
    if isinstance(target_ctx_dim, int) and target_ctx_dim > 0 and C.shape[1] != target_ctx_dim:
        if C.shape[1] > target_ctx_dim:
            C = C[:, :target_ctx_dim]
        else:
            pad = np.zeros((C.shape[0], target_ctx_dim - C.shape[1]), dtype=np.float32)
            C = np.concatenate([C, pad], axis=1)

    return H, C.astype(np.float32), new_tail

# ============================ META – H/C Builder ===========================
TRAIN_BASE_ORDER = ("rf", "lgb", "xgb", "ft", "cnn")

# ============================ FINAL-MIX (A & B) ============================
def compute_final_mix(
    base: dict,
    p_meta: Optional[np.ndarray],
    alpha: float,
    mode: str = "A",
    regime_vals: Optional[dict] = None,
    log=None
):
    """
    Mischt Meta-Probas und Basismittel.
      A: linear  p_final = (1-α)*p_meta + α*mean(base)
      B: dynamisch; α_dyn = α * (1 - disagreement_norm) * atr_factor
         - disagreement_norm: Normierung der Std der Basis-Probas pro Zeile
         - atr_factor: bei hoher ATR schrumpfen wir α (mehr Vertrauen in Meta)

    Returns:
      p_final (M,), p_base_avg (M,), w_used (M,)
    """
    # --- Base sammeln und robust stacken ---
    base_arrays = [v for v in (base or {}).values() if isinstance(v, np.ndarray)]
    if len(base_arrays) == 0:
        # Fallback: nur Meta oder 0.5
        if p_meta is not None:
            pf = np.clip(np.asarray(p_meta, dtype=np.float32).reshape(-1), 1e-7, 1.0-1e-7)
            return pf, None, None
        return None, None, None

    P = np.stack(base_arrays, axis=1).astype(np.float32)   # (M, K)
    p_base_avg = P.mean(axis=1)                            # (M,)
    M = p_base_avg.shape[0]

    # --- Wenn kein Meta verfügbar → nur Basismittel ---
    if p_meta is None:
        pf = np.clip(p_base_avg, 1e-7, 1.0-1e-7).astype(np.float32)
        return pf, p_base_avg.astype(np.float32), np.ones_like(pf, dtype=np.float32)

    # --- α und Modus normalisieren ---
    alpha = float(max(0.0, min(1.0, alpha if alpha is not None else 0.0)))
    mode = (mode or "A").upper()

    # --- Gewicht w pro Zeile: Form (M,) sicherstellen ---
    if mode == "A":
        w = np.full((M,), fill_value=alpha, dtype=np.float32)
    else:
        # B: α_dyn = α * (1 - disagreement_norm) * atr_factor
        std_row = P.std(axis=1)                                # (M,)
        # 0.25 als robuste Skala (Probas in [0,1], K~5); clip auf [0,1]
        disagreement_norm = np.clip(std_row / 0.25, 0.0, 1.0)  # (M,)
        w = alpha * (1.0 - disagreement_norm)                  # (M,)

        # ATR-basiert leicht schrumpfen (hohe ATR → eher Meta)
        if isinstance(regime_vals, dict) and isinstance(regime_vals.get("atr", None), np.ndarray):
            atr = np.asarray(regime_vals["atr"], dtype=np.float32).reshape(-1)
            if atr.shape[0] != M:
                # auf M trimmen/padden
                if atr.shape[0] < M:
                    fill = atr[-1] if atr.size > 0 else 0.0
                    atr = np.pad(atr, (0, M - atr.shape[0]), constant_values=fill)
                atr = atr[:M]
            med = float(np.median(atr)) + 1e-6
            atr_factor = 1.0 / (1.0 + (atr / med))            # ~ (0,1]
            w *= np.clip(atr_factor, 0.2, 1.0)

        w = w.astype(np.float32)

    # --- Shapes angleichen (robust gegen Off-by-One) ---
    p_meta = np.asarray(p_meta, dtype=np.float32).reshape(-1)
    if p_meta.shape[0] != M:
        m = min(M, p_meta.shape[0])
        if log is not None:
            log.warning(f"[MIX] Länge uneinheitlich: base={M}, meta={p_meta.shape[0]} → trimme auf {m}")
        p_base_avg = p_base_avg[:m]
        w = w[:m]
        p_meta = p_meta[:m]
        M = m

    # --- Finale Mischung ---
    p_final = (1.0 - w) * p_meta + w * p_base_avg
    p_final = np.clip(p_final, 1e-6, 1.0 - 1e-6).astype(np.float32)
    return p_final, p_base_avg.astype(np.float32), w.astype(np.float32)

# ============================ META LOADER/PRED =============================
def _pick_n_heads(d_model: int, max_heads: int = 8) -> int:
    for h in range(min(max_heads, d_model), 0, -1):
        if d_model % h == 0:
            return h
    return 1

def _resolve_meta_class(log):
    if not MetaClassCandidates:
        log.error("[META][BUILD] Keine Meta-Klasse auffindbar.")
        return None, "none"
    name, cls = MetaClassCandidates[0]
    log.debug(f"[META] Klasse gewählt: {name}")
    return cls, name

def _infer_ctx_dim_from_state_dict(sd) -> int | None:
    """ctx_dim aus Linear ctx_proj.weight (shape=[d_model, ctx_dim]) lesen."""
    if not isinstance(sd, dict):
        return None
    for k, w in sd.items():
        if isinstance(k, str) and "ctx_proj.weight" in k and hasattr(w, "shape") and getattr(w, "ndim", 0) == 2:
            return int(w.shape[1])
    return None

def _infer_d_model_from_state_dict(sd) -> int | None:
    """d_model robust aus Checkpoint ableiten (input_proj, self_attn, heads)."""
    if not isinstance(sd, dict):
        return None
    w = sd.get("input_proj.weight", None)
    if getattr(w, "ndim", 0) == 2:
        return int(w.shape[0])
    for k, t in sd.items():
        if isinstance(k, str) and "self_attn.in_proj_weight" in k and getattr(t, "ndim", 0) == 2:
            if t.shape[0] % 3 == 0:
                return int(t.shape[0] // 3)
    for key in ("w_head.weight", "p_head.weight"):
        t = sd.get(key, None)
        if getattr(t, "ndim", 0) == 2:
            return int(t.shape[1])
    return None

def load_meta_model_robust(model_dir, K: int, L: int, device, log):
    """
    Robuster Meta-Loader:
      - liest d_model/dropout aus best_params.json (bevorzugt meta.*)
      - leitet ctx_dim & d_model (und ggf. K_ckpt) aus dem Checkpoint ab
      - wenn K != K_ckpt: strippe K-abhängige Keys (input_proj, w_head, p_head)
      - lädt mit strict=False; meta.expected_ctx_dim wird gesetzt
    RETURN: (meta_model, {"K":..., "L":..., "ctx_dim":...})
    """
    MetaClass, src_name = _resolve_meta_class(log)
    if MetaClass is None:
        return None, {"K": K, "L": L, "ctx_dim": K}

    # HP aus best_params.json
    d_model, dropout = 64, 0.10
    try:
        bp = json.loads((Path(model_dir) / "best_params.json").read_text(encoding="utf-8"))
        if "meta" in bp and isinstance(bp["meta"], dict):
            d_model = int(bp["meta"].get("d_token", d_model))
            dropout = float(bp["meta"].get("dropout", dropout))
        log.info(f"[META][HP] d_model={d_model}, dropout={dropout} (aus best_params.json)")
    except Exception as e:
        log.warning(f"[META][HP] keine/ungültige best_params.json → d_model={d_model}, dropout={dropout} ({e})")

    # Checkpoint laden
    ckpt = None
    for name in ("meta.pt", "meta.pth", "meta_state.pt"):
        p = Path(model_dir) / name
        if p.exists():
            ckpt = p; break
    if ckpt is None:
        log.info("[META][LOAD] Kein meta.pt gefunden → Meta deaktiviert.")
        return None, {"K": K, "L": L, "ctx_dim": K}

    try:
        raw = torch.load(ckpt, map_location="cpu")
        sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        # Prefixe robust entfernen
        if isinstance(sd, dict):
            for pref in ("module.", "model.", "meta.", "state_dict."):
                sub = {k[len(pref):]: v for k, v in sd.items() if isinstance(k, str) and k.startswith(pref)}
                if sub:
                    sd = sub; break
    except Exception as e:
        log.error(f"[META][LOAD] Checkpoint konnte nicht gelesen werden: {e} → Meta deaktiviert.")
        return None, {"K": K, "L": L, "ctx_dim": K}

    # ctx_dim/d_model/K_ckpt ableiten
    ctx_dim_req = _infer_ctx_dim_from_state_dict(sd) or K
    d_model_ckpt = _infer_d_model_from_state_dict(sd)
    if d_model_ckpt is not None and d_model_ckpt != d_model:
        log.warning(f"[META] d_model im Checkpoint={d_model_ckpt} ≠ best_params={d_model} → setze d_model={d_model_ckpt}.")
        d_model = d_model_ckpt

    # K des Checkpoints heuristisch aus Heads/Input-Proj lesen
    def _infer_K_from_sd(sd):
        K_head = None
        if "w_head.weight" in sd and hasattr(sd["w_head.weight"], "shape"):
            K_head = int(sd["w_head.weight"].shape[0])
        if "p_head.weight" in sd and hasattr(sd["p_head.weight"], "shape"):
            K_head = int(sd["p_head.weight"].shape[0]) if K_head is None else K_head
        K_in = None
        if "input_proj.weight" in sd and hasattr(sd["input_proj.weight"], "shape"):
            # [d_model, K]
            K_in = int(sd["input_proj.weight"].shape[1])
        return K_head or K_in

    K_ckpt = _infer_K_from_sd(sd) or K
    if K_ckpt != K:
        log.warning(f"[META] K im Checkpoint={K_ckpt}, aktive Basismodelle K={K} → lade ohne K-abhängige Layer.")
        # K-abhängige Keys entfernen
        for key in list(sd.keys()):
            if key.startswith("input_proj.") or key.startswith("w_head.") or key.startswith("p_head."):
                sd.pop(key, None)

    n_heads = _pick_n_heads(d_model)

    # Modell bauen
    log.info(f"[META][BUILD] src={src_name} | K={K}, L={L}, ctx_dim(built)={ctx_dim_req}, d_model={d_model}, n_heads={n_heads}")
    try:
        meta = MetaClass(K=K, L=L, ctx_dim=ctx_dim_req, d_model=d_model, n_heads=n_heads, dropout=dropout).to(device)
    except TypeError:
        meta = MetaClass(K=K, L=L, ctx_dim=ctx_dim_req, d_model=d_model, n_heads=n_heads).to(device)
    meta.expected_ctx_dim = int(ctx_dim_req)
    meta.eval()

    # Gewichte tolerant laden (nach Stripping)
    ik = meta.load_state_dict(sd, strict=False)
    miss = getattr(ik, "missing_keys", [])
    unex = getattr(ik, "unexpected_keys", [])
    if miss: log.warning(f"[META][LOAD] Missing keys: {list(miss)}")
    if unex: log.warning(f"[META][LOAD] Unexpected keys: {list(unex)}")
    pcount = sum(p.numel() for p in meta.parameters())
    log.info(f"[META][LOAD] Geladen aus '{ckpt.name}' | params={pcount:,}")

    return meta, {"K": K, "L": L, "ctx_dim": ctx_dim_req}


# ============================ REGIME-FEATURES ==============================
def _safe_div(a, b, eps=1e-12):
    return np.divide(a, b + eps, out=np.zeros_like(a, dtype=np.float32), where=(b!=0))

def compute_regime_features_ohlc(df_ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    ATR(14) + Bollinger %B(20) nur mit pandas/numpy (keine ta-Abhängigkeit im Inferenzpfad).
    Erwartet Spalten: High/Low/Close (Groß-/Kleinschreibung tolerant).
    """
    if df_ohlc.empty:
        return pd.DataFrame(index=df_ohlc.index, columns=["atr","bbp"], dtype=np.float32)

    cols = {c.lower(): c for c in df_ohlc.columns}
    H = df_ohlc[cols.get("high","High")].astype(np.float32)
    Lw= df_ohlc[cols.get("low","Low")].astype(np.float32)
    C = df_ohlc[cols.get("close","Close")].astype(np.float32)

    prev_C = C.shift(1)
    tr = pd.concat([
        (H - Lw).abs(),
        (H - prev_C).abs(),
        (Lw - prev_C).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean().astype(np.float32)

    mid = C.rolling(20, min_periods=1).mean()
    sd  = C.rolling(20, min_periods=1).std()
    upper = mid + 2.0*sd
    lower = mid - 2.0*sd
    rng = (upper - lower)
    bbp = _safe_div((C - lower), (rng)).clip(0.0, 1.0).astype(np.float32)

    out = pd.DataFrame({"atr": atr.values.astype(np.float32),
                        "bbp": bbp.values.astype(np.float32)},
                        index=df_ohlc.index)
    return out

# ---- Fallback-Logger, falls cfg["log"] fehlt ----
def _fallback_logger():
    import logging, sys
    logger = logging.getLogger("predict_v3")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | predict_v3 | %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger


# ============================ META: batched predict (MoE + Mix) ============
@torch.no_grad()
def predict_meta_batched_v3(
    meta_model,
    H: np.ndarray,                   # [M, L, K]  (Basis-Outputs Verlauf)
    C: np.ndarray,                   # [M, ctx]   (Kontext: Zeit, Regime, …)
    device: torch.device,
    batch_size: int,
    alpha_base_mix: float,           # α in p_mix = α * p_base + (1-α) * p_now
    log
) -> np.ndarray:
    if meta_model is None or H is None or len(H) == 0:
        return np.empty((0,), dtype=np.float32)

    M = int(H.shape[0])
    out = np.empty((M,), dtype=np.float32)

    meta_model.eval()
    for s in range(0, M, int(batch_size)):
        e  = min(M, s + int(batch_size))
        Hb = torch.tensor(H[s:e], dtype=torch.float32, device=device)
        Cb = torch.tensor(C[s:e], dtype=torch.float32, device=device)
        try:
            w, p_now = meta_model(Hb, Cb)          # w:[B,K] (Gating), p_now:[B,K]
            p_base   = Hb[:, -1, :]                # letzte Zeitscheibe der Basis-Outputs
            p_mix    = alpha_base_mix * p_base + (1.0 - alpha_base_mix) * p_now
            p_hat    = (w * p_mix).sum(dim=1).clamp(1e-7, 1-1e-7)
            out[s:e] = p_hat.float().detach().cpu().numpy()
        except Exception as ex:
            log.error(f"[META] Inferenzfehler @ batch {s}:{e}: {ex}", exc_info=True)
            out[s:e] = 0.5

    # Fallback: wenn Meta ~0.5 konstant → Basis-Mittel
    if np.allclose(out.mean(), 0.5, atol=1e-3) or (np.var(out) < 1e-6):
        log.warning("[META] Output ~0.5-konstant → Fallback auf Basismittel.")
        base_avg = H[:, -1, :].mean(axis=1).astype(np.float32)
        return base_avg
    return out.astype(np.float32)

def stream_predict(args, cfg):
    import gc
    import json
    import pandas as pd
    import numpy as np
    from pathlib import Path

    # -------- Logger robust holen --------
    log = None
    try:
        if isinstance(cfg, dict):
            log = cfg.get("log", None)
    except Exception:
        log = None
    if log is None:
        try:
            log = get_logger("predict_v3", verbose=getattr(args, "verbose", False))
        except NameError:
            import logging
            logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO)
            log = logging.getLogger("predict_v3")

    # -------- Komfort: kurze Aliase --------
    out_path   = Path(args.output).as_posix()
    model_dir  = Path(args.model_dir)
    device     = pick_device(args.device, log)
    amp_dtype  = resolve_amp_dtype(args.precision, device)
    seq_len    = int(args.seq_len)
    horizon_b  = max(1, int(args.label_horizon_min) // freq_to_minutes(args.freq))

    # --- NEW: L_meta aus train_meta.json / meta_spec.json lesen ---
    def _read_meta_history_L(model_dir, fallback):
        for name in ("train_meta.json", "meta_spec.json"):
            p = Path(model_dir) / name
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    if "meta_history_L" in d and int(d["meta_history_L"]) > 0:
                        return int(d["meta_history_L"])
                    if "L" in d and int(d["L"]) > 0:
                        return int(d["L"])
                except Exception:
                    pass
        return fallback

    L_meta = _read_meta_history_L(model_dir, fallback=min(12, seq_len))
    log.info(f"[META] History-Länge für Inferenz: L_meta={L_meta} (seq_len={seq_len})")

    # --- Final-Mix Parameter ---
    final_mode  = (getattr(args, "final_mix", "A") or "A").upper()
    alpha_final = getattr(args, "alpha_final_mix", None)
    if alpha_final is None:
        # Fallback für alte CLI: --alpha-base-mix nutzen
        alpha_final = float(getattr(args, "alpha_base_mix", 0.0))
        if final_mode != "A" and log is not None:
            log.info(f"[MIX] --alpha-final-mix nicht gesetzt → nutze --alpha-base-mix={alpha_final:.3f} für Final-Mix {final_mode}")

    # -------- Modelle/Artefakte laden --------
    feat_list = read_feature_list(model_dir, log)  # Trainings-Featureliste

    rf_list, lgb_list, xgb_list = load_tree_models(model_dir, log)  # Trees

    ft_model = build_and_load_ft(model_dir, len(feat_list), device, log)  # FT

    cnn_model, cnn_norm = build_and_load_cnn(model_dir, len(feat_list), device, log)  # CNN (+Norm)

    meta_model = None
    meta_tail  = None  # für Meta-History

    reader = open_tick_reader(args.ea_tick_file, args.ea_chunk_rows, log)
    wrote_header = False
    total_rows = 0

    for ichunk, df_raw in enumerate(reader, 1):
        log.info(f"--- Chunk #{ichunk} -----------------------------------------------")
        log.debug(f"df_raw.shape={df_raw.shape}\n{df_raw.head(3)}")

        # 1) Resample
        ohlc_all, idx = resample_ticks(df_raw, args.freq, log, price_col="Tick_Bid", time_col="Time")

        # 2) Features wie im Training
        feat_all = build_feature_matrix(ohlc_all, feat_list, log)

        # 3) Windowing & Emissionen
        emit_index, X_seq_feat, X_last_feat, X_flat_feat = make_emissions(
            feat_all, idx, seq_len, horizon_b, log
        )  # shapes: (M, L, F), (M, F), (M, L*F)

        log.debug(f"[Shapes] X_seq(ohlc)={X_seq_feat.shape}, X_last(ohlc)={X_last_feat.shape}")

        if X_seq_feat.shape[0] == 0:
            continue

        # 4) Basis-Modelle
        p_rf, p_lgb, p_xgb = predict_trees_proba(rf_list, lgb_list, xgb_list, X_flat_feat, log)
        p_ft  = predict_ft_batched(ft_model,  X_flat_feat, device, args.nn_batch_size, None, amp_dtype, log) if ft_model  is not None else None
        p_cnn = predict_cnn_batched(cnn_model, X_seq_feat, device, args.nn_batch_size, cnn_norm, amp_dtype, log) if cnn_model is not None else None

        base = {
            "rf":  p_rf  if p_rf  is not None else None,
            "lgb": p_lgb if p_lgb is not None else None,
            "xgb": p_xgb if p_xgb is not None else None,
            "ft":  p_ft  if p_ft  is not None else None,
            "cnn": p_cnn if p_cnn is not None else None,
        }
        base = {k: v for k, v in base.items() if isinstance(v, np.ndarray)}
        log.debug(f"[Base] K={len(base)} | shapes: {{ {', '.join(f'{k}: {v.shape}' for k,v in base.items())} }}")

        # 5) Regime-Features (ATR/BBP)
        regime_vals, atr_arr, bbp_arr = compute_regime_features(ohlc_all, emit_index, log)
        log.debug(f"[Regime] atr/bbp shapes: ({atr_arr.shape[0]},)/({bbp_arr.shape[0]},)")

        # 6) Meta (lazy)
        p_meta = None
        if args.use_meta and len(base) > 0:
            if meta_model is None:
                meta_model, _ = load_meta_model_robust(model_dir=model_dir, K=len(base), L=L_meta, device=device, log=log)

            if meta_model is not None:
                target_ctx_dim = int(getattr(meta_model, "expected_ctx_dim", 0) or 0) or None
                H_meta, C_meta, meta_tail = build_meta_inputs_from_base_preserve_H(
                    base,           # base_preds_dict
                    L_meta,         # History-Länge
                    meta_tail,      # prev_tail
                    emit_index,     # timestamps
                    target_ctx_dim, # ctx-dim (optional)
                    regime_vals,    # regime_dict
                    log=log
                )
                if H_meta is not None and H_meta.shape[0] > 0:
                    if hasattr(meta_model, "predict_proba"):
                        p_meta = np.asarray(meta_model.predict_proba(H_meta, C_meta), dtype=np.float32).reshape(-1)
                    else:
                        with torch.no_grad():
                            Ht = torch.as_tensor(H_meta, dtype=torch.float32, device=device)
                            Ct = torch.as_tensor(C_meta, dtype=torch.float32, device=device)
                            try:
                                logits = meta_model(Ht, Ct)
                            except TypeError:
                                logits = meta_model(Ht)
                            if isinstance(logits, (list, tuple)):
                                logits = logits[0]
                            p_meta = torch.sigmoid(logits.float().view(-1)).cpu().numpy().astype(np.float32)
                else:
                    log.warning("[META] Keine H/C gebaut – Final-Mix nutzt Basismittel als Fallback.")

        # 7) Final-Mix (A/B)
        p_final, p_base_avg, w_used = compute_final_mix(
            base=base,
            p_meta=p_meta,
            alpha=alpha_final,
            mode=final_mode,
            regime_vals=regime_vals,
            log=log
        )
        if p_final is None:
            valid = [v for v in base.values() if isinstance(v, np.ndarray)]
            if valid:
                p_final = np.mean(np.stack(valid, axis=1), axis=1).astype(np.float32)
            else:
                p_final = np.full((len(emit_index),), 0.5, dtype=np.float32)

        # 8) Output
        out = pd.DataFrame({
            "Time":        emit_index,
            "proba_final": p_final.astype(np.float32),
            "proba_meta":  (p_meta if p_meta is not None else np.full_like(p_final, np.nan, dtype=np.float32)),
            "proba_rf":    (p_rf  if p_rf  is not None else np.full_like(p_final, np.nan, dtype=np.float32)),
            "proba_lgb":   (p_lgb if p_lgb is not None else np.full_like(p_final, np.nan, dtype=np.float32)),
            "proba_xgb":   (p_xgb if p_xgb is not None else np.full_like(p_final, np.nan, dtype=np.float32)),
            "proba_ft":    (p_ft  if p_ft  is not None else np.full_like(p_final, np.nan, dtype=np.float32)),
            "proba_cnn":   (p_cnn if p_cnn is not None else np.full_like(p_final, np.nan, dtype=np.float32)),
            "atr":         atr_arr.astype(np.float32),
            "bbp":         bbp_arr.astype(np.float32),
        }).reset_index(drop=True)

        mode = "a" if total_rows > 0 else "w"
        out.to_csv(out_path, index=False, mode=mode, header=(total_rows == 0), float_format="%.6f")
        total_rows += len(out)
        log.info(f"[OUT] +{len(out):,} Zeilen → total={total_rows:,} @ {out_path}")

        # Cleanup
        del out, X_seq_feat, X_last_feat, X_flat_feat, emit_index, ohlc_all, feat_all, df_raw
        gc.collect()


# ============================ CLI =========================================
def parse_args():
    ap = argparse.ArgumentParser(description="Hybrid Ensemble Streaming Predict (v3)")
    ap.add_argument("--ea-tick-file", type=str, required=True, help="Pfad zur EA Tick CSV")
    ap.add_argument("--ea-chunk-rows", type=int, default=200_000, help="Zeilen pro Chunk (CSV Reader)")
    ap.add_argument("--freq", type=str, default="5min", help="Resample-Frequenz (z.B. '1min','5min','1h')")
    ap.add_argument("--seq-len", type=int, default=24, help="Fensterlänge L")
    ap.add_argument("--label-horizon-min", type=int, default=5, help="Horizont in Minuten für y (nur für Windowing/Emit)")
    ap.add_argument("--model-dir", type=str, required=True, help="Verzeichnis mit rf_list.pkl / lgb_list.pkl / xgb_list.pkl / ft.pt / cnn.pt / meta.pt / temp_scaler.pt")
    ap.add_argument("--output", type=str, required=True, help="Pfad zur Ausgabe-CSV")
    ap.add_argument("--use-meta", type=int, default=1, help="1=MetaMoE verwenden, 0=Basis-Mittel")
    ap.add_argument("--use-regime", type=int, default=1, help="1=Regime-Features (ATR/%B) in Meta-Kontext")
    ap.add_argument("--alpha-base-mix", type=float, default=0.7, help="α im Mix p_mix=α*p_base+(1-α)*p_now")
    ap.add_argument("--precision", type=str, default="auto", help="AMP Präzision: auto|fp32|fp16|bf16")
    ap.add_argument("--apply-temp-scaling", type=int, default=1, help="1=FT-Logits durch Temperatur teilen")
    ap.add_argument("--nn-batch-size", type=int, default=1024, help="Batchgröße für FT/CNN-Inferenz")
    ap.add_argument("--device", type=str, default="auto", help="cuda|cpu|auto")
    ap.add_argument("--verbose", action="store_true", help="Mehr Logs")
    ap.add_argument(
    "--final-mix",
    type=str,
    default="A",
    choices=["A", "B"],
    help="Finale Mischung: A = linear (α), B = dynamisch (α·(1-Disagreement)·ATR-Faktor)"
)

    return ap.parse_args()

def main():
    args = parse_args()
    cfg = {}
    try:
        stream_predict(args, cfg)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
