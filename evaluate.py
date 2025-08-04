#!/usr/bin/env python
# evaluate.py
"""
Hybrid-LongTrend – Evaluation & Batch-Inference
Usage:
    python evaluate.py ^
        --model-dir models\hybrid_longtrend_20250802_184344 ^
        --test-file data\rawtickdata3.txt ^
        --seq-len   24 ^
        --bar-min   60
"""
from __future__ import annotations
import argparse, sys, json, pickle
from pathlib import Path
from typing  import List

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix,
)

import torch
from torch.utils.data import TensorDataset, DataLoader

# ───────────────── Pfade & lokale Utilities ───────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))                # utils.*
from utils.features import enrich
from labeling        import make_trend_labels

# ─────────────────────── Argument-Parsing ─────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--model-dir", required=True, type=Path)
p.add_argument("--test-file", required=True, type=Path)
p.add_argument("--seq-len",   type=int, default=24)
p.add_argument("--bar-min",   type=int, default=5,
               help="Aggregations­intervall (Minuten) bei Tick-Input")
args = p.parse_args()

# ──────────────── IMPORT-Fix für Trainer-Klassen ──────────────────────
try:
    # Paket-Installation mit Namespace 'trainers'
    from trainers.hybrid_longtrend_trainer import (
        FTWrapped as _FTWrapper,
        SimpleCNN as _CNNEncoder,
        MetaTransformer,
    )
except ModuleNotFoundError:
    # Lokaler Modul-Import
    from hybrid_longtrend_trainer import (
        FTWrapped as _FTWrapper,
        SimpleCNN as _CNNEncoder,
        MetaTransformer,
    )

from rtdl import FTTransformer             # Backbone aus rtdl
FTBackbone  = FTTransformer                # Alias
FTWrapper   = _FTWrapper
CNNEncoder  = _CNNEncoder

# ─────────────── Chunk-Reader für große Tick-Dateien ──────────────────
def iter_ticks_as_bars(path: Path, bar_min: int, chunksize: int = 2_000_000):
    """Yield-Generator: Tick-CSV in Chunks → OHLCV-Bars"""
    for ch in pd.read_csv(path, parse_dates=["Time"], chunksize=chunksize):
        if {"Tick_Bid", "Tick_Ask"}.issubset(ch.columns):
            ch["price"] = (ch["Tick_Bid"] + ch["Tick_Ask"]) / 2
        else:
            ch["price"] = ch["Tick_Last"]

        ch = ch.rename(columns={"Time": "timestamp"}).set_index("timestamp")
        bars = ch["price"].resample(f"{bar_min}min").ohlc()
        bars["volume"] = ch["Tick_Volume"].resample(f"{bar_min}min").sum()
        bars.dropna(inplace=True)
        bars.columns = ["open", "high", "low", "close", "volume"]
        yield bars.reset_index()

def load_candles(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"]) \
             .set_index("timestamp")

# Daten­quelle zu Iterator
if args.test_file.suffix.lower() in {".txt", ".tick", ".dat"}:
    bar_iter = iter_ticks_as_bars(args.test_file, args.bar_min)
else:
    bar_iter = [load_candles(args.test_file)]

# ───────────────── Utility: JSON-oder-YAML-Fallback ──────────────────
from typing import Dict, Any, List

def json_or_yaml(
    json_path: Path,
    yaml_keys: List[str] | None,
    default: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Versucht zuerst, eine JSON-Datei zu laden. Gibt es die nicht, wird in der
    bereits geladenen config.yaml (cfg_yaml) entlang der verschachtelten Keys
    nachgeschaut. Ist auch dort nichts vorhanden, kommt `default` zurück.
    """
    if json_path.exists():
        return json.load(open(json_path))
    cur: Dict[str, Any] = cfg_yaml
    if yaml_keys:
        for k in yaml_keys:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
    return cur if cur else (default or {})



def adapt_ft_cfg(raw: dict) -> dict:
    """Behält nur Keys, die der aktuelle FTTransformer wirklich akzeptiert.
       Mapped 'd_token' → das erste passende Synonym im Signature."""
    sig_keys = set(inspect.signature(FTBackbone).parameters)

    cfg = {k: v for k, v in raw.items() if k in sig_keys}

    # d_token → alternativer Name, falls nötig
    if "d_token" in raw and "d_token" not in sig_keys:
        for alt in ("d_model", "d_embed", "token_dim"):
            if alt in sig_keys:
                cfg[alt] = raw["d_token"]
                break
    return cfg

# ───────────────────────── Modelle laden ──────────────────────────────
# ───────────────────────── Modelle laden ──────────────────────────────
def load_pickle(p: Path):
    with open(p, "rb") as f: return pickle.load(f)

md = args.model_dir
cfg_yaml_path = md / "config.yaml"
try:
    with open(cfg_yaml_path, "r", encoding="utf-8") as f:   # ← Encoding fix
        cfg_yaml = yaml.safe_load(f)
except FileNotFoundError:
    print("⚠  config.yaml fehlt – nutze Default-Parameter")
    cfg_yaml = {}
except UnicodeDecodeError as e:
    raise RuntimeError(
        f"{cfg_yaml_path} enthält Nicht-ASCII-Zeichen. "
        "Speichere die Datei als UTF-8 oder öffne sie mit korrektem Encoding."
    ) from e

# ---------- FT --------------------------------------------------------
import inspect

def filter_cfg(cfg: Dict[str, Any], cls) -> Dict[str, Any]:
    valid = inspect.signature(cls).parameters
    return {k: v for k, v in cfg.items() if k in valid}

ft_cfg_raw = json_or_yaml(md / "ft_cfg.json", ["model", "ft_params"], {})
ft_cfg     = adapt_ft_cfg(ft_cfg_raw)

# Fallback für rtdl<0.3: map d_token → d_model
if "d_token" in ft_cfg_raw and "d_token" not in ft_cfg:
    ft_cfg["d_model"] = ft_cfg_raw["d_token"]


ft = FTBackbone(**ft_cfg)
ft.load_state_dict(torch.load(md / "ft.pt", map_location="cpu"))
ft.eval()

ft_wrap = FTWrapper(ft)
ft_wrap.load_state_dict(
    torch.load(md / "ft_wrap.pt", map_location="cpu"), strict=False
)
ft_wrap.eval()

# ---------- CNN -------------------------------------------------------
cnn_cfg_path = md / "cnn_cfg.json"
if cnn_cfg_path.exists():
    cnn_cfg = json.load(open(cnn_cfg_path))
    cnn = CNNEncoder(**cnn_cfg)
    cnn.load_state_dict(torch.load(md / "cnn.pt", map_location="cpu"))
    cnn.eval()
else:
    # kein cfg → Instanzierung verschieben, wenn wir die erste Feature-Matrix kennen
    cnn      = None
    cnn_cfg  = None
    print("⚠  cnn_cfg.json nicht gefunden – CNN wird beim ersten Chunk erzeugt")

# ---------- klassische Modelle ---------------------------------------
rf_list  = load_pickle(md / "rf.pkl")
lgb_list = load_pickle(md / "lgb.pkl")
xgb_list = load_pickle(md / "xgb.pkl")

# ---------- Meta-Transformer -----------------------------------------
meta_cfg_path = md / "meta_cfg.json"
if meta_cfg_path.exists():
    meta_cfg = json.load(open(meta_cfg_path))
else:
    meta_cfg = cfg_yaml["meta"]                    # YAML-Fallback
    print("⚠  meta_cfg.json nicht gefunden – nehme Parameter aus config.yaml")

meta = MetaTransformer(**meta_cfg)
meta.load_state_dict(torch.load(md / "meta.pt", map_location="cpu"))
meta.eval()


def mean_preds(models, X_flat, kind):
    if kind == "rf":
        return np.mean([m.predict_proba(X_flat)[:,1] for m in models], axis=0)
    if kind == "lgb":
        return np.mean([m.predict(X_flat) for m in models], axis=0)
    if kind == "xgb":
        return np.mean([m.predict_proba(X_flat)[:,1] for m in models], axis=0)
    raise ValueError(kind)

# ─────────── Trend-Hyperparameter (aus config.yaml) ───────────────────
cfg_path = md / "config.yaml"
if cfg_path.exists():
    trend_cfg = yaml.safe_load(open(cfg_path))["trend"]
else:
    trend_cfg = {}
dc_thres = trend_cfg.get("dc_threshold_pct", 0.05)
w_list   = trend_cfg.get("windows",          [6,12,24,48,96])
tau      = trend_cfg.get("t_stat_thresh",    2.2)

# ───────────────────── Gesamtergebnis-Listen ──────────────────────────
all_probs, all_labels = [], []

# ─────────────────────── Verarbeitung je Chunk ────────────────────────
for df_raw in bar_iter:
    # 1) Spalten-Alias & price
    df_raw = df_raw.rename(
        columns={c: "Close" for c in df_raw.columns
                 if c.lower() in {"close","close_price"}})
    if "price" not in df_raw and "Close" in df_raw:
        df_raw["price"] = df_raw["Close"]

    # 2) Labels
    if "label" not in df_raw:
        trend = make_trend_labels(df_raw.copy(), dc_thres, w_list, tau)
        df_raw["label"] = (trend["trend_side"] == 1).astype(np.float32)

    # 3) Features
    df_feat = enrich(df_raw).dropna()

        # nach: df_feat = enrich(df_raw).dropna()
    if cnn is None:
        cnn_cfg = {            # Minimal-Defaults jetzt sicher
            "n_channels": df_feat.shape[1],
            "n_conv":     3,
            "ks":         3,
            "pool":       2,
            "emb_dim":    64,
        }
        cnn = CNNEncoder(**cnn_cfg)
        cnn.load_state_dict(torch.load(md / "cnn.pt", map_location="cpu"))
        cnn.eval()


    for col in ["open","high","low","close","volume"]:
        match = next((c for c in df_raw if c.lower()==col), None)
        if match and col not in df_feat:
            df_feat[col] = df_raw[match].values[-len(df_feat):]

    vals = df_feat.to_numpy(dtype=np.float32)
    labels_arr = df_raw["label"].loc[df_feat.index].to_numpy(dtype=np.float32)

    # 4) Sequenzen
    seq_len = args.seq_len
    X, y = [], []
    for i in range(seq_len, len(vals)):
        X.append(vals[i-seq_len:i])
        y.append(labels_arr[i])
    if not X:           # Chunk zu kurz
        continue
    X = np.stack(X); y = np.array(y)

    dl = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                    batch_size=256, shuffle=False)

    # 5) Ensemble-Forward
    probs, labels_out = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb_np = xb.numpy().reshape(len(xb), -1)
            p_rf  = mean_preds(rf_list,  xb_np, "rf")
            p_lgb = mean_preds(lgb_list, xb_np, "lgb")
            p_xgb = mean_preds(xgb_list, xb_np, "xgb")
            p_ft  = torch.sigmoid(ft_wrap(xb)).numpy().ravel()
            p_cnn = torch.sigmoid(cnn(xb.permute(0,2,1))).numpy().ravel()
            stacked = np.vstack([p_rf,p_lgb,p_xgb,p_ft,p_cnn]).T.astype(np.float32)
            p_meta  = torch.sigmoid(meta(torch.tensor(stacked))).numpy().ravel()
            probs.append(p_meta); labels_out.append(yb.numpy())
    all_probs.append(np.concatenate(probs))
    all_labels.append(np.concatenate(labels_out))

# ─────────────────────── Gesamt-Metriken ──────────────────────────────
probs       = np.concatenate(all_probs)
labels_out  = np.concatenate(all_labels)

acc  = accuracy_score(labels_out,(probs>=0.5).astype(int))
auc  = roc_auc_score(labels_out, probs)
cm   = confusion_matrix(labels_out,(probs>=0.5).astype(int))

print("\n────────  Evaluation  ────────")
print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("Confusion matrix [TN FP; FN TP]:")
print(cm)
print("\nDetailed report:")
print(classification_report(labels_out,(probs>=0.5).astype(int),
                            target_names=["neg","pos"]))
