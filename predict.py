#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Memory-safe predict script for the full ensemble (RF/LGB/XGB/FT/CNN) + MetaMoE.

Highlights
----------
• Streamt EA-Ticks per Chunk (kein Voll-Laden in RAM)
• Resample mit "Carry" des letzten Buckets
• Sequenzen/Labels (mit Horizon) chunkweise + History-Puffer
• Batched NN-Inference
• DA/MDA Sanity-Check on-the-fly
"""
from __future__ import annotations

import argparse, json, os, sys, pickle, gc
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from contextlib import contextmanager
from collections import OrderedDict


import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Optional: sklearn/lightgbm/xgboost
try:
    import joblib
except Exception:
    joblib = None
try:
    import lightgbm as lgb  # noqa: F401
except Exception:
    lgb = None
try:
    import xgboost as xgb  # noqa: F401
except Exception:
    xgb = None

# Project classes (werden dynamisch importiert)
FTModel = None
CNNModel = None
MetaMoEClass = None
FTWrappedClass = None

# ---------------- Lazy wrappers to reconstruct models from state_dict ----------------
class _LazyFT:
    def __init__(self, model_dir: Path, device: torch.device, cfg: Dict[str, Any], state_dict):
        self.model_dir = model_dir
        self.device = device
        self.cfg = cfg
        self.state_dict = state_dict
        self._model = None

    def __call__(self, x_num: torch.Tensor):
        if self._model is None:
            # Build when first tensor seen (to know feature dim)
            D = x_num.shape[1]
            try:
                import joblib
                from trainers.hybrid_longtrend_trainer import FTWrapped
                import rtdl
                # WICHTIG: kein TaskType verwenden, sonst erwartet PEFT 'input_ids'
                from peft import LoraConfig, get_peft_model
            except Exception as e:
                raise RuntimeError(f"FT-LazyLoad: benötigte Pakete fehlen: {e}")
            # Read study for best params if available
            n_blocks = 4
            pkl = self.model_dir / "ft_study.pkl"
            if pkl.exists():
                try:
                    study = joblib.load(pkl)
                    if hasattr(study, "best_params"):
                        n_blocks = int(study.best_params.get("n_blocks", n_blocks))
                except Exception:
                    pass
            # Make base
            ft_base = rtdl.FTTransformer.make_default(
                n_num_features=D,
                cat_cardinalities=(),
                d_out=1,
                n_blocks=n_blocks
            )
            # LoRA config like training
            try:
                lcfg = LoraConfig(
                    r=4,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=['ffn.linear_first']
                )
                ft_base = get_peft_model(ft_base, lcfg)
            except Exception:
                # If PEFT missing, proceed without; we'll load with strict=False
                pass
            model = FTWrapped(ft_base).to(self.device).eval()
            # Load weights
            try:
                model.load_state_dict(self.state_dict, strict=True)
            except Exception:
                model.load_state_dict(self.state_dict, strict=False)
                print("⚠️  FT: state_dict wurde mit strict=False geladen (PEFT/Keys weichen ab).", flush=True)
            self._model = model
        return self._model(x_num)


class _LazyCNN:
    def __init__(self, model_dir: Path, device: torch.device, cfg: Dict[str, Any], state_dict):
        self.model_dir = model_dir
        self.device = device
        self.cfg = cfg
        self.state_dict = state_dict
        self._model = None

    def __call__(self, x_seq: torch.Tensor):
        # x_seq: (N, F, T)
        if self._model is None:
            F = x_seq.shape[1]
            try:
                import joblib
                from trainers.hybrid_longtrend_trainer import SimpleCNN
            except Exception as e:
                raise RuntimeError(f"CNN-LazyLoad: benötigte Pakete fehlen: {e}")
            n_filters = 32
            pkl = self.model_dir / "cnn_study.pkl"
            if pkl.exists():
                try:
                    study = joblib.load(pkl)
                    if hasattr(study, "best_params"):
                        n_filters = int(study.best_params.get("n_filters", n_filters))
                except Exception:
                    pass
            model = SimpleCNN(F, n_filters).to(self.device).eval()
            try:
                model.load_state_dict(self.state_dict, strict=True)
            except Exception:
                model.load_state_dict(self.state_dict, strict=False)
                print("⚠️  CNN: state_dict mit strict=False geladen.", flush=True)
            self._model = model
        return self._model(x_seq)

class _LazyMeta:
    """
    Baut MetaMoE erst beim ersten Call – Dimensionen werden aus dem Checkpoint
    gelesen, damit ctx/d_model zu den Gewichten passen.
    """
    def __init__(self, model_dir: Path, device: torch.device, cfg: Dict[str, Any], state_dict: "OrderedDict"):
        self.model_dir = model_dir
        self.device = device
        self.cfg = cfg
        self.state_dict = state_dict
        self._model = None
        self.expected_ctx_dim = 0
        self.expected_d_model = None

    def __call__(self, H: torch.Tensor, C: Optional[torch.Tensor] = None):
        if self._model is None:
            if MetaMoEClass is None:
                raise ImportError("MetaMoE Klasse nicht gefunden (siehe _try_import_models).")

            sd = self.state_dict
            # ---- Dimensionen aus state_dict lesen (bevorzugt ctx_proj -> garantiert Add-kompatibel)
            d_model = None
            ctx_dim = None
            if "ctx_proj.weight" in sd and hasattr(sd["ctx_proj.weight"], "shape"):
                w = sd["ctx_proj.weight"]
                d_model = int(w.shape[0])   # out_features
                ctx_dim = int(w.shape[1])   # in_features
            if d_model is None and "input_proj.weight" in sd and hasattr(sd["input_proj.weight"], "shape"):
                w = sd["input_proj.weight"]
                d_model = int(w.shape[0])
            # Fallbacks aus config
            meta_cfg = self.cfg.get("meta", {})
            if d_model is None:
                d_model = int(meta_cfg.get("d_token", 96))
            if ctx_dim is None:
                ctx_dim = int(meta_cfg.get("ctx_dim", 0))

            self.expected_d_model = d_model
            self.expected_ctx_dim = ctx_dim

            # Kopfanzahl passend zu d_model wählen
            for h in (8, 4, 2, 1):
                if d_model % h == 0:
                    n_heads = h
                    break

            # K, L aus H ableiten
            B, L, K = H.shape
            self._model = MetaMoEClass(
                K=K, L=L,
                ctx_dim=ctx_dim,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=1,
                dropout=float(meta_cfg.get("dropout", 0.1)),
            ).to(self.device).eval()

            # Gewichte laden
            try:
                self._model.load_state_dict(sd, strict=True)
            except Exception:
                print("⚠️  META: state_dict mit strict=False geladen (Keys/Formate weichen ab).", flush=True)
                self._model.load_state_dict(sd, strict=False)

        # ---- Kontext-Form: erwartet (B, ctx_dim)
        if self.expected_ctx_dim > 0:
            if C is None:
                C = torch.zeros((H.shape[0], self.expected_ctx_dim), device=self.device, dtype=H.dtype)
            elif C.ndim == 3:
                C = C[:, -1, :]  # (B,L,C) -> letzter Schritt
            if C.shape[1] != self.expected_ctx_dim:
                C = torch.zeros((H.shape[0], self.expected_ctx_dim), device=self.device, dtype=H.dtype)
        else:
            C = torch.zeros((H.shape[0], 0), device=self.device, dtype=H.dtype)

        return self._model(H, C)


@contextmanager
def _no_grad():
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        yield
    finally:
        torch.set_grad_enabled(prev)


def _write_chunk_csv(df: pd.DataFrame, out_path: Path, header: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if header else "a"
    df.to_csv(out_path, mode=mode, header=header, index=False)


def smart_device(req: str) -> torch.device:
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_yaml(p: Path) -> Dict[str, Any]:
    import yaml
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_input(p: Path) -> np.ndarray:
    """
    Lädt (N,T,F) aus .npy/.npz (Key 'X' oder erster Array).
    """
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
        if arr.ndim != 3:
            raise ValueError(f"{p} muss Shape (N,T,F) haben, gefunden {arr.shape}")
        return arr.astype(np.float32)
    if p.suffix.lower() == ".npz":
        z = np.load(p)
        if "X" in z:
            arr = z["X"]
        else:
            if len(z.files) == 0:
                raise ValueError(f"{p} enthält keinen Array")
            arr = z[z.files[0]]
        if arr.ndim != 3:
            raise ValueError(f"{p} muss Shape (N,T,F) haben, gefunden {arr.shape}")
        return arr.astype(np.float32)
    raise ValueError(f"Unsupported input format: {p.suffix}. Nutze .npy/.npz oder --ea-tick-file.")


def _try_import_models():
    global FTModel, CNNModel, MetaMoEClass, FTWrappedClass
    candidates = [
        ("trainers.hybrid_longtrend_trainer", None, "SimpleCNN", "MetaMoE", "FTWrapped"),
        ("models.cnn", None, "SimpleCNN", None, None),
        ("models.meta", None, None, "MetaMoE", None),
    ]
    for mod, ft_cls, cnn_cls, meta_cls, ftwrap_cls in candidates:
        try:
            m = __import__(mod, fromlist=[ft_cls or "_", cnn_cls or "_", meta_cls or "_", ftwrap_cls or "_"])
            if cnn_cls and hasattr(m, cnn_cls): CNNModel = getattr(m, cnn_cls)
            if meta_cls and hasattr(m, meta_cls) and MetaMoEClass is None:
                MetaMoEClass = getattr(m, meta_cls)
            if ftwrap_cls and hasattr(m, ftwrap_cls) and FTWrappedClass is None:
                FTWrappedClass = getattr(m, ftwrap_cls)
        except Exception:
            continue


def load_rf_list(model_dir: Path):
    pkl = model_dir / "rf_list.pkl"
    if pkl.exists():
        if joblib is not None:
            return joblib.load(pkl)
        with open(pkl, "rb") as f:
            return pickle.load(f)
    return []


def load_lgb_list(model_dir: Path):
    pkl = model_dir / "lgb_list.pkl"
    if pkl.exists():
        if joblib is not None:
            return joblib.load(pkl)
        with open(pkl, "rb") as f:
            return pickle.load(f)
    return []


def load_xgb_list(model_dir: Path):
    pkl = model_dir / "xgb_list.pkl"
    if pkl.exists():
        if joblib is not None:
            return joblib.load(pkl)
        with open(pkl, "rb") as f:
            return pickle.load(f)
    return []


def load_ft(model_dir: Path, device: torch.device, cfg: Dict[str, Any]):
    pt = model_dir / "ft.pt"
    if not pt.exists():
        return None, None
    obj = torch.load(pt, map_location="cpu")
    # Case 1: full nn.Module checkpoint or dict with 'model'
    if isinstance(obj, nn.Module):
        model = obj.to(device).eval()
        return model, None
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], nn.Module):
        model = obj["model"].to(device).eval()
        return model, obj.get("meta", None)
    # Case 2: pure state_dict -> build lazily on first use
    if isinstance(obj, (dict, OrderedDict)) or hasattr(obj, "keys"):
        try:
            sd = obj if hasattr(obj, "keys") else obj.get("state_dict", obj)
            lazy = _LazyFT(model_dir, device, cfg, sd)
            return lazy, None
        except Exception as e:
            raise RuntimeError(f"Kann Modell aus state_dict nicht rekonstruieren: {e}")
    raise RuntimeError("ft.pt unbekanntes Format – erwartet nn.Module oder state_dict.")


def load_cnn(model_dir: Path, device: torch.device, cfg: Dict[str, Any]):
    pt = model_dir / "cnn.pt"
    if not pt.exists():
        return None, None
    obj = torch.load(pt, map_location="cpu")
    # Case 1: full nn.Module checkpoint or dict with 'model'
    if isinstance(obj, nn.Module):
        model = obj.to(device).eval()
        return model, None
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], nn.Module):
        model = obj["model"].to(device).eval()
        return model, obj.get("meta", None)
    # Case 2: pure state_dict -> build lazily on first use
    if isinstance(obj, (dict, OrderedDict)) or hasattr(obj, "keys"):
        try:
            sd = obj if hasattr(obj, "keys") else obj.get("state_dict", obj)
            lazy = _LazyCNN(model_dir, device, cfg, sd)
            return lazy, None
        except Exception as e:
            raise RuntimeError(f"Kann Modell aus state_dict nicht rekonstruieren: {e}")
    raise RuntimeError("cnn.pt unbekanntes Format – erwartet nn.Module oder state_dict.")


def load_meta(model_dir: Path, device: torch.device, cfg: Dict[str, Any]):
    pt = model_dir / "meta.pt"
    if not pt.exists():
        return None, None
    obj = torch.load(pt, map_location="cpu")
    # Case 1: full nn.Module checkpoint or dict with 'model'
    if isinstance(obj, nn.Module):
        model = obj.to(device).eval()
        return model, None
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], nn.Module):
        model = obj["model"].to(device).eval()
        return model, obj.get("meta", None)
    # Case 2: pure state_dict -> build lazily on first use
    if isinstance(obj, (dict, OrderedDict)) or hasattr(obj, "keys"):
        try:
            sd = obj if hasattr(obj, "keys") else obj.get("state_dict", obj)
            lazy = _LazyMeta(model_dir, device, cfg, sd)
            return lazy, None
        except Exception as e:
            raise RuntimeError(f"Kann Modell aus state_dict nicht rekonstruieren: {e}")
    raise RuntimeError("meta.pt unbekanntes Format – erwartet nn.Module oder state_dict.")

def load_temp_scaler(model_dir: Path) -> Optional[float]:
    """
    Lädt optional eine Temperatur T aus temp_scaler.pt.
    Unterstützt:
      - reiner float/int
      - dict mit Schlüsseln {'T', 'temperature', 'temp', 'tau'}
      - Objekt mit Attribut 'T' oder 'temperature'
    """
    pt = model_dir / "temp_scaler.pt"
    if not pt.exists():
        return None
    obj = torch.load(pt, map_location="cpu")
    # primitive
    if isinstance(obj, (int, float)):
        return float(obj)
    # dict
    if isinstance(obj, dict):
        for k in ("T", "temperature", "temp", "tau"):
            if k in obj:
                try:
                    return float(obj[k])
                except Exception:
                    pass
    # object attribute
    for attr in ("T", "temperature"):
        if hasattr(obj, attr):
            try:
                return float(getattr(obj, attr))
            except Exception:
                pass
    return None

def flatten_seq(X_seq: np.ndarray) -> np.ndarray:
    # (N,T,F) -> (N, T*F)
    N, T, F = X_seq.shape
    return X_seq.reshape(N, T * F).astype(np.float32, copy=False)

def build_meta_inputs_from_preds(preds_base: np.ndarray, L: int, ctx_dim: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Erzeuge H (N, L, K) als rollendes Fenster der Basis-Probabilitäten
    und C als 2D-Kontext (N, ctx_dim). Für i-ter Zeitpunkt liegt die
    jüngste Basis-Proba am Ende von H[i].
    """
    preds_base = np.asarray(preds_base, dtype=np.float32)
    if preds_base.ndim != 2:
        raise ValueError(f"preds_base erwartet (N,K), bekommen {preds_base.shape}")
    N, K = preds_base.shape
    H = np.zeros((N, L, K), dtype=np.float32)
    for i in range(N):
        s = max(0, i - L + 1)
        frag = preds_base[s:i+1]            # (len, K)
        H[i, -frag.shape[0]:, :] = frag     # rechtsbündig einfügen
    C = np.zeros((N, ctx_dim), dtype=np.float32) if ctx_dim > 0 else np.zeros((N, 0), dtype=np.float32)
    return H, C

def predict_trees(rf_list, lgb_list, xgb_list, X_flat: np.ndarray):
    import numpy as np
    import pandas as pd

    def _as_2d(X):
        Xv = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
        if Xv.ndim == 1:
            Xv = Xv.reshape(-1, 1)
        return Xv

    def _binary_proba(model, X):
        """
        Liefert P(y=1) robust für verschiedene Modelldialekte:
        - sklearn (predict_proba / predict)
        - LightGBM Booster (predict)
        - XGBoost Booster (predict auf DMatrix)
        """
        Xv = _as_2d(X)

        # --- LightGBM: native Booster ---
        try:
            import lightgbm as lgb
            if isinstance(model, lgb.Booster):
                y = model.predict(
                    Xv,
                    num_iteration=getattr(model, "best_iteration", None) or None,
                )
                y = np.asarray(y)
                if y.ndim == 2:
                    # (n,2) -> Spalte für Klasse 1; (n,1) -> squeeze
                    y = y[:, 1] if y.shape[1] == 2 else y.squeeze(-1)
                else:
                    y = y.squeeze()
                return y.astype(np.float32, copy=False)
        except Exception:
            pass

        # --- XGBoost: native Booster ---
        try:
            import xgboost as xgb
            if isinstance(model, xgb.Booster):
                dm = xgb.DMatrix(Xv)
                # best_iteration kann None sein; dann nimmt XGB alle Trees
                best_it = getattr(model, "best_iteration", None)
                if best_it is not None:
                    y = model.predict(dm, iteration_range=(0, best_it + 1))
                else:
                    y = model.predict(dm)
                y = np.asarray(y).squeeze()
                return y.astype(np.float32, copy=False)
        except Exception:
            pass

        # --- sklearn-like APIs ---
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(Xv))
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1].astype(np.float32, copy=False)
            return proba.squeeze().astype(np.float32, copy=False)

        if hasattr(model, "predict"):
            y = np.asarray(model.predict(Xv))
            if y.ndim == 2 and y.shape[1] >= 2:
                return y[:, 1].astype(np.float32, copy=False)
            return y.squeeze().astype(np.float32, copy=False)

        raise TypeError(f"Unsupported model type for probability output: {type(model)}")

    def _avg_proba(models, X):
        vals = []
        for m in (models or []):
            if m is None:
                continue
            p = _binary_proba(m, X)
            vals.append(p.reshape(-1).astype(np.float32, copy=False))
        if not vals:
            # Falls keine Modelle vorhanden: neutrale 0.5
            return np.full((len(X),), 0.5, dtype=np.float32)
        return np.mean(np.vstack(vals), axis=0).astype(np.float32, copy=False)

    p_rf  = _avg_proba(rf_list, X_flat)
    p_lgb = _avg_proba(lgb_list, X_flat)
    p_xgb = _avg_proba(xgb_list, X_flat)
    return p_rf, p_lgb, p_xgb



def predict_ft_batched(ft_model: Optional[nn.Module],
                       X_seq: np.ndarray,
                       device: torch.device,
                       batch_size: int) -> np.ndarray:
    if ft_model is None:
        return np.full((len(X_seq),), 0.5, dtype=np.float32)
    X_flat = flatten_seq(X_seq)
    out_all = np.empty((len(X_seq),), dtype=np.float32)
    with _no_grad():
        for i in range(0, len(X_flat), batch_size):
            xb = torch.from_numpy(X_flat[i:i+batch_size]).to(device)
            ob = ft_model(xb)
            if isinstance(ob, dict) and "logits" in ob:
                ob = ob["logits"]
            if ob.ndim > 1:
                ob = ob.squeeze(-1)
            pb = torch.sigmoid(ob).float().cpu().numpy().astype(np.float32, copy=False)
            out_all[i:i+batch_size] = pb
            del xb, ob
    return out_all


def predict_cnn_batched(cnn_model: Optional[nn.Module],
                        X_seq: np.ndarray,
                        device: torch.device,
                        batch_size: int) -> np.ndarray:
    if cnn_model is None:
        return np.full((len(X_seq),), 0.5, dtype=np.float32)
    out_all = np.empty((len(X_seq),), dtype=np.float32)
    with _no_grad():
        for i in range(0, len(X_seq), batch_size):
            Xb = np.transpose(X_seq[i:i+batch_size], (0, 2, 1)).copy()  # (N,F,T)
            xb = torch.from_numpy(Xb).to(device)
            ob = cnn_model(xb)
            if isinstance(ob, (list, tuple)):
                ob = ob[0]
            if ob.ndim > 1:
                ob = ob.squeeze(-1)
            pb = torch.sigmoid(ob).float().cpu().numpy().astype(np.float32, copy=False)
            out_all[i:i+batch_size] = pb
            del xb, ob
    return out_all

def predict_meta(meta_model: Optional[nn.Module],
                 preds_base: np.ndarray,
                 L: int, ctx_dim: int,
                 device: torch.device,
                 alpha_base_mix: float,
                 temp_T: Optional[float] = None) -> np.ndarray:
    """
    Meta-Verschmelzung mit Fallbacks:
      - baut H/C aus Basis-Probas
      - robust gegen NaNs/Shapes
      - vermeidet degenerierte 0.5-Ausgabe
    """
    preds_base = np.asarray(preds_base, dtype=np.float32)
    if preds_base.ndim != 2:
        raise ValueError(f"preds_base erwartet (N,K), bekommen {preds_base.shape}")
    N, K = preds_base.shape

    def _safe_sigmoid_logit_temp(p: torch.Tensor, T: Optional[float]):
        p = torch.nan_to_num(p, nan=0.5).clamp(1e-6, 1-1e-6)
        if T is not None and T > 0:
            p = torch.sigmoid(torch.logit(p) / float(T)).clamp(1e-6, 1-1e-6)
        return p

    # Fallback: kein Meta-Modell → Durchschnitt
    if meta_model is None:
        p = preds_base.mean(axis=1)
        t = torch.tensor(p, dtype=torch.float32)
        return _safe_sigmoid_logit_temp(t, temp_T).cpu().numpy().astype(np.float32, copy=False)

    # H/C bauen
    H_np, C_np = build_meta_inputs_from_preds(preds_base, L=L, ctx_dim=ctx_dim)
    with _no_grad():
        H = torch.from_numpy(H_np).to(device)
        C = torch.from_numpy(C_np).to(device)
        out = meta_model(H, C)

        if isinstance(out, (list, tuple)) and len(out) == 2:
            w, p_now = out
        else:
            p_now = out
            w = torch.full_like(p_now, 1.0 / p_now.shape[1])

        # Shapes absichern
        if p_now.ndim != 2 or p_now.shape[1] != K or w.shape != p_now.shape:
            # Fallback: einfacher Durchschnitt der Basismodelle
            p = preds_base.mean(axis=1)
            t = torch.tensor(p, dtype=torch.float32, device=device)
            t = _safe_sigmoid_logit_temp(t, temp_T)
            return t.cpu().numpy().astype(np.float32, copy=False)

        p_base = H[:, -1, :]                                     # (N,K)
        p_mix  = alpha_base_mix * p_base + (1.0 - alpha_base_mix) * p_now
        p_hat  = (w * p_mix).sum(dim=1)                          # (N,)
        p_hat  = _safe_sigmoid_logit_temp(p_hat, temp_T)

        # Degeneration: alles ≈ 0.5? → Fallback auf Base-Avg
        if torch.allclose(p_hat, torch.full_like(p_hat, 0.5), atol=1e-8):
            p_base_avg = p_base.mean(dim=1)                      # (N,)
            p_hat = _safe_sigmoid_logit_temp(p_base_avg, temp_T)

        return p_hat.float().detach().cpu().numpy().astype(np.float32, copy=False)

# ------------------ EA Tick → OHLC (Streaming, robust + mit Logging) ------------------
from typing import Tuple
import pandas as pd
import numpy as np
import logging

def _setup_local_logger(name: str = "ea"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger

def _pick_time_column(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    # 1) Kombi aus Date/Time?
    date_keys = [c for c in df.columns if str(c).lower() in ("date", "datum")]
    time_keys = [c for c in df.columns if str(c).lower() in ("time", "zeit")]
    if date_keys and time_keys:
        s = pd.to_datetime(
            df[date_keys[0]].astype(str).str.strip() + " " + df[time_keys[0]].astype(str).str.strip(),
            errors="coerce",
            utc=False,
        )
        return s, f"{date_keys[0]} + {time_keys[0]}"

    # 2) Übliche Einzelspalten
    candidates = [
        "Time", "time", "timestamp", "Timestamp", "Datetime", "DateTime",
        "date_time", "datetime", "ts", "TimeUTC", "time_utc"
    ]
    for c in candidates:
        if c in df.columns:
            s = df[c]
            if not pd.api.types.is_datetime64_any_dtype(s):
                s = pd.to_datetime(s, errors="coerce", utc=False)
            return s, c

    raise KeyError(
        f"Keine Zeitspalte gefunden. Verfügbare Spalten: {list(df.columns)}. "
        "Erwartet z.B. 'Time', 'timestamp', 'Datetime' oder 'Date'+'Time'."
    )

def _pick_price_series(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    # Bevorzugung: Mid (Bid/Ask) > Price/Last > Close
    if {"Bid", "Ask"}.issubset(df.columns):
        return (df["Bid"] + df["Ask"]) / 2.0, "mid(Bid,Ask)"
    for key in ["Price", "price", "Last", "last", "Close", "close"]:
        if key in df.columns:
            return df[key], key
    # Heuristik: erste numerische Spalte
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        return df[num_cols[0]], num_cols[0]
    raise KeyError(
        f"Keine Preis-Spalte gefunden. Verfügbare Spalten: {list(df.columns)}. "
        "Erwartet z.B. 'Bid'+'Ask', 'Price', 'Last' oder 'Close'."
    )

def _resample_ticks_with_carry(buf_df: pd.DataFrame, new_df: pd.DataFrame, freq: str):
    """
    Tick -> OHLC Bars mit Carry-Over der letzten (angebrochenen) Periode.
    Erzeugt Spalten: Time, Open, High, Low, Close, Volume
      - Preisquelle: bevorzugt 'Tick_Bid', dann 'Bid','Price','Last','Mid','Tick_Last'
      - Volumen: falls Spalte existiert (z.B. 'Tick_Volume','Volume','Qty','quantity','Size') -> Summe
                 sonst Fallback = Anzahl Ticks pro Bar (group.size()).
    """
    if new_df is None:
        new_df = pd.DataFrame()
    if buf_df is None:
        buf_df = pd.DataFrame()

    # zusammenführen
    if not buf_df.empty:
        df = pd.concat([buf_df, new_df], axis=0, ignore_index=True)
    else:
        df = new_df.copy()

    if df.empty:
        return pd.DataFrame(columns=["Time","Open","High","Low","Close","Volume"]), pd.DataFrame()

    # Zeitspalte finden
    t_candidates = ["Time", "time", "timestamp", "Datetime", "DateTime", "date"]
    tcol = next((c for c in t_candidates if c in df.columns), None)
    if tcol is None:
        raise KeyError(f"No time column found. Looked for: {t_candidates}")

    # Preisquelle finden
    p_candidates = ["Tick_Bid", "Bid", "Price", "Last", "Mid", "Tick_Last", "Ask", "Tick_Ask"]
    pcol = next((c for c in p_candidates if c in df.columns), None)
    if pcol is None:
        raise KeyError(f"No price column found. Looked for: {p_candidates}")

    # Volumenquelle (optional)
    v_candidates = ["Tick_Volume", "Volume", "Qty", "quantity", "Size", "Vol"]
    vcol = next((c for c in v_candidates if c in df.columns), None)

    # Zeit konvertieren & auf Buckets runden
    dft = df.copy()
    dft["Time"] = pd.to_datetime(dft[tcol], errors="coerce")
    dft = dft.dropna(subset=["Time"])
    if dft.empty:
        return pd.DataFrame(columns=["Time","Open","High","Low","Close","Volume"]), pd.DataFrame()

    dft["bucket"] = dft["Time"].dt.floor(freq)
    last_bucket_start = dft["bucket"].max()

    # "fertige" Buckets vs. Carry (letzter Bucket bleibt offen)
    done = dft[dft["bucket"] < last_bucket_start].copy()
    carry = dft[dft["bucket"] >= last_bucket_start].copy()

    if not done.empty:
        done = done.sort_values("Time")
        g = done.groupby("bucket", sort=True)
        o = g[pcol].first().rename("Open")
        h = g[pcol].max().rename("High")
        l = g[pcol].min().rename("Low")
        c = g[pcol].last().rename("Close")
        if vcol:
            vol = g[vcol].sum().rename("Volume")
        else:
            vol = g.size().rename("Volume")
        out = pd.concat([o, h, l, c, vol], axis=1).reset_index().rename(columns={"bucket": "Time"})
    else:
        out = pd.DataFrame(columns=["Time","Open","High","Low","Close","Volume"])

    # Logging
    if not out.empty:
        tr0, tr1 = out["Time"].iloc[0], out["Time"].iloc[-1]
        bars = len(out)
    else:
        tr0 = tr1 = None
        bars = 0

    print(
        f"[INFO] Resample: raw={len(new_df)}, buf={len(buf_df)}, "
        f"used_time='{tcol}', used_price='{pcol}', "
        f"range=[{tr0} → {tr1}], bars={bars}, carry_rows={len(carry)} "
        f"(last_bucket_start={last_bucket_start}, freq={freq})"
    )

    return out, carry

def _normalize_ohlc_casing(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        return df
    df = df.copy()

    # Upper -> lower
    if "Open"   in df.columns and "open"   not in df.columns: df["open"]   = df["Open"]
    if "High"   in df.columns and "high"   not in df.columns: df["high"]   = df["High"]
    if "Low"    in df.columns and "low"    not in df.columns: df["low"]    = df["Low"]
    if "Close"  in df.columns and "close"  not in df.columns: df["close"]  = df["Close"]
    if "Volume" in df.columns and "volume" not in df.columns: df["volume"] = df["Volume"]

    # lower -> Upper
    if "open"   in df.columns and "Open"   not in df.columns: df["Open"]   = df["open"]
    if "high"   in df.columns and "High"   not in df.columns: df["High"]   = df["high"]
    if "low"    in df.columns and "Low"    not in df.columns: df["Low"]    = df["low"]
    if "close"  in df.columns and "Close"  not in df.columns: df["Close"]  = df["close"]
    if "volume" in df.columns and "Volume" not in df.columns: df["Volume"] = df["volume"]

    return df

def _build_sequences_from_df(df_feat, cols: List[str], seq_len: int):
    Xmat = df_feat[cols].astype(np.float32).to_numpy()
    N = len(Xmat)
    if N <= seq_len:
        raise ValueError(f"Not enough rows after resample/enrich: N={N}, need > seq_len={seq_len}")
    T = seq_len; F = len(cols)
    X_seq = np.zeros((N - T, T, F), dtype=np.float32)
    for i in range(T, N):
        X_seq[i - T] = Xmat[i-T:i]
    return X_seq


def _labels_from_close_horizon(df_ohlc: pd.DataFrame, horizon_bars: int) -> np.ndarray:
    close = (df_ohlc["close"] if "close" in df_ohlc.columns else df_ohlc["Close"]).astype(float)
    nxt   = close.shift(-horizon_bars)
    y     = (nxt > close).astype(np.float32)
    y     = y.values
    return y


def _compute_da(y_true: np.ndarray, p: np.ndarray, thr: float = 0.5) -> float:
    pred = (p >= thr).astype(np.float32)
    return float((pred == y_true).mean()) if len(y_true) else float("nan")


def _compute_mda(y_true: np.ndarray, p: np.ndarray, band: float = 0.02) -> Tuple[float, float]:
    mask = np.abs(p - 0.5) > band
    if mask.sum() == 0:
        return float("nan"), 0.0
    acc = float(((p[mask] >= 0.5).astype(np.float32) == y_true[mask]).mean())
    coverage = float(mask.mean())
    return acc, coverage


# ------------------ Streaming-Pipeline ------------------
def _stream_predict_ea(args, cfg):
    from utils.features import enrich
    log = _setup_local_logger("ea")

    model_dir = Path(args.model_dir)
    device    = smart_device(args.device)

    # Modelle laden
    rf_list  = load_rf_list(model_dir)
    lgb_list = load_lgb_list(model_dir)
    xgb_list = load_xgb_list(model_dir)
    ft_model, _   = load_ft(model_dir, device, cfg)
    cnn_model, _  = load_cnn(model_dir, device, cfg)
    meta_model, _ = load_meta(model_dir, device, cfg)
    temp_T        = load_temp_scaler(model_dir)

    meta_cfg = cfg.get("meta", {})
    L        = args.L if args.L is not None else int(meta_cfg.get("L", 16))
    ctx_dim  = int(meta_cfg.get("ctx_dim", 0))
    alpha    = float(meta_cfg.get("alpha_base_mix", 1.0))

    cols     = cfg["data"]["numerical_cols"]
    seq_len  = int(cfg["training"].get("seq_len", 24))
    horizon_bars = int(round(args.label_horizon_min / _freq_to_minutes(args.freq)))

    out_path = Path(args.output)
    header_written = False

    ohlc_hist   : pd.DataFrame = pd.DataFrame()
    base_hist   : np.ndarray   = np.empty((0, 5), dtype=np.float32)
    tick_carry  : pd.DataFrame = pd.DataFrame()

    da_total = da_correct = 0
    mda_total = mda_correct = 0

    reader = pd.read_csv(args.ea_tick_file, chunksize=args.ea_chunk_rows, engine="python")

    def _mda_mask(pv: np.ndarray, band: float) -> np.ndarray:
        # Für band==0 sollen ALLE Punkte in die MDA eingehen (keine 0%-Coverage mehr)
        if band <= 0:
            return np.ones_like(pv, dtype=bool)
        return np.abs(pv - 0.5) > band

    for df_raw in reader:
        # (1) Tick → OHLC
        ohlc_chunk, tick_carry = _resample_ticks_with_carry(tick_carry, df_raw, args.freq)
        if ohlc_chunk.empty:
            continue

        # (2) History begrenzen
        need_hist = (seq_len - 1) + horizon_bars
        if len(ohlc_hist) > need_hist:
            ohlc_hist = ohlc_hist.iloc[-need_hist:].copy()

        # (3) EINMAL zusammenkleben
        ohlc_all = pd.concat([ohlc_hist, ohlc_chunk], axis=0, ignore_index=True)

        # (4) Index/Time + Casing
        if "Time" in ohlc_all.columns:
            ts = pd.to_datetime(ohlc_all["Time"], errors="coerce")
        elif "timestamp" in ohlc_all.columns:
            ts = pd.to_datetime(ohlc_all["timestamp"], errors="coerce")
        else:
            ts = pd.to_datetime(ohlc_all.index, errors="coerce")
        ohlc_all["timestamp"] = ts
        ohlc_all = ohlc_all.set_index("timestamp", drop=False)
        ohlc_all = _normalize_ohlc_casing(ohlc_all)

        # (5) Features/Labels
        feat_all = enrich(ohlc_all)
        if "Volume" in cols and "Volume" not in feat_all.columns:
            if "Volume" in ohlc_all.columns:
                feat_all["Volume"] = ohlc_all["Volume"].astype(np.float32)
            elif "volume" in ohlc_all.columns:
                feat_all["Volume"] = ohlc_all["volume"].astype(np.float32)
            else:
                feat_all["Volume"] = 0.0
        missing = [c for c in cols if c not in feat_all.columns]
        if missing:
            raise KeyError(f"Missing feature columns after enrich: {missing}")
        y_all = _labels_from_close_horizon(ohlc_all, horizon_bars)

        # (6) Fenster-Enden – **Off-by-1 Fix** am Chunk-Übergang
        if len(feat_all) <= seq_len + horizon_bars:
            ohlc_hist = ohlc_all.copy()
            continue
        first_end = max(seq_len - 1, len(ohlc_hist) - 1)  # << fix (statt len(ohlc_hist))
        last_end  = len(ohlc_all) - 1 - horizon_bars
        if last_end < first_end:
            ohlc_hist = ohlc_all.copy()
            continue
        j_first = first_end - (seq_len - 1)
        j_last  = last_end  - (seq_len - 1)
        M_emit  = j_last - j_first + 1

        log.info(
            f"Windowing: hist={len(ohlc_hist)}, chunk_bars={len(ohlc_chunk)}, "
            f"total={len(ohlc_all)}, seq_len={seq_len}, horizon_bars={horizon_bars}, "
            f"emit_windows={M_emit} (ends {first_end}→{last_end}), features={len(cols)}"
        )

        # (7) Sequenzen bauen (nur benötigte)
        X_all = feat_all[cols].astype(np.float32).to_numpy()
        X_emit = np.empty((M_emit, seq_len, len(cols)), dtype=np.float32)
        for k, e in enumerate(range(first_end, last_end + 1)):
            X_emit[k] = X_all[e - (seq_len - 1): e + 1]

        if "Time" in ohlc_all.columns:
            times_emit = ohlc_all["Time"].iloc[first_end : last_end + 1].astype(str).values
        else:
            times_emit = ohlc_all["timestamp"].iloc[first_end : last_end + 1].astype(str).values
        y_emit = y_all[first_end:last_end+1].astype(np.float32)

        # (8) Base-Preds
        X_flat = flatten_seq(X_emit)
        p_rf, p_lgb, p_xgb = predict_trees(rf_list, lgb_list, xgb_list, X_flat)
        p_ft  = predict_ft_batched(ft_model,  X_emit, device, args.nn_batch_size)
        p_cnn = predict_cnn_batched(cnn_model, X_emit, device, args.nn_batch_size)
        preds_base_chunk = np.vstack([p_rf, p_lgb, p_xgb, p_ft, p_cnn]).T  # (M_emit,5)

        # (9) Meta inkl. History
        seq_for_meta = np.vstack([base_hist, preds_base_chunk]) if len(base_hist) else preds_base_chunk
        p_meta_full  = predict_meta(meta_model, seq_for_meta, L=L, ctx_dim=ctx_dim,
                                    device=device, alpha_base_mix=alpha, temp_T=temp_T)
        p_meta_emit  = p_meta_full[-M_emit:]

        # Degeneration-Guard + Diagnose
        if np.allclose(p_meta_emit, 0.5, atol=1e-8):
            # Fallback auf Base-Avg dieses Chunks
            p_base_avg = preds_base_chunk.mean(axis=1).astype(np.float32)
            p_meta_emit = p_base_avg
            log.warning("Meta gab nur 0.5 zurück – falle auf Base-Avg pro Chunk zurück.")

        out = pd.DataFrame({
            "time":   times_emit,
            "label":  y_emit,
            "proba_meta": p_meta_emit,
            "proba_rf":   p_rf,
            "proba_lgb":  p_lgb,
            "proba_xgb":  p_xgb,
            "proba_ft":   p_ft,
            "proba_cnn":  p_cnn,
        })

        # (10) Metriken
        da_chunk = float("nan"); mda_chunk = float("nan"); cov_chunk = 0.0
        lbl_mask = ~np.isnan(out["label"].values)
        if lbl_mask.any():
            yv = out["label"].values[lbl_mask].astype(np.float32)
            pv = out["proba_meta"].values[lbl_mask].astype(np.float32)
            da_total += len(yv)
            da_correct += int(((pv >= 0.5).astype(np.float32) == yv).sum())

            m = _mda_mask(pv, args.mda_band)
            mda_total += int(m.sum())
            if m.any():
                mda_correct += int(((pv[m] >= 0.5).astype(np.float32) == yv[m]).sum())
                mda_chunk = (((pv[m] >= 0.5).astype(np.float32) == yv[m]).mean())
            cov_chunk = float(m.mean())
            da_chunk  = (((pv >= 0.5).astype(np.float32) == yv).mean())

        _write_chunk_csv(out, out_path, header=not header_written)
        header_written = True

        log.info(
            f"Emit: rows={len(out)}, time_range=[{times_emit[0]} → {times_emit[-1]}], "
            f"DA_chunk={da_chunk:.4f}, MDA_chunk(band={args.mda_band})={mda_chunk:.4f}, coverage={cov_chunk:.2%}"
        )

        # (11) History updaten
        base_hist = seq_for_meta[-(L-1):] if L > 1 else np.empty((0,5), dtype=np.float32)
        ohlc_hist = ohlc_all.iloc[-((seq_len - 1) + horizon_bars):].copy()

        del ohlc_chunk, ohlc_all, feat_all, X_all, X_emit, preds_base_chunk, p_meta_full
        gc.collect()

    # Letzten Tick-Carry noch resamplen & final verarbeiten
    if not tick_carry.empty:
        final_ohlc, _ = _resample_ticks_with_carry(pd.DataFrame(), tick_carry, args.freq)
        if not final_ohlc.empty:
            # History bereits begrenzt – daher nur EINMAL konkatenieren
            ohlc_all = pd.concat([ohlc_hist, final_ohlc], axis=0, ignore_index=True)
            if "Time" in ohlc_all.columns:
                ts = pd.to_datetime(ohlc_all["Time"], errors="coerce")
            elif "timestamp" in ohlc_all.columns:
                ts = pd.to_datetime(ohlc_all["timestamp"], errors="coerce")
            else:
                ts = pd.to_datetime(ohlc_all.index, errors="coerce")
            ohlc_all["timestamp"] = ts
            ohlc_all = ohlc_all.set_index("timestamp", drop=False)
            ohlc_all = _normalize_ohlc_casing(ohlc_all)

            if len(ohlc_all) >= seq_len + horizon_bars + 1:
                feat_all = enrich(ohlc_all)
                y_all = _labels_from_close_horizon(ohlc_all, horizon_bars)

                first_end = max(len(ohlc_hist), seq_len - 1)
                last_end  = len(ohlc_all) - 1 - horizon_bars
                if last_end >= first_end:
                    X_all = feat_all[cols].astype(np.float32).to_numpy()
                    M_emit = (last_end - first_end + 1)
                    X_emit = np.empty((M_emit, seq_len, len(cols)), dtype=np.float32)
                    for k, e in enumerate(range(first_end, last_end + 1)):
                        X_emit[k] = X_all[e - (seq_len - 1): e + 1]

                    if "Time" in ohlc_all.columns:
                        times_emit = ohlc_all["Time"].iloc[first_end : last_end + 1].astype(str).values
                    else:
                        times_emit = ohlc_all["timestamp"].iloc[first_end : last_end + 1].astype(str).values
                    y_emit     = y_all[first_end:last_end+1].astype(np.float32)

                    X_flat = flatten_seq(X_emit)
                    p_rf, p_lgb, p_xgb = predict_trees(rf_list, lgb_list, xgb_list, X_flat)
                    p_ft  = predict_ft_batched(ft_model,  X_emit, device, args.nn_batch_size)
                    p_cnn = predict_cnn_batched(cnn_model, X_emit, device, args.nn_batch_size)
                    preds_base_chunk = np.vstack([p_rf, p_lgb, p_xgb, p_ft, p_cnn]).T

                    seq_for_meta = np.vstack([base_hist, preds_base_chunk]) if len(base_hist) else preds_base_chunk
                    p_meta_full  = predict_meta(meta_model, seq_for_meta, L=L, ctx_dim=ctx_dim, device=device, alpha_base_mix=alpha, temp_T=temp_T)
                    p_meta_emit  = p_meta_full[-M_emit:]

                    out = pd.DataFrame({
                        "time":   times_emit,
                        "label":  y_emit,
                        "proba_meta": p_meta_emit,
                        "proba_rf":   p_rf,
                        "proba_lgb":  p_lgb,
                        "proba_xgb":  p_xgb,
                        "proba_ft":   p_ft,
                        "proba_cnn":  p_cnn,
                    })

                    da_chunk = float("nan"); mda_chunk = float("nan"); cov_chunk = 0.0
                    lbl_mask = ~np.isnan(out["label"].values)
                    if lbl_mask.any():
                        yv = out["label"].values[lbl_mask].astype(np.float32)
                        pv = out["proba_meta"].values[lbl_mask].astype(np.float32)
                        da_total += len(yv)
                        da_correct += int(((pv >= 0.5).astype(np.float32) == yv).sum())
                        m = np.abs(pv - 0.5) > args.mda_band
                        mda_total += int(m.sum())
                        if m.any():
                            mda_correct += int(((pv[m] >= 0.5).astype(np.float32) == yv[m]).sum())
                        da_chunk = ( (pv >= 0.5).astype(np.float32) == yv ).mean()
                        cov_chunk = float(m.mean())
                        mda_chunk = ( ((pv[m] >= 0.5).astype(np.float32) == yv[m]).mean() ) if m.any() else float("nan")

                    _write_chunk_csv(out, out_path, header=not header_written)
                    header_written = True
                    log.info(
                        f"Emit(final): rows={len(out)}, time_range=[{times_emit[0]} → {times_emit[-1]}], "
                        f"DA_chunk={da_chunk:.4f}, MDA_chunk(band={args.mda_band})={mda_chunk:.4f}, coverage={cov_chunk:.2%}"
                    )
                    del feat_all, X_all, X_emit, preds_base_chunk, p_meta_full
                    gc.collect()

    # Final Metrics
    if header_written and da_total > 0:
        da  = da_correct / da_total
        mda = (mda_correct / mda_total) if mda_total > 0 else float("nan")
        cov = (mda_total / da_total) if da_total > 0 else 0.0
        print(f"meta: DA={da:.4f} | MDA(band={args.mda_band})={mda:.4f} | coverage={cov:.2%}")
    print(f"Saved predictions → {out_path}")


def _freq_to_minutes(freq: str) -> float:
    """
    Grobe Umrechnung nach Minuten (z. B. '1min'→1, '5s'→5/60).
    """
    s = freq.strip().lower()
    if s.endswith("min"):
        return float(s.replace("min",""))
    if s.endswith("s"):
        return float(s[:-1]) / 60.0
    if s.endswith("h"):
        return float(s[:-1]) * 60.0
    # fallback: 1 min
    return 1.0


def main():
    parser = argparse.ArgumentParser(description="Ensemble + MetaMoE inference (memory-safe)")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--config",    type=str, required=True)
    parser.add_argument("--input",     type=str, default=None, help=".npy / .npz (N,T,F) – optional (nicht für EA-Streaming)")
    parser.add_argument("--output",    type=str, required=True)
    parser.add_argument("--device",    type=str, default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--L",         type=int, default=None, help="Override meta window length")
    # EA-Tick Streaming
    parser.add_argument("--ea-tick-file", type=str, default=None, help="EA Raw ticks: Time,Tick_Bid,Tick_Ask,Tick_Last,Tick_Volume")
    parser.add_argument("--freq",           type=str, default="1min", help="Resample Frequenz (z.B. 1min, 5s, 1h)")
    parser.add_argument("--label-horizon-min", type=float, default=60.0, help="Label-Horizont in Minuten (z.B. 60)")
    parser.add_argument("--mda-band",       type=float, default=0.02, help="Neutralband um 0.5 für MDA")
    parser.add_argument("--ea-chunk-rows",  type=int,   default=500_000, help="Zeilen pro Tick-Chunk")
    parser.add_argument("--nn-batch-size",  type=int,   default=8192,    help="Batchgröße für FT/CNN Inference")
    args = parser.parse_args()

    _try_import_models()
    cfg = load_yaml(Path(args.config))

    # Streaming-Pfad
    if args.ea_tick_file:
        _stream_predict_ea(args, cfg)
        return

    # Fallback: bereits vorbereitete Features (N,T,F) komplett laden
    if not args.input:
        raise ValueError("Entweder --ea-tick-file ODER --input (N,T,F) angeben.")
    X_seq = load_input(Path(args.input))

    model_dir = Path(args.model_dir)
    device    = smart_device(args.device)

    rf_list  = load_rf_list(model_dir)
    lgb_list = load_lgb_list(model_dir)
    xgb_list = load_xgb_list(model_dir)
    ft_model, _   = load_ft(model_dir, device, cfg)
    cnn_model, _  = load_cnn(model_dir, device, cfg)
    meta_model, _ = load_meta(model_dir, device, cfg)
    temp_T        = load_temp_scaler(model_dir)

    # Meta predictions
    meta_cfg = cfg.get("meta", {})
    L = args.L or int(meta_cfg.get("L", 16))
    alpha = float(meta_cfg.get("alpha_base_mix", 1.0))
    ctx_dim = int(meta_cfg.get("ctx_dim", 0))
    temp_T = load_temp_scaler(model_dir)

    X_flat = flatten_seq(X_seq)
    p_rf, p_lgb, p_xgb = predict_trees(rf_list, lgb_list, xgb_list, X_flat)
    p_ft  = predict_ft_batched(ft_model,  X_seq, device, args.nn_batch_size)
    p_cnn = predict_cnn_batched(cnn_model, X_seq, device, args.nn_batch_size)

    preds_base = np.vstack([p_rf, p_lgb, p_xgb, p_ft, p_cnn]).T   # (N,5)
    p_meta = predict_meta(meta_model, preds_base, L=L, ctx_dim=ctx_dim,
                          device=device, alpha_base_mix=alpha, temp_T=temp_T)

    out = pd.DataFrame({
        "proba_meta": p_meta,
        "proba_rf":   p_rf,
        "proba_lgb":  p_lgb,
        "proba_xgb":  p_xgb,
        "proba_ft":   p_ft,
        "proba_cnn":  p_cnn,
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved predictions → {args.output} (N={len(out)})")


if __name__ == "__main__":
    main()
