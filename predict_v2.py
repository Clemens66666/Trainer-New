# predict_v2.py
# -*- coding: utf-8 -*-
"""
Variante 2: robuster Streaming-Predictor mit sehr ausführlichen Logs.
- Tick->OHLC-Resample (mit Carry über Chunk-Grenzen)
- Fenster-/Horizon-Slices ohne Off-by-One
- FT/CNN/Trees + optionales Meta (MoE/Transformer) mit fehlerrobustem Laden
- CSV-Streaming-Ausgabe mit konsistenten Längen
"""

from __future__ import annotations
import argparse, os, sys, json, gc, math, time, traceback, contextlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


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

# ==== FEATURE DEFINITION FOR CNN (must match training) ====
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
    # robust, rein pandas
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = (roll_up / (roll_down + 1e-12)).fillna(0.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return (rsi / 100.0).astype(np.float32)  # normalisiert auf [0,1]

def enrich_for_cnn(ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    Erwartet OHLCV mit Index=Datetime und Spalten: open, high, low, close, volume
    Liefert DataFrame mit exakt FEATURE_ORDER_CNN in dieser Reihenfolge.
    """
    df = ohlc.copy()
    df = _add_time_feats(df)

    # Indikatoren auf 'close'
    c = df["close"].astype(float)
    df["sma_10"] = c.rolling(10, min_periods=1).mean().astype(np.float32)
    df["ema_20"] = c.ewm(span=20, adjust=False).mean().astype(np.float32)
    df["rsi_14"] = _rsi(c, 14).astype(np.float32)

    # Nur die 10 Features in richtiger Reihenfolge, fehlende auffüllen
    for col in FEATURE_ORDER_CNN:
        if col not in df.columns:
            df[col] = 0.0
    df = df[FEATURE_ORDER_CNN].astype(np.float32)
    return df

def make_windows_from_features(df_feat: pd.DataFrame, seq_len: int, horizon_bars: int):
    """
    Baut Sliding Windows:
      - X_seq:  [N, L, F]   (für CNN/FT)
      - X_last: [N, F]      (für FT)
      - X_flat: [N, L*F]    (für Baumbasierte, falls genutzt)
      - emit_index: Index der Zielzeilen (Ende jedes Fensters)
    """
    F = df_feat.shape[1]
    arr = df_feat.values  # [T, F]
    T = arr.shape[0]
    L = int(seq_len)
    H = int(max(1, horizon_bars))  # z.B. 1 bei 60min horizon_min=60

    if T < L + H:
        return np.empty((0, L, F), np.float32), np.empty((0, F), np.float32), np.empty((0, L*F), np.float32), df_feat.index[:0]

    # Emitpunkte so, dass wir H Schritte in die Zukunft schauen (hier nur für Labels; Pred ist auf "jetzt")
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

def _gpu_mem_str() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    return (
        f"GPU0 used={_gpu_mem('used')} reserved={_gpu_mem('reserved')} "
        f"free={_gpu_mem('free')} / total={_gpu_mem('total')}"
    )

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
# Wir versuchen alle relevanten Klassen aus deinem Trainer zu holen.
# Falls SimpleCNN/MetaMoE nicht vorhanden sind, liefern wir Fallback-Klassen.
try:
    from trainers.hybrid_longtrend_trainer import FTWrapped
except Exception:
    FTWrapped = None

try:
    from trainers.hybrid_longtrend_trainer import SimpleCNN as TrainerCNN
except Exception:
    TrainerCNN = None

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

# ========= SimpleCNN wrapper (shape: [B, C, L]) =========
class SimpleCNN(nn.Module):
    def __init__(self, in_chans: int, n_filters: int = 64, kernel_size: int = 3):
        super().__init__()
        self.in_chans = in_chans
        self.n_filters = n_filters
        self.block1 = nn.Sequential(
            nn.Conv1d(in_chans, n_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_filters),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Linear(n_filters, 1)

    def forward(self, x):  # x: [B, C, L]
        h = self.block1(x)
        h = self.block2(h).squeeze(-1)  # [B, n_filters]
        out = torch.sigmoid(self.head(h)).squeeze(-1)  # [B]
        return out

def _read_best_params_json(model_dir: Path, log):
    bp = (model_dir / "best_params.json")
    if not bp.exists():
        return {}
    try:
        import json
        d = json.loads(bp.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception as e:
        log.warning(f"[CNN][HP] best_params.json konnte nicht gelesen werden: {e}")
        return {}


# ============================ I/O & RESAMPLE ================================
def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns: return c
        if c.lower() in cols: return cols[c.lower()]
    return None

def _read_tick_chunks(path: str, chunksize: int, log):
    use_cols = ["Time","Tick_Bid","Tick_Ask","Tick_Last","Tick_Volume"]
    dtypes   = {"Tick_Bid":float, "Tick_Ask":float, "Tick_Last":float, "Tick_Volume":float}
    try:
        it = pd.read_csv(path, chunksize=chunksize, usecols=use_cols, dtype=dtypes)
        for ch in it:
            yield ch
    except ValueError:
        # Fallback: alle Spalten einlesen
        it = pd.read_csv(path, chunksize=chunksize)
        for ch in it:
            yield ch

def _resample_ticks_with_carry(
    carry: pd.DataFrame, df_raw: pd.DataFrame, freq: str, log
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Kombiniert carry+df_raw, bildet OHLC-Buckets (bis *vor* dem letzten Bucket),
    und liefert den letzten unvollständigen Bucket als neuer carry zurück.
    """
    t0 = time.time()
    df = pd.concat([carry, df_raw], axis=0, ignore_index=True)

    tcol = _find_col(df, ["Time","time","timestamp","Datetime","date","Date"])
    pcol = _find_col(df, ["Tick_Bid","Bid","Price","Mid","Last","Close","tick_bid"])
    if tcol is None:
        raise KeyError("Zeitspalte nicht gefunden (z.B. 'Time').")
    if pcol is None:
        raise KeyError("Preisspalte nicht gefunden (z.B. 'Tick_Bid').")

    df = df[[tcol, pcol]].copy()
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce", utc=False)
    df = df.dropna(subset=[tcol, pcol])
    if df.empty:
        return pd.DataFrame(columns=["Time","Open","High","Low","Close","Volume"]), pd.DataFrame(columns=df.columns)
    df = df.sort_values(tcol).set_index(tcol)

    bucket = df.index.floor(freq)
    df["_bucket"] = bucket
    last_bucket = df["_bucket"].iloc[-1]
    mask_full = df["_bucket"] != last_bucket
    df_full   = df.loc[mask_full]
    df_carry  = df.loc[~mask_full].drop(columns=["_bucket"])

    if df_full.empty:
        log.debug("[Resample] Kein vollständiger Bucket; alles Carry.")
        return pd.DataFrame(columns=["Time","Open","High","Low","Close","Volume"]), df_carry.reset_index()

    agg = df_full.groupby("_bucket")[pcol].agg(["first","max","min","last","count"])
    agg = agg.rename(columns={"first":"Open","max":"High","min":"Low","last":"Close","count":"Volume"})
    agg.index.name = "Time"
    ohlc = agg.reset_index()

    log.info(
        f"[Resample] raw={len(df_raw):,}, buf={len(carry):,}, used_time='{tcol}', used_price='{pcol}', "
        f"range=[{ohlc['Time'].iloc[0]} → {ohlc['Time'].iloc[-1]}], bars={len(ohlc):,}, "
        f"carry_rows={len(df_carry):,} (freq={freq}) | {_gpu_mem_str()} | took {time.time()-t0:.3f}s"
    )
    return ohlc, df_carry.reset_index()

def _freq_to_minutes(freq: str) -> int:
    f = freq.lower().strip()
    if f.endswith("min"):
        return int(f.replace("min",""))
    if f.endswith("h"):
        return 60*int(f[:-1])
    if f.endswith("s"):
        return max(1, int(int(f[:-1])/60))
    try:
        return int(f)
    except Exception:
        return 1

def _build_windows_idx(n_rows: int, seq_len: int, horizon_bars: int, hist_len: int) -> Tuple[int,int,slice,slice]:
    last_end  = n_rows - horizon_bars - 1
    first_end = max(seq_len - 1, hist_len - 1)
    if last_end < first_end:
        return -1, -2, slice(0,0), slice(0,0)
    y_slice = slice(first_end + horizon_bars, last_end + horizon_bars + 1)
    t_slice = slice(first_end,               last_end + 1)
    return first_end, last_end, y_slice, t_slice

def _make_windows(ohlc: pd.DataFrame, seq_len: int, horizon_bars: int, hist_len: int, log):
    """
    Macht gleitende Fenster auf OHLC.
    Rückgabe: X_seq [M, T, F], X_last [M, F], ends_idx [M]
    Features: [Open,High,Low,Close,Volume] → F=5
    """
    if ohlc.empty:
        return np.empty((0, seq_len, 5), np.float32), np.empty((0,5), np.float32), np.array([], dtype=int)

    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in ohlc.columns]
    if len(cols) < 4:
        raise ValueError(f"OHLC unvollständig: Spalten gefunden={ohlc.columns.tolist()}")

    X = ohlc[cols].to_numpy(dtype=np.float32, copy=False)
    F = X.shape[1]
    first_end, last_end, y_slice, t_slice = _build_windows_idx(len(X), seq_len, horizon_bars, hist_len)
    if last_end < first_end:
        log.info("[Windowing] Zu wenig Bars für Fensterung.")
        return np.empty((0, seq_len, F), np.float32), np.empty((0,F), np.float32), np.array([], dtype=int)

    M = last_end - first_end + 1
    T = seq_len
    X_seq = np.empty((M, T, F), dtype=np.float32)
    ends  = np.empty((M,), dtype=int)

    for i, e in enumerate(range(first_end, last_end+1)):
        s = e - (T - 1)
        X_seq[i] = X[s:e+1]
        ends[i]  = e

    X_last = X[t_slice]
    log.info(f"[Windowing] hist={hist_len}, chunk_bars={len(X)}, total={len(X)}, seq_len={seq_len}, horizon_bars={horizon_bars}")
    log.info(f"[Emit] ends {ends[0]}→{ends[-1]} → rows={len(ends)}")
    return X_seq, X_last, ends

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

def _load_torch_state(path: Path, log) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = torch.load(str(path), map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            log.info(f"[LOAD] Torch-Container mit 'state_dict' in {path.name}")
            return obj["state_dict"]
        if isinstance(obj, dict):
            log.info(f"[LOAD] Torch state_dict (dict) aus {path.name}")
            return obj
        if isinstance(obj, nn.Module):
            log.info(f"[LOAD] Torch nn.Module aus {path.name} → state_dict() genutzt")
            return obj.state_dict()
        log.warning(f"[LOAD] Unerwarteter Torch-Typ in {path.name}: {type(obj)}")
        return None
    except Exception as e:
        log.error(f"[LOAD] {path.name} konnte nicht geladen werden: {e}")
        return None
# ========= META =========
try:
    from trainers.hybrid_longtrend_trainer import MetaMoE  # bevorzugte Quelle
    META_SRC = "trainer.MetaMoE"
except Exception:
    META_SRC = "local.MetaMoE"
    class MetaMoE(nn.Module):
        # Minimaler Fallback – nur falls Import fehlschlägt (gleiche Signatur)
        def __init__(self, K:int, L:int, ctx_dim:int, d_model:int, n_heads:int=1, dropout:float=0.1):
            super().__init__()
            self.K, self.L, self.ctx_dim = K, L, ctx_dim
            self.ctx_proj = nn.Linear(ctx_dim, d_model) if ctx_dim>0 else nn.Identity()
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=max(1,n_heads), dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
            self.head_w   = nn.Linear(d_model, K)
            self.head_now = nn.Linear(d_model, K)

        def forward(self, H, C):
            # H:[B,L,K] → zuerst pro Modell Feature, dann über Zeit mappen
            B,L,K = H.shape
            x = H  # [B,L,K]
            # einfacher Trick: pro Zeit Schritt Mittel über K bilden, concat mit ctx
            x = x.mean(dim=2)  # [B,L]
            x = x.unsqueeze(-1)  # [B,L,1]
            if not isinstance(self.ctx_proj, nn.Identity):
                c = self.ctx_proj(C).unsqueeze(1).expand(-1, self.L, -1)  # [B,L,d_model]
            else:
                c = torch.zeros((B,L,x.shape[-1]), device=x.device, dtype=x.dtype)
            z = self.encoder(c)
            w = torch.softmax(self.head_w(z[:, -1, :]), dim=-1)
            p = torch.sigmoid(self.head_now(z[:, -1, :]))
            return w, p


def combine_base_and_meta(base_preds: dict, meta_model, meta_info: dict,
                          L:int, device, batch_size:int, log):
    """
    Gibt finale Probas (p_final) zurück. Fallback: ungwichteter Base-Average.
    """
    valid = [(k, np.asarray(v, np.float32)) for k,v in base_preds.items() if v is not None]
    if not valid:
        return np.full((0,), 0.5, np.float32)
    M = min(len(v) for _,v in valid)
    if any(len(v)!=M for _,v in valid):
        valid = [(k, v[:M]) for k,v in valid]
    base_avg = np.mean(np.stack([v for _,v in valid], axis=1), axis=1).astype(np.float32)

    if meta_model is None:
        log.info(f"[COMBINE] Meta nicht verfügbar → Base-Average (M={M}, K={len(valid)})")
        return base_avg

    base_trim = dict(valid)
    # emit_times muss vom Call-Site übergeben werden → wir lesen ihn aus meta_info
    emit_times = meta_info.get("emit_times", None)
    if emit_times is None:
        log.warning("[META] emit_times fehlen → Base-Average")
        return base_avg

    # (alte H/C-Erstellung entfernt – wir nutzen nur noch build_meta_inputs_from_base an anderer Stelle)
    H, C, _keys = build_meta_inputs_from_base(base_trim, L=L, log=log,
                                              target_ctx_dim=getattr(meta_model, "expected_ctx_dim", None))
    if H is None:
        log.info("[COMBINE] H/C nicht gebaut → Base-Average")
        return base_avg

    p_meta = predict_meta_batched(meta_model, H, C, device, batch_size, log)
    m = min(len(p_meta), len(base_avg))
    p_final = np.clip(0.8*p_meta[:m] + 0.2*base_avg[:m], 1e-7, 1-1e-7).astype(np.float32)
    return p_final

class _LazyFT(nn.Module):
    """
    Baut FTWrapped erst beim ersten Aufruf (wenn D bekannt).
    """
    def __init__(self, model_dir: Path, state: Dict[str, Any], device: torch.device, log):
        super().__init__()
        self.model_dir = model_dir
        self.state     = state
        self.device    = device
        self.log       = log
        self.model     = None
        self.ready     = False

    def _build(self, D: int):
        self.log.debug(f"[FT] Build with D={D}")
        if FTWrapped is None:
            raise RuntimeError("FTWrapped nicht importierbar (trainer).")
        try:
            import rtdl
        except Exception as e:
            raise RuntimeError(f"[FT] rtdl Import-Fehler: {e}")

        # n_blocks aus ft_study.pkl (optional)
        n_blocks = 3
        pkl = self.model_dir / "ft_study.pkl"
        if pkl.exists() and joblib is not None:
            try:
                study = joblib.load(pkl)
                if hasattr(study, "best_params"):
                    n_blocks = int(study.best_params.get("n_blocks", n_blocks))
                    self.log.info(f"[FT] n_blocks from study: {n_blocks}")
            except Exception as e:
                self.log.warning(f"[FT] ft_study.pkl unreadable: {e}")

        ft_base = rtdl.FTTransformer.make_default(
            n_num_features=D, cat_cardinalities=(), d_out=1, n_blocks=n_blocks
        )

        # Optional LoRA
        try:
            from peft import LoraConfig, get_peft_model
            lcfg = LoraConfig(r=4, lora_alpha=16, lora_dropout=0.05,
                              target_modules=['ffn.linear_first'])
            ft_base = get_peft_model(ft_base, lcfg)
            self.log.info("[FT] PEFT/LoRA aktiviert.")
        except Exception as e:
            self.log.info(f"[FT] PEFT nicht aktiv: {e}")

        model = FTWrapped(ft_base).to(self.device).eval()

        # strict versuchen → Fallback
        try:
            model.load_state_dict(self.state, strict=False)
            self.log.info("[FT] state_dict STRICT geladen.")
        except Exception as e:
            self.log.warning(f"[FT] strict=True fehlgeschlagen ({e}) → strict=False")
            ik = model.load_state_dict(self.state, strict=True)
            miss = getattr(ik, "missing_keys", [])
            unex = getattr(ik, "unexpected_keys", [])
            if miss:
                self.log.warning(f"[FT] Missing keys: {miss}")
            if unex:
                self.log.warning(f"[FT] Unexpected keys: {unex}")

        self.model = model
        self.ready = True

    @torch.no_grad()
    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        if not self.ready:
            self._build(D=x_num.shape[1])
        return self.model(x_num)

def load_ft(model_dir: Path, device: torch.device, log):
    st = _load_torch_state(model_dir / "ft.pt", log)
    if st is None:
        log.warning("[FT] Kein ft.pt gefunden → FT deaktiviert.")
        return None
    return _LazyFT(model_dir, st, device, log)

# --- PATCH: Helper zum Ermitteln der erwarteten Eingangs-Kanäle aus einem state_dict
def _first_conv1d_in_channels_from_state_dict(state_dict) -> Optional[int]:
    """Finde den ersten 3D-Conv-Weight (out, in, k) und gib in_channels zurück."""
    for k, v in (state_dict.items() if isinstance(state_dict, dict) else []):
        if torch.is_tensor(v) and v.ndim == 3:
            # shape = [out_channels, in_channels, kernel]
            return int(v.shape[1])
    return None

# --- PATCH: CNN richtig laden (inkl. best_params + Auto-Anpassung der Kanalzahl)
def load_cnn_model(model_dir: Path, n_feat: int, device: torch.device, log):
    """
    Lädt SimpleCNN/TrainerCNN. Liest:
      - n_filters aus best_params.json['cnn']['n_filters'] (fallback=32)
      - erwartete in_channels direkt aus dem Checkpoint (falls abweichend von n_feat).
    Setzt model.expected_in_channels = tatsächlich erwartete Kanalzahl (für die Inferenz).
    """
    model_dir = Path(model_dir)
    bp_path = model_dir / "best_params.json"
    ckpt_candidates = [model_dir / "cnn.pt", model_dir / "cnn.pth", model_dir / "cnn_state.pt"]

    n_filters = 32
    try:
        if bp_path.exists():
            with open(bp_path, "r", encoding="utf-8") as f:
                bp = json.load(f)
            if isinstance(bp, dict) and "cnn" in bp and "n_filters" in bp["cnn"]:
                n_filters = int(bp["cnn"]["n_filters"])
        log.info(f"[CNN][HP] n_filters={n_filters} (aus best_params.json)")
    except Exception as e:
        log.warning(f"[CNN][HP] best_params.json nicht nutzbar → n_filters={n_filters} (default). ({e})")

    # Checkpoint suchen
    ckpt_path = None
    for c in ckpt_candidates:
        if c.exists():
            ckpt_path = c
            break
    if ckpt_path is None:
        log.warning("[CNN][LOAD] Kein Checkpoint gefunden → CNN deaktiviert.")
        return None

    # State zuerst laden, um in_channels zu ermitteln
    try:
        state_raw = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_raw, dict) and "state_dict" in state_raw:
            sd = state_raw["state_dict"]
        elif isinstance(state_raw, dict) and "model_state" in state_raw:
            sd = state_raw["model_state"]
        else:
            # ggf. Prefixe entfernen
            sd = None
            if isinstance(state_raw, dict):
                for pref in ("module.", "model.", "cnn."):
                    sub = {k[len(pref):]: v for k, v in state_raw.items()
                           if isinstance(k, str) and k.startswith(pref)}
                    if sub:
                        sd = sub
                        break
                if sd is None:
                    sd = state_raw
            else:
                sd = state_raw
    except Exception as e:
        log.error(f"[CNN][LOAD] Checkpoint konnte nicht gelesen werden: {e}")
        return None

    # Erwartete Kanalzahl aus dem Checkpoint ableiten
    exp_in = _first_conv1d_in_channels_from_state_dict(sd) or int(n_feat)
    if exp_in != n_feat:
        log.warning(f"[CNN][LOAD] Kanalzahl aus Checkpoint: expected_in={exp_in}, aktuell F={n_feat} → Modell auf {exp_in} Kanäle bauen (Input wird zur Laufzeit gemappt).")

    # Modell instanziieren
    CNNCls = TrainerCNN if 'TrainerCNN' in globals() and TrainerCNN is not None else SimpleCNN
    model = CNNCls(n_feat=exp_in, n_filters=n_filters).to(device)

    # Gewichte laden (tolerant)
    try:
        ik = model.load_state_dict(sd, strict=False)
        miss = getattr(ik, "missing_keys", [])
        unex = getattr(ik, "unexpected_keys", [])
        if miss:
            log.warning(f"[CNN][LOAD] Missing keys: {list(miss)}")
        if unex:
            log.warning(f"[CNN][LOAD] Unexpected keys: {list(unex)}")
        model.eval()
        model.expected_in_channels = int(exp_in)  # <- merken für predict
        pcount = sum(p.numel() for p in model.parameters())
        log.info(f"[CNN][LOAD] Geladen aus '{ckpt_path.name}' | params={pcount:,} | expected_in={exp_in}")
        return model
    except Exception as e:
        log.error(f"[CNN][LOAD] Fehler beim Laden: {e}")
        return None

# ============================ INFERENCE HELPERS ============================
def _extract_logits_tensor(raw_out) -> torch.Tensor:
    """
    Robust gegen verschiedene Rückgabetypen:
    - Tensor
    - dict mit Keys: logits/logit/pred/preds/out/output/y
    - tuple/list → erster Tensor
    """
    _log = get_logger("predict_v2")
    if torch.is_tensor(raw_out):
        return raw_out
    if isinstance(raw_out, dict):
        for k in ("logits", "logit", "pred", "preds", "out", "output", "y"):
            v = raw_out.get(k, None)
            if torch.is_tensor(v):
                _log.debug(f"[FT] Output=dict → benutze Key '{k}' ({tuple(v.shape)})")
                return v
        for k, v in raw_out.items():
            if torch.is_tensor(v):
                _log.debug(f"[FT] Output=dict → benutze ersten Tensor bei Key '{k}' ({tuple(v.shape)})")
                return v
        raise TypeError(f"[FT] dict-output ohne Tensor: keys={list(raw_out.keys())}")
    if isinstance(raw_out, (list, tuple)):
        for v in raw_out:
            if torch.is_tensor(v):
                _log.debug(f"[FT] Output=list/tuple → benutze erstes Tensor-Element ({tuple(v.shape)})")
                return v
        raise TypeError("[FT] list/tuple-output ohne Tensor.")
    raise TypeError(f"[FT] Unerwartiger Output-Typ: {type(raw_out)}")

@torch.no_grad()
def predict_ft_batched(
    ft_model,
    X_last: np.ndarray,
    device: torch.device,
    batch_size: int,
    amp_dtype: Optional[torch.dtype],
    log,
) -> np.ndarray:
    if ft_model is None or X_last is None or len(X_last) == 0:
        return np.empty((0,), dtype=np.float32)

    M, F = X_last.shape
    out = np.empty((M,), dtype=np.float32)
    ft_model.eval()
    use_cuda_amp = device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16)

    bs = max(1, int(batch_size))
    for s in range(0, M, bs):
        e = min(M, s + bs)
        xb = torch.from_numpy(X_last[s:e]).to(device, non_blocking=True)
        try:
            if use_cuda_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    pred = ft_model(xb)
            else:
                pred = ft_model(xb)
            logits = _extract_logits_tensor(pred).reshape(-1)
            probs  = torch.sigmoid(logits).float().cpu().numpy()
            out[s:e] = probs.astype(np.float32)
        except Exception as ex:
            log.error(f"[FT] Inferenzfehler @ batch {s}:{e}: {ex}", exc_info=True)
            out[s:e] = 0.5
    return out

def _z_norm_per_sample(X_seq: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = X_seq.mean(axis=1, keepdims=True)
    sd = X_seq.std(axis=1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return ((X_seq - mu) / sd).astype(np.float32, copy=False)

# --- PATCH: Helper um erwartete in_channels aus dem Modell zu holen
def _cnn_in_channels(cnn_model: nn.Module) -> Optional[int]:
    if hasattr(cnn_model, "expected_in_channels"):
        return int(cnn_model.expected_in_channels)
    # Fallback: ersten Conv1d suchen
    for m in cnn_model.modules():
        if isinstance(m, nn.Conv1d):
            return int(m.in_channels)
    return None


# --- PATCH: CNN Inferenz mit Kanal-Mapping
@torch.no_grad()
def predict_cnn_batched(cnn_model: nn.Module,
                        X_seq: np.ndarray,   # [N,T,F_curr]
                        device: torch.device,
                        batch_size: int,
                        log) -> np.ndarray:
    if cnn_model is None or X_seq is None or len(X_seq) == 0:
        return np.empty((0,), dtype=np.float32)

    N, T, F = X_seq.shape
    out = np.empty((N,), dtype=np.float32)

    # Z-Norm pro Sample
    Xn = _z_norm_per_sample(X_seq)  # [N,T,F]

    # Erwartete Kanäle des Modells
    exp_C = _cnn_in_channels(cnn_model) or F
    if exp_C != F:
        if exp_C > F:
            pad = exp_C - F
            log.warning(f"[CNN] Channel-Mismatch: Input F={F} < expected {exp_C} → Pad mit {pad} Nulllayer.")
            Xn = np.concatenate([Xn, np.zeros((N, T, pad), dtype=Xn.dtype)], axis=2)
        else:
            log.warning(f"[CNN] Channel-Mismatch: Input F={F} > expected {exp_C} → Truncate auf erste {exp_C} Kanäle.")
            Xn = Xn[:, :, :exp_C]
        F = exp_C  # aktualisieren

    bs = max(1, int(batch_size))
    use_cuda_amp = (device.type == "cuda")
    autocast_ctx = (lambda: torch.cuda.amp.autocast()) if use_cuda_amp else contextlib.nullcontext

    for s in range(0, N, bs):
        e = min(N, s + bs)
        xb = torch.tensor(Xn[s:e], dtype=torch.float32, device=device).permute(0, 2, 1)  # [B,C=Tfeat,F]-> [B,F,T]→hier [B,exp_C,T]
        try:
            with autocast_ctx():
                logits = cnn_model(xb)  # [B]
            p = torch.sigmoid(logits).float().detach().cpu().numpy().reshape(-1)
            out[s:e] = np.clip(p, 1e-7, 1 - 1e-7).astype(np.float32)
        except Exception as ex:
            log.error(f"[CNN] Inferenzfehler @ batch {s}:{e}: {ex}")
            out[s:e] = 0.5

    log.debug(f"[CNN] predict: in={X_seq.shape} → mapped={(N, T, F)} → out={out.shape}")
    return out

def _expected_features(model) -> Optional[int]:
    # sklearn
    if hasattr(model, "n_features_in_"):
        try: return int(model.n_features_in_)
        except Exception: pass
    # LightGBM Booster
    if "lightgbm" in type(model).__module__.lower() and hasattr(model, "num_feature"):
        try: return int(model.num_feature())
        except Exception: pass
    # XGBoost Booster
    if "xgboost" in type(model).__module__.lower() and hasattr(model, "num_features"):
        try: return int(model.num_features())
        except Exception: pass
    return None

def _prepare_X_for_model(model, X_last: Optional[np.ndarray], X_flat: Optional[np.ndarray]) -> np.ndarray:
    exp = _expected_features(model)
    cand = [arr for arr in (X_last, X_flat) if arr is not None]
    if not cand:
        raise RuntimeError("keine Eingaben für Tree-Modelle verfügbar")

    if exp is not None:
        for arr in cand:
            if arr.shape[1] == exp:
                return arr
        # Sonst: nähesten nehmen und pad/trim
        base = None
        if X_flat is not None and X_flat.shape[1] < exp:
            base = X_flat
        elif X_last is not None:
            base = X_last
        else:
            base = cand[0]
        if base.shape[1] < exp:
            pad = np.zeros((base.shape[0], exp - base.shape[1]), dtype=base.dtype)
            return np.concatenate([base, pad], axis=1)
        else:
            return base[:, :exp]
    # Keine Info → X_flat bevorzugen
    return X_flat if X_flat is not None else X_last

def _predict_trees(models: List[Any], X_last: np.ndarray, X_flat: np.ndarray, name: str, log) -> Optional[np.ndarray]:
    if not models:
        return None
    M = X_last.shape[0] if X_last is not None else X_flat.shape[0]
    preds = []
    any_success = False
    for i, m in enumerate(models, start=1):
        try:
            X_use = _prepare_X_for_model(m, X_last, X_flat)
            if "xgboost" in type(m).__module__.lower():
                import xgboost as xgb
                dm = xgb.DMatrix(X_use)
                proba = m.predict(dm).reshape(-1)
                if proba.min() < 0 or proba.max() > 1:
                    proba = 1.0 / (1.0 + np.exp(-proba))
            else:
                if hasattr(m, "predict_proba"):
                    proba = m.predict_proba(X_use)
                    proba = proba[:, 1] if proba.ndim == 2 else np.asarray(proba).reshape(-1)
                elif hasattr(m, "predict"):
                    proba = np.asarray(m.predict(X_use)).reshape(-1)
                    if proba.min() < 0 or proba.max() > 1:
                        proba = 1.0 / (1.0 + np.exp(-proba))
                elif hasattr(m, "decision_function"):
                    df = np.asarray(m.decision_function(X_use)).reshape(-1)
                    proba = 1.0 / (1.0 + np.exp(-df))
                else:
                    raise RuntimeError("Kein predict_proba/decision_function")
            if len(proba) != M:
                raise RuntimeError(f"Längenmismatch ({len(proba)} != {M})")
            preds.append(proba.astype(np.float32))
            any_success = True
        except Exception as ex:
            log.warning(f"[{name}] Vorhersage fehlgeschlagen: {ex}")
    if not any_success:
        return None
    return np.mean(np.stack(preds, axis=1), axis=1).astype(np.float32)

# ============================ META LOADER/PRED =============================

# --- D2: Kanonischer Meta-Loader ----------------------------------------
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

def load_meta_model(model_dir: Path, K: int, L: int, ctx_dim: int, device: torch.device, log):
    """
    Baut das Meta-Modell ROBUST:
      - K = Anzahl Basismodelle (für H und die K-Köpfe)
      - ctx_dim wird NICHT aus dem Checkpoint "erraten", sondern = K gesetzt
      - erwartete Kontext-Dimension wird NACH dem Bauen via meta.ctx_proj.in_features gelesen
    """
    MetaClass, src_name = _resolve_meta_class(log)
    if MetaClass is None:
        return None

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

    n_heads = _pick_n_heads(d_model)

    # Checkpoint finden & laden
    ckpt = None
    for name in ("meta.pt", "meta.pth", "meta_state.pt"):
        p = Path(model_dir) / name
        if p.exists():
            ckpt = p
            break
    if ckpt is None:
        log.info("[META][LOAD] Kein meta.pt gefunden → Meta deaktiviert.")
        return None

    try:
        state = torch.load(ckpt, map_location="cpu")
        sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    except Exception as e:
        log.error(f"[META][LOAD] Checkpoint konnte nicht gelesen werden: {e} → Meta deaktiviert.")
        return None

    # Modell mit ctx_dim = K bauen (keine Heuristik aus dem Checkpoint!)
    try:
        try:
            meta = MetaClass(K=K, L=L, ctx_dim=K, d_model=d_model, n_heads=n_heads,
                             n_layers=1, dropout=dropout).to(device)
        except TypeError:
            # Fallback, falls MetaClass keine n_layers/ dropout-Keys o.ä. hat
            meta = MetaClass(K=K, L=L, ctx_dim=K, d_model=d_model, n_heads=n_heads).to(device)
    except Exception as e:
        log.error(f"[META][BUILD] Instanziierung fehlgeschlagen: {e}")
        return None

    # erwartete Kontext-Dimension aus dem tatsächlich gebauten Modell lesen
    exp_ctx = None
    if hasattr(meta, "ctx_proj") and isinstance(meta.ctx_proj, nn.Linear):
        exp_ctx = int(meta.ctx_proj.in_features)
    else:
        exp_ctx = K  # konservativ

    meta.expected_ctx_dim = int(exp_ctx)
    log.info(f"[META][BUILD] src={src_name} | K={K}, L={L}, ctx_dim(built)={exp_ctx}, d_model={d_model}, n_heads={n_heads}")

    # Gewichte tolerant laden
    ik = meta.load_state_dict(sd, strict=False)
    miss = getattr(ik, "missing_keys", [])
    unex = getattr(ik, "unexpected_keys", [])
    if miss: log.warning(f"[META][LOAD] Missing keys: {list(miss)}")
    if unex: log.warning(f"[META][LOAD] Unexpected keys: {list(unex)}")

    meta.eval()
    pcount = sum(p.numel() for p in meta.parameters())
    log.info(f"[META][LOAD] Geladen aus '{ckpt.name}' | params={pcount:,}")
    return meta


@torch.no_grad()
def predict_meta_batched(meta_model,
                         H: np.ndarray,   # [N, L, K]
                         C: np.ndarray,   # [N, ctx_dim]
                         device: torch.device,
                         batch_size: int,
                         log):
    N, L, K = H.shape
    out = np.empty((N,), dtype=np.float32)
    bs = max(1, int(batch_size))
    for s in range(0, N, bs):
        e = min(N, s + bs)
        Hb = torch.tensor(H[s:e], dtype=torch.float32, device=device)
        Cb = torch.tensor(C[s:e], dtype=torch.float32, device=device)
        try:
            w, p_now = meta_model(Hb, Cb)              # [B,K], [B,K]
            p_base   = Hb[:, -1, :]                    # [B,K]
            # hier simple Mischung aus aktueller Base und Meta-Schätzung:
            p_mix    = 0.7 * p_now + 0.3 * p_base
            p_hat    = (w * p_mix).sum(dim=1).clamp(1e-7, 1-1e-7)
            out[s:e] = p_hat.float().detach().cpu().numpy()
        except Exception as ex:
            log.error(f"[META] Inferenzfehler @ batch {s}:{e}: {ex}")
            out[s:e] = 0.5
    log.debug(f"[META] predict: H={H.shape}, C={C.shape} → out={out.shape}")
    return out


# --- D2: Meta-Inputs Builder (pad/trim auf target_ctx_dim) --------------
def build_meta_inputs_from_base(base_preds: Dict[str, np.ndarray],
                                 L: int,
                                 log,
                                 target_ctx_dim: Optional[int] = None):
     keys = sorted([k for k in base_preds.keys() if base_preds[k] is not None])
     if not keys:
         return None, None, keys
     P = np.stack([base_preds[k].astype(np.float32) for k in keys], axis=1)  # [N,K]
     N, K = P.shape
     if target_ctx_dim is not None and target_ctx_dim != K:
         if target_ctx_dim > K:
             pad_cols = target_ctx_dim - K
             log.warning(f"[META][BUILD] ctx_dim pad: K={K} → {target_ctx_dim} (fülle {pad_cols} Spalten mit 0.5)")
             P = np.concatenate([P, np.full((N, pad_cols), 0.5, dtype=np.float32)], axis=1)
         else:
             log.warning(f"[META][BUILD] ctx_dim trim: K={K} → {target_ctx_dim} (verwende erste {target_ctx_dim} Modelle)")
             P = P[:, :target_ctx_dim]
             keys = keys[:target_ctx_dim]
         K = target_ctx_dim
     H = np.zeros((N, L, K), dtype=np.float32)
     H[:, -1, :] = P
     C = P.copy()
     log.debug(f"[META][BUILD] keys={keys}, H={H.shape}, C={C.shape}")
     return H, C, keys

# ============================ OUTPUT HELPERS ===============================
def _write_chunk_csv(df: pd.DataFrame, out_path: Path, header: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if header else "a"
    df.to_csv(out_path, mode=mode, header=header, index=False)

def build_meta_inputs_from_base_preserve_H(
   base_preds_dict: dict,
   L: int,
   target_ctx_dim: Optional[int] = None,
   prev_tail: Optional[np.ndarray] = None,
   timestamps: Optional[np.ndarray] = None,
   log=None,
):
   """
   Baut Meta-Inputs für Streaming:
     - H: [M, L, K]   ← Sliding Windows über die letzten L Schritte je Basismodell.
           K = # aktiver Basismodelle (RF, LGB, XGB, FT, CNN ...)  → **nie** pad/trim auf H!
     - C: [M, target_ctx_dim]  ← nur hier pad/trim (0.5) oder optionale Zeitfeatures.
+
   Args:
     base_preds_dict : dict[str, np.ndarray] mit gleichlangem M.
     L               : Window-Länge für Meta (z.B. seq_len).
     target_ctx_dim  : erwartete Kontextbreite lt. Checkpoint (z.B. 7).
     prev_tail       : optional (L-1, K) — Tail vom letzten Chunk.
     timestamps      : optional (M,) — pandas Timestamps oder np.datetime64, für hour_sin/cos.
   Returns:
     H: np.ndarray [M, L, K]
     C: np.ndarray [M, target_ctx_dim]
     new_tail: np.ndarray [(L-1), K]  — fürs nächste Chunk.
   """
   # -- 1) Basen deterministisch sortieren und stapeln
   keys = sorted([k for k, v in base_preds_dict.items() if v is not None])
   if not keys:
       return None, None, None
   P = np.stack([base_preds_dict[k].astype(np.float32) for k in keys], axis=1)  # [M, K]
   M, K = P.shape

   # -- 2) prev_tail robust auf (L-1, K) bringen
   L = int(L)
   need = max(0, L - 1)
   if need == 0:
       tail = np.zeros((0, K), dtype=np.float32)
   else:
       if prev_tail is None:
           tail = np.zeros((need, K), dtype=np.float32)
       else:
           pt = np.asarray(prev_tail, dtype=np.float32)
           # Spalten auf K mappen (ohne H zu padden):
           if pt.ndim != 2:
               pt = np.zeros((need, K), dtype=np.float32)
           else:
               # Breite justieren
               if pt.shape[1] > K:      # zu breit → trim links
                   pt = pt[:, :K]
               elif pt.shape[1] < K:    # zu schmal → rechts mit 0.5 auffüllen (neutral)
                   padc = K - pt.shape[1]
                   pt = np.concatenate([pt, np.full((pt.shape[0], padc), 0.5, np.float32)], axis=1)
               # Höhe exakt (L-1)
               if pt.shape[0] > need:
                   pt = pt[-need:, :]
               elif pt.shape[0] < need:
                   miss = need - pt.shape[0]
                   pt = np.concatenate([np.zeros((miss, K), dtype=np.float32), pt], axis=0)
           tail = pt

   # -- 3) Extended Sequenz = [tail ; P]  → damit sind alle Ler-Fenster direkt indexierbar
   P_ext = np.concatenate([tail, P], axis=0)  # [(L-1)+M, K]

   # -- 4) H füllen: jedes Sample i bekommt Fenster P_ext[i : i+L]
   H = np.zeros((M, L, K), dtype=np.float32)
   for i in range(M):
       H[i, :, :] = P_ext[i : i + L, :]

   # -- 5) C bauen: nur C wird auf target_ctx_dim gemappt
   C = P.copy()  # Start: letzte Schritt-Probs je Basismodell
   if target_ctx_dim is not None:
       tcd = int(target_ctx_dim)
       if tcd > C.shape[1]:
           extra = tcd - C.shape[1]
           # Optional: Zeitfeatures, falls Timestamps da und >=2 Slots frei
           fea = []
           if timestamps is not None and extra >= 2:
               # hour_sin/cos ∈ [-1,1]
               ts = pd.to_datetime(pd.Series(timestamps))
               ang = 2.0 * np.pi * (ts.dt.hour.values.astype(np.float32) / 24.0)
               fea.append(np.sin(ang).reshape(-1, 1).astype(np.float32))
               fea.append(np.cos(ang).reshape(-1, 1).astype(np.float32))
           if fea:
               F = np.concatenate(fea, axis=1)
               F = F[:, :min(F.shape[1], extra)]
               C = np.concatenate([C, F], axis=1)
               extra -= F.shape[1]
           if extra > 0:
               C = np.concatenate([C, np.full((M, extra), 0.5, dtype=np.float32)], axis=1)
           if log:
               log.warning(f"[META][BUILD] C pad: {K} → {tcd} (fülle {tcd - K} Spalten, inkl. evtl. Zeitfeatures/0.5)")
       elif tcd < C.shape[1]:
           C = C[:, :tcd]
           if log:
               log.warning(f"[META][BUILD] C trim: {C.shape[1]} → {tcd} (verwende erste Spalten)")

   # -- 6) neuer Tail = letzte (L-1) Reihen von P_ext (kein reshape!)
   new_tail = P_ext[-need:, :] if need > 0 else np.zeros((0, K), dtype=np.float32)

   if log:
       log.debug(f"[META][BUILD] keys={keys} | H={H.shape} | C={C.shape} | tail={new_tail.shape}")
   return H, C, new_tail

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
   # 1) input_proj.weight: [d_model, in_dim]
   w = sd.get("input_proj.weight", None)
   if getattr(w, "ndim", 0) == 2:
       return int(w.shape[0])
   # 2) self_attn.in_proj_weight: [3*d_model, d_model]
   for k, t in sd.items():
       if isinstance(k, str) and "self_attn.in_proj_weight" in k and getattr(t, "ndim", 0) == 2:
           if t.shape[0] % 3 == 0:
               return int(t.shape[0] // 3)
   # 3) Kopfgewichte: w_head / p_head: [K, d_model]
   for key in ("w_head.weight", "p_head.weight"):
       t = sd.get(key, None)
       if getattr(t, "ndim", 0) == 2:
           return int(t.shape[1])
   return None

def load_meta_model_robust(model_dir, K: int, L: int, device, log):
   """
   Robuster Meta-Loader:
     - ließt d_model/dropout aus best_params.json (bevorzugt meta.d_token)
     - überschreibt d_model mit Wert aus Checkpoint, falls vorhanden
     - ctx_dim wird aus Checkpoint (ctx_proj.weight.shape[1]) abgeleitet
     - läd tolerant (strict=False)
   RETURN: (meta_model, meta_info_dict)
   """
   from pathlib import Path
   import json, torch
   from trainers.hybrid_longtrend_trainer import MetaMoE as _Meta

   model_dir = Path(model_dir)
   # --- Hyperparameter defaults
   d_model, dropout = 96, 0.10
   try:
       bp = json.loads((model_dir / "best_params.json").read_text(encoding="utf-8"))
       if isinstance(bp, dict) and "meta" in bp and isinstance(bp["meta"], dict):
           # bevorzugt: nested meta
           d_model = int(bp["meta"].get("d_token", d_model))
           dropout = float(bp["meta"].get("dropout", dropout))
       else:
           # Fallback: flache Keys
           d_model = int(bp.get("meta_d_model", bp.get("d_model", d_model)))
           dropout = float(bp.get("meta_dropout", bp.get("dropout", dropout)))
       log.info(f"[META][HP] d_model={d_model}, dropout={dropout} (aus best_params.json)")
   except Exception:
       log.info(f"[META][HP] d_model={d_model}, dropout={dropout} (defaults)")

   # --- Checkpoint laden
   ckpt = None
   for name in ("meta.pt", "meta.pth", "meta_state.pt"):
       p = model_dir / name
       if p.exists():
           ckpt = p
           break
   if ckpt is None:
       log.warning("[META][LOAD] Kein meta.pt gefunden → Meta deaktiviert.")
       return None, {"K": K, "L": L, "ctx_dim": K}

   raw = torch.load(ckpt, map_location="cpu")
   sd  = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
   # Prefixe robust abstreifen (immer versuchen)
   if isinstance(sd, dict):
       for pref in ("module.", "model.", "meta.", "state_dict."):
           sub = {k[len(pref):]: v for k, v in sd.items() if isinstance(k, str) and k.startswith(pref)}
           if sub:
               sd = sub
               break

   # --- ctx_dim & d_model aus Checkpoint ableiten
   ctx_dim_req = _infer_ctx_dim_from_state_dict(sd)
   if ctx_dim_req is None:
       ctx_dim_req = K
       log.warning(f"[META] ctx_dim nicht im Checkpoint gefunden → fallback ctx_dim={ctx_dim_req}")
   elif ctx_dim_req != K:
       log.warning(f"[META] ctx_dim im Checkpoint ist {ctx_dim_req}, aktive Basismodelle K={K} → "
                   f"baue Meta mit ctx_dim={ctx_dim_req} (Inputs werden später gemappt/padded).")

   d_model_ckpt = _infer_d_model_from_state_dict(sd)
   if d_model_ckpt is not None and d_model_ckpt != d_model:
       log.warning(f"[META] d_model im Checkpoint={d_model_ckpt} ≠ best_params={d_model} → setze d_model={d_model_ckpt} für kompatibles Laden.")
       d_model = d_model_ckpt

   n_heads = _pick_n_heads(d_model)

   # --- Modell bauen und tolerant laden
   log.info(f"[META][BUILD] src=trainer.MetaMoE | K={K}, L={L}, ctx_dim(built)={ctx_dim_req}, d_model={d_model}, n_heads={n_heads}")
   try:
       meta = _Meta(K=K, L=L, ctx_dim=ctx_dim_req, d_model=d_model, n_heads=n_heads, dropout=dropout).to(device)
   except TypeError:
       meta = _Meta(K=K, L=L, ctx_dim=ctx_dim_req, d_model=d_model, n_heads=n_heads).to(device)
   meta.expected_ctx_dim = int(ctx_dim_req)
   meta.eval()

   ik = meta.load_state_dict(sd, strict=False)
   miss = getattr(ik, "missing_keys", [])
   unex = getattr(ik, "unexpected_keys", [])
   if miss: log.warning(f"[META][LOAD] Missing keys: {list(miss)}")
   if unex: log.warning(f"[META][LOAD] Unexpected keys: {list(unex)}")
   pcount = sum(p.numel() for p in meta.parameters())
   log.info(f"[META][LOAD] Geladen aus '{ckpt.name}' | params={pcount:,}")

   return meta, {"K": K, "L": L, "ctx_dim": ctx_dim_req}

# ============================ STREAM PREDICT ===============================
def stream_predict(args, cfg):
    log = get_logger("predict_v2", verbose=args.verbose)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    if device.type != "cuda":
        device = torch.device(device)

    log.info(f"Start | device={args.device} → {device.type} | {_gpu_mem_str()}")
    log.info(f"Args: {vars(args)}")

    model_dir = Path(args.model_dir)

    # Klassische Modelle
    rf_models  = load_rf_models(model_dir, log)
    lgb_models = load_lgb_models(model_dir, log)
    xgb_models = load_xgb_models(model_dir, log)

    # FT (lazy)
    ft_model = load_ft(model_dir, device, log)
    if ft_model is not None:
        log.info("[LOAD] Torch state_dict (dict) geladen aus ft.pt")

    # CNN (lazy, sobald F bekannt)
    cnn_model = None

    # Meta wird gebaut, sobald K/L/ctx_dim feststehen
    meta_model = None

    # Reader
    reader = _read_tick_chunks(args.ea_tick_file, chunksize=args.ea_chunk_rows, log=log)
    log.info(f"Reader geöffnet: {args.ea_tick_file} (chunksize={args.ea_chunk_rows})")

    out_path = Path(args.output) if args.output else None
    header_written = False
    total_rows = 0

    carry_ticks = pd.DataFrame(columns=["Time","Tick_Bid"])  # initial leer
    hist_ohlc   = pd.DataFrame(columns=["Time","Open","High","Low","Close","Volume"])

    seq_len = int(args.seq_len)
    horizon_bars = max(1, int(args.label_horizon_min // _freq_to_minutes(args.freq)))

    amp_dtype = amp_dtype_from_arg(args.precision)

    for ichunk, df_raw in enumerate(reader, start=1):
        log.info(f"--- Chunk #{ichunk} -----------------------------------------------")
        log.debug(f"df_raw.shape={df_raw.shape}\n{df_raw.head(3)}")

        # Resample
        ohlc_chunk, carry_ticks = _resample_ticks_with_carry(carry_ticks, df_raw, args.freq, log)

        # Historie anhängen (Index sauber halten)
        ohlc_all = pd.concat([hist_ohlc, ohlc_chunk], axis=0, ignore_index=True)

                # Windowing (OHLC für Trees/FT)
        X_seq_ohlc, X_last, ends = _make_windows(
            ohlc_all, seq_len=seq_len, horizon_bars=horizon_bars, hist_len=len(hist_ohlc), log=log
        )
        M_emit = len(ends)
        if M_emit == 0:
            log.warning("[Emit] Keine Fenster → nächsten Chunk lesen…")
            hist_ohlc = ohlc_all.iloc[-min(seq_len, len(ohlc_all)):].copy()
            continue
        N, T, F_ohlc = X_seq_ohlc.shape
        X_flat = X_seq_ohlc.reshape(N, T * F_ohlc).astype(np.float32, copy=False)
        log.debug(f"[Shapes] X_seq(ohlc)={X_seq_ohlc.shape}, X_last={X_last.shape}, X_flat={X_flat.shape}")

        # CNN-Feature-Sequenzen (10 Kanäle) – OHLCV -> Enrichment
        ohlc_idx = ohlc_all.copy()
        ohlc_idx["Time"] = pd.to_datetime(ohlc_idx["Time"])
        ohlc_idx = ohlc_idx.set_index("Time")
        df_feat = enrich_for_cnn(
            ohlc_idx[["Open","High","Low","Close","Volume"]].rename(
                columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}
            )
        )
        X_seq_cnn, _, _, _ = make_windows_from_features(df_feat, seq_len=seq_len, horizon_bars=horizon_bars)
        # ggf. Längen angleichen
        if len(X_seq_cnn) != N:
            m = min(N, len(X_seq_cnn))
            log.warning(f"[CNN] Window-Länge abweichend (base={N}, cnn={len(X_seq_cnn)}) → trim auf {m}")
            X_seq_ohlc = X_seq_ohlc[:m]
            X_last     = X_last[:m]
            X_flat     = X_flat[:m]
            ends       = ends[:m]
            N          = m
            X_seq_cnn  = X_seq_cnn[:m]

        # CNN beim ersten Mal laden (Feature-Anzahl df_feat.shape[1] = 10)
        if cnn_model is None:
            cnn_model = load_cnn_model(model_dir, n_feat=df_feat.shape[1], device=device, log=log)

        p_rf  = _predict_trees(rf_models,  X_last, X_flat, "RF",  log)  if rf_models  else np.full((N,), 0.5, np.float32)
        p_lgb = _predict_trees(lgb_models, X_last, X_flat, "LGB", log)  if lgb_models else np.full((N,), 0.5, np.float32)
        p_xgb = _predict_trees(xgb_models, X_last, X_flat, "XGB", log)  if xgb_models else np.full((N,), 0.5, np.float32)
        p_ft  = predict_ft_batched(ft_model,  X_last, device, args.nn_batch_size, amp_dtype, log) if ft_model else np.full((N,), 0.5, np.float32)
        p_cnn = predict_cnn_batched(cnn_model, X_seq_cnn, device, args.nn_batch_size, log) if cnn_model else np.full((N,), 0.5, np.float32)

        # Alle Basen auf die gleiche Länge trimmen (sicheres M)
        base_raw = {"rf":p_rf, "lgb":p_lgb, "xgb":p_xgb, "ft":p_ft, "cnn":p_cnn}
        M = min(len(v) for v in base_raw.values() if v is not None)
        base = {k: (v[:M] if v is not None else None) for k,v in base_raw.items()}
        N = M

        base = {
            "rf":  p_rf,
            "lgb": p_lgb,
            "xgb": p_xgb,
            "ft":  p_ft,
            "cnn": p_cnn,
        }
        K = sum(1 for v in base.values() if v is not None)
        log.debug(f"[Base] K={K} | RF/LGB/XGB/FT/CNN shapes: "
                  f"{tuple(map(lambda a: None if a is None else a.shape, base.values()))}")

        # --- PATCH E: Meta ggf. bauen (nur 1x) → K korrekt setzen, ctx_dim aus Checkpoint
        if meta_model is None:
            L_meta   = seq_len
            K_base   = sum(1 for v in base.values() if v is not None)
            meta_model, meta_info = load_meta_model_robust(
                model_dir=model_dir, K=K_base, L=L_meta, device=device, log=log
            )
            meta_tail = None  # History-Carry für H

        # --- PATCH E: Meta-Inputs bauen (H unverändert, NUR C auf ctx_dim mappen)
        if meta_model is not None:
            target_ctx = int(getattr(meta_model, "expected_ctx_dim", meta_info.get("ctx_dim", 0) or 0))
            H, C, meta_tail = build_meta_inputs_from_base_preserve_H(
                base_preds_dict=base, L=seq_len, target_ctx_dim=target_ctx,
                prev_tail=meta_tail, timestamps=None, log=log
            )
            if H is None:
                p_final = np.mean(np.stack([v for v in base.values()], axis=1), axis=1).astype(np.float32)
                log.info(f"[COMBINE] Meta-Inputs leer → Base-Average (M={N}, K={sum(1 for v in base.values() if v is not None)})")
            else:
                p_meta  = predict_meta_batched(meta_model, H, C, device, batch_size=max(1024, args.nn_batch_size), log=log)
                p_final = p_meta.astype(np.float32, copy=False)
                log.info(f"[COMBINE] Meta aktiv (K_in={H.shape[2]}, L={H.shape[1]}, ctx_dim={C.shape[1]}) → Meta-Ausgabe verwendet.")
        else:
            p_final = np.mean(np.stack([v for v in base.values()], axis=1), axis=1).astype(np.float32)
            log.info(f"[COMBINE] Meta nicht verfügbar → Base-Average (M={N}, K={sum(1 for v in base.values() if v is not None)})")




        # Metriken
        da_chunk = float((p_final >= 0.5).mean())
        band = float(args.mda_band)
        mask = np.abs(p_final - 0.5) >= band
        mda_chunk = float(np.mean((p_final[mask] >= 0.5).astype(np.float32))) if mask.any() else float("nan")
        cov = float(mask.mean() * 100.0)

        log.debug(f"[GPU] {_gpu_mem_str()}")
        log.info(f"[Chunk#{ichunk}] DA_chunk={da_chunk:.4f}, MDA_chunk(band={band})={mda_chunk:.4f}, coverage={cov:.2f}%")

        # Output
        out = pd.DataFrame({
            "proba":      p_final,
            "proba_rf":   base["rf"],
            "proba_lgb":  base["lgb"],
            "proba_xgb":  base["xgb"],
            "proba_ft":   base["ft"],
            "proba_cnn":  base["cnn"],
        })
        if out_path is not None:
            _write_chunk_csv(out, out_path, header=not header_written)
            header_written = True
            total_rows += len(out)
            log.info(f"[Write] +{len(out):,} rows → {out_path} (total written: {total_rows:,})")

        # History für nächsten Chunk
        hist_ohlc = ohlc_all.iloc[-min(seq_len, len(ohlc_all)):].copy()

    log.info(f"[DONE] Rows={total_rows:,}")

# ============================ CLI / MAIN ==================================
def load_config(path: Optional[str], log) -> Dict[str, Any]:
    if not path:
        log.warning("Keine --config angegeben; Minimal-Defaults werden genutzt.")
        return {"training":{"seq_len":24}}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        log.info(f"config.yaml geladen ({path})")
        return cfg
    except Exception as e:
        log.warning(f"config.yaml konnte nicht geladen werden ({path}): {e}")
        return {"training":{"seq_len":24}}

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("predict_v2")
    ap.add_argument("--model-dir", required=True, type=str)
    ap.add_argument("--config", required=False, type=str, default=None)

    # EA tick input
    ap.add_argument("--ea-tick-file", required=True, type=str)
    ap.add_argument("--freq", required=True, type=str, help="z.B. '1min', '60min', '4h'")
    ap.add_argument("--ea-chunk-rows", type=int, default=500_000)

    # labels / windows
    ap.add_argument("--label-horizon-min", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=24)

    # mda
    ap.add_argument("--mda-band", type=float, default=0.001)

    # output
    ap.add_argument("--output", required=True, type=str)

    # runtime
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--nn-batch-size", type=int, default=1024)
    ap.add_argument("--precision", type=str, default="auto", help="auto|fp32|fp16|bf16")
    ap.add_argument("--verbose", action="store_true")

    return ap.parse_args()

def main():
    args = parse_args()
    log = get_logger("bootstrap", verbose=args.verbose)
    cfg = load_config(args.config, log)
    try:
        stream_predict(args, cfg)
    except Exception as e:
        log.error(f"Fataler Fehler: {e}\n{traceback.format_exc()}")
        sys.exit(2)

if __name__ == "__main__":
    main()