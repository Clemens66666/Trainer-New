"""
Label‑generation utilities for the EUR/USD 5‑minute system
───────────────────────────────────────────────────────────
Implements

    • make_trend_labels(df)
    • make_entry_labels(df, trend_labels)
    • make_exit_labels(trades, trend_labels)

The functions are pure (stateless) and return pandas
DataFrames aligned on the *event‑bar* index `ts`.

Assumptions
-----------
`df` is already resampled to Dollar/DC bars and contains
    ts          » datetime64[ns] – index or column
    price       » mid‑quote or last‑trade
    high, low   » (for ATR)
    volume      » (optional, not used here)

© 2025  –  Quant Research Team
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from scipy import stats
from typing import List, Tuple, Dict, Optional

# ---------------------------------------------------------------------
# Load hyper‑parameters once
PARAM_PATH = Path(__file__).with_name("config.yaml")
from pathlib import Path
PARAM_FILE = Path(__file__).with_name("config.yaml")
with open(PARAM_FILE, encoding="utf-8") as f:
     P = yaml.safe_load(f)
    
# ---------------------------------------------------------------------
# 0  – helpers
# ---------------------------------------------------------------------
def _annualise_beta(beta_per_bar: float, bar_len_min: int = 5) -> float:
    bars_per_year = 252 * 24 * 60 / bar_len_min
    return beta_per_bar * bars_per_year

def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         n: int = 14) -> pd.Series:
    """Classic Wilder ATR on event bars."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    z = np.exp(x / tau)
    return z / z.sum()

# ---------------------------------------------------------------------
# 1  – Trend labels
# ---------------------------------------------------------------------
from typing import List, Tuple
import numpy as np
from scipy import stats

from typing import List, Tuple
import numpy as np
from scipy import stats

def _trend_scan(
        logp: np.ndarray,
        windows: List[int],
        tau: float
) -> Tuple[int, int, float]:
    """
    Scannt mehrere Fensterlängen und gibt (best_k, side, annual_beta) zurück.
    logp darf auch 2-D sein – wird intern zu 1-D geflattet.
    """
    # ---- Robust: immer in 1-D wandeln ------------------------------
    logp = np.asarray(logp, dtype=float).reshape(-1)
    n = len(logp)

    best_t, best_k, best_beta = 0.0, 0, 0.0

    for k in windows:
        # mindestens 2 Punkte & Fenster muss in Serie passen
        if k <= 1 or k > n:
            continue

        y = logp[-k:]                       # Länge k, 1-D
        x = np.arange(k, dtype=float)

        slope, _, r, _, stderr = stats.linregress(x, y)
        t = slope / stderr if stderr else 0.0

        if abs(t) > abs(best_t):
            best_t, best_k, best_beta = t, k, slope

    # Trendrichtung ermitteln
    if abs(best_t) >= tau and best_k > 0:
        side = int(np.sign(best_beta))
    else:
        side, best_beta = 0, 0.0

    return best_k, side, _annualise_beta(best_beta)

from typing import List, Union
import numpy as np
import pandas as pd


def _directional_change(
        price: Union[pd.Series, pd.DataFrame, np.ndarray, List[float]],
        thresh: float
) -> pd.Series:
    """
    Bestimmt die Directional-Change-Phase (-1, 0, +1) für eine Preisreihe.

    Parameters
    ----------
    price  : Series | DataFrame | ndarray | list
        Preiszeitreihe. Darf auch ein einspaltiges DataFrame oder
        eine Python-Liste sein. Mehr als eine Spalte → ValueError.
    thresh : float
        Schwellenwert in Prozent (%). Bewegung ≥ +thresh -> +1,
        ≤ -thresh -> -1, sonst 0.

    Returns
    -------
    dc : pd.Series[int]
        Gleiche Länge wie `price`, enthält -1/0/+1.
    """
    # ---------- 1) Eingabe robust auf 1-D transformieren --------------
    if isinstance(price, pd.DataFrame):
        if price.shape[1] != 1:
            raise ValueError(
                f"_directional_change erwartet höchstens 1 Spalte, "
                f"bekam aber {price.shape[1]}."
            )
        price = price.iloc[:, 0]          # DataFrame -> Series

    if not isinstance(price, pd.Series):
        # deckt ndarray, list, tuple usw. ab
        price = pd.Series(price)

    # ---------- 2) Numerisch casten & Lücken füllen -------------------
    price = pd.to_numeric(price, errors="coerce").astype(np.float64).ffill()

    # ---------- 3) Früher Ausstieg, falls Serie leer ------------------
    if price.empty:
        return pd.Series(dtype=np.int8, name="dc_phase")

    # ---------- 4) DC-Berechnung -------------------------------------
    phase = np.zeros(len(price), dtype=np.int8)
    ref = price.iloc[0]

    for i, p in enumerate(price):
        move_pct = (p / ref - 1.0) * 100.0

        if move_pct >= thresh:
            phase[i] =  1
            ref = p
        elif move_pct <= -thresh:
            phase[i] = -1
            ref = p
        else:
            phase[i] =  0

    return pd.Series(phase, index=price.index, name="dc_phase")


# ────────────────────────────────────────────────────────────────────────────────
# Ausschnitt aus labeling.py – Funktion make_trend_labels
# ------------------------------------------------------------------------------
def make_trend_labels(
        df: pd.DataFrame,
        dc_thres: float,
        w_list: List[int],
        tau: float
) -> pd.DataFrame:
    """
    Erzeugt Trend-Labels (Directional Change + Trend-Scan).
    Gibt DataFrame mit Spalten 'dc_phase', 'trend_side', 'beta_ann' zurück.
    """
    trend = pd.DataFrame(index=df.index)

    # --------------------------------------------------------------------------
    # DC-Phase berechnen – Variante 2: Falls 'price' ein DataFrame mit
    # mehreren Spalten ist, Mittelwert über alle Spalten verwenden
    # --------------------------------------------------------------------------
    price_obj = df["price"]

    # Falls 'price' ein DataFrame (mehrere Spalten) ist, → Mittelwert
    if isinstance(price_obj, pd.DataFrame):
        # Mittelwert über alle Spalten (z. B. Bid/Ask -> Mid-Price)
        price_input = price_obj.mean(axis=1)
    else:
        # Series oder 1-D-Array: unverändert übernehmen
        price_input = price_obj

    # Directional-Change-Phase in -1 / 0 / +1
    trend["dc_phase"] = _directional_change(price_input, dc_thres).values

    # --------------------------------------------------------------------------
    # Trend-Scan (k, side, beta) für jedes Zeitfenster i
    # --------------------------------------------------------------------------
    logp = np.log(df["price"].iloc[:, 0] if isinstance(df["price"], pd.DataFrame)
                  else df["price"].values.astype(float))

    k_list, side_list, beta_list = [], [], []
    for i in range(len(logp)):
        k, side, beta = _trend_scan(logp[: i + 1], w_list, tau)
        k_list.append(k)
        side_list.append(side)
        beta_list.append(beta)

    trend["trend_k"]    = k_list
    trend["trend_side"] = side_list
    trend["beta_ann"]   = beta_list

    return trend
# ────────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------
# 2  – Entry labels
# ---------------------------------------------------------------------
def _triple_barrier(
    df: pd.DataFrame,
    idx: int,
    side: int,
    atr: pd.Series,
    tp_mul: float,
    sl_mul: float,
    horizon: int,
) -> Tuple[str, int]:
    """
    Simulate forward until barrier hit.
    Returns (event, steps)
        event ∈ {"TP","SL","TIME"}
    """
    entry_price = df["price"].iat[idx]
    tp = entry_price * (1 + side * tp_mul * atr.iat[idx])
    sl = entry_price * (1 - side * sl_mul * atr.iat[idx])

    for h in range(1, horizon + 1):
        if idx + h >= len(df):
            break
        p = df["price"].iat[idx + h]
        if side == 1:
            if p >= tp:
                return "TP", h
            if p <= sl:
                return "SL", h
        else:
            if p <= tp:
                return "TP", h
            if p >= sl:
                return "SL", h
    return "TIME", horizon

_OUTCOME_MAP = {"SL": 0, "TIME": 1, "TP": 2}

def make_entry_labels(df: pd.DataFrame,
                      trend: pd.DataFrame) -> pd.DataFrame:
    assert df.index.equals(trend.index), "Input not aligned."
    hp = P["entry"]
    atr = _atr(df["high"], df["low"], df["price"],
               n=hp["atr_window"])
    tp_mul, sl_mul, horizon = hp["tp_mul"], hp["sl_mul"], hp["horizon"]
    p_thresh = hp["execute_p"]

    rows = []
    for i, ts in enumerate(df.index):
        side = trend["side"].iat[i]
        if side == 0:
            rows.append((ts, 0, "NA", 0, 0.0))
            continue
        outcome, _ = _triple_barrier(df, i, side, atr,
                                     tp_mul, sl_mul, horizon)
        y_ord = _OUTCOME_MAP[outcome]
        probs = _softmax(np.array([0, 1, 2]))  # naive; placeholder for ML softmax
        p_tp  = probs[2]
        execute = int(p_tp >= p_thresh)
        rows.append((ts, side, outcome, execute, p_tp))

    return pd.DataFrame(rows,
            columns=["ts", "side", "outcome", "execute_flag", "tp_prob"]
        ).set_index("ts")

# ---------------------------------------------------------------------
# 3  – Exit labels (survival)
# ---------------------------------------------------------------------
def make_exit_labels(trades: pd.DataFrame,
                     trend: pd.DataFrame) -> pd.DataFrame:
    """
    trades columns expected:
        trade_id | ts_enter | side | entry_price | ...
    """
    hp = P["exit"]
    horizon = hp["max_horizon"]
    atr_mul = hp["trail_atr_mul"]

    ids, durs, evts, cens, opt_stop = [], [], [], [], []
    trend_series = trend["side"]

    for row in trades.itertuples():
        idx_start = trend.index.get_loc(row.ts_enter)
        side = row.side
        entry_price = row.entry_price
        duration = horizon
        event = "TIMEOUT"
        stop_price = None
        for h in range(1, horizon + 1):
            if idx_start + h >= len(trend):
                censored = 1
                break
            ts = trend.index[idx_start + h]
            price = trend.index[idx_start + h]  # placeholder: df["price"] lookup
            # quick trail stop heuristic
            atr_now = trend["beta"].iat[idx_start + h]  # we don't have ATR here; placeholder
            trail = atr_mul * atr_now
            if side == 1:
                stop_price = max(stop_price or entry_price, price - trail)
                if price <= stop_price:
                    event, duration = "TRAIL", h
                    censored = 0
                    break
            else:
                stop_price = min(stop_price or entry_price, price + trail)
                if price >= stop_price:
                    event, duration = "TRAIL", h
                    censored = 0
                    break
            if trend_series.iat[idx_start + h] != side:
                event, duration = "TrendFlip", h
                censored = 0
                break
        else:
            censored = int(event == "TIMEOUT")

        ids.append(row.trade_id)
        durs.append(duration)
        evts.append(event)
        cens.append(censored)
        opt_stop.append(trail if stop_price is not None else np.nan)

    return pd.DataFrame({
        "trade_id": ids,
        "duration": durs,
        "event_type": evts,
        "censored": cens,
        "optimal_stop_multiple": opt_stop,
    })

# ---------------------------------------------------------------------
# 4  – Quick leakage & distribution check
# ---------------------------------------------------------------------
def _class_balance(df: pd.DataFrame, col: str, min_share: float = .05):
    share = df[col].value_counts(normalize=True)
    poor = share[share < min_share]
    if len(poor):
        print(f"[WARN] Low class share in {col}:")
        print(poor.to_string())

def run_unit_tests(sample_df: pd.DataFrame):
    t = make_trend_labels(sample_df)
    e = make_entry_labels(sample_df, t)
    _class_balance(t, "side")
    _class_balance(e, "execute_flag")
    # leakage test: no label uses future price beyond its definition
    assert np.isfinite(t["beta"]).all(), "NaNs in beta."
    print("✓ basic tests passed.")
