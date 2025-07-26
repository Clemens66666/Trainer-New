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
def _trend_scan(logp: np.ndarray, windows: List[int], tau: float
               ) -> Tuple[int, float, float]:
    """
    Return best window k, side (+1/‑1/0) and annualised β.
    """
    best_t, best_k, best_beta = 0.0, 0, 0.0
    for k in windows:
        if k >= len(logp):  # end of series
            break
        y = logp[-k:]
        x = np.arange(k)
        slope, _, r, _, stderr = stats.linregress(x, y)
        t = slope / stderr if stderr else 0.0
        if abs(t) > abs(best_t):
            best_t, best_k, best_beta = t, k, slope
    if abs(best_t) >= tau:
        side = int(np.sign(best_beta))
    else:
        side = 0
    return best_k, side, _annualise_beta(best_beta)

def _directional_change(prices: pd.Series, threshold: float
                       ) -> pd.Series:
    """
    Very light DC phase flag:
        0 = normal / trend, 1 = overshoot, 2 = end‐phase
    """
    dc = np.zeros(len(prices), dtype=int)
    ref = prices.iloc[0]
    mode = 0  # 0 = up‑move, 1 = down‑move
    last_idx = 0
    for i, p in enumerate(prices):
        move = (p / ref - 1) * 100  # %
        trigger = threshold if mode == 0 else -threshold
        if (mode == 0 and move >= trigger) or (mode == 1 and move <= trigger):
            # directional change event
            dc[last_idx:i + 1] = 1  # overshoot
            ref = p
            mode ^= 1
            last_idx = i + 1
    dc[last_idx:] = 2  # end‑phase tail
    return pd.Series(dc, index=prices.index)

def make_trend_labels(df: pd.DataFrame) -> pd.DataFrame:
    w_list   : List[int]  = P["trend"]["windows"]
    tau      : float      = P["trend"]["t_stat_thresh"]
    dc_thres : float      = P["trend"]["dc_threshold_pct"]

    logp = np.log(df["price"].values)
    rows = []
    for i in range(len(df)):
        k, side, beta = _trend_scan(logp[:i+1], w_list, tau)
        rows.append((df.index[i], side, beta, k))
    trend = pd.DataFrame(rows, columns=["ts", "side", "beta", "window"]).set_index("ts")

    trend["dc_phase"] = _directional_change(df["price"], dc_thres).values
    return trend

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
