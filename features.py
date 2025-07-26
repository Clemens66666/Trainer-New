# utils/features.py
# ---------------------------------------------------------------------
#  Technische Indikatoren + Zeit‑Features für Candle‑DataFrames
# ---------------------------------------------------------------------
from __future__ import annotations

from typing   import Sequence, Iterable
from pathlib  import Path

import numpy  as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
#  Hilfs‑Funktionen
# ---------------------------------------------------------------------
def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up  = up.ewm(com=period - 1, adjust=False).mean()
    roll_down = down.ewm(com=period - 1, adjust=False).mean()
    rs  = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _bollinger_pct_b(series: pd.Series, window: int = 20, k: float = 2.0
                     ) -> pd.Series:
    sma = _sma(series, window)
    std = series.rolling(window, min_periods=window).std()
    upper = sma + k * std
    lower = sma - k * std
    pct_b = (series - lower) / (upper - lower)
    return pct_b


# ──────────────────────────────────────────────────────────────────────
#  Haupt‑Funktion
# ---------------------------------------------------------------------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt dem übergebenen Candle‑DataFrame neue Spalten hinzu:
        • SMA‑10   (sma_10)
        • EMA‑20   (ema_20)
        • RSI‑14   (rsi_14)
        • ATR‑14   (atr_14)
        • Bollinger‑%B (bb_pct_b)
        • Momentum‑Lags (mom_1 … mom_3)
        • Sin/Cos‑Zeit‑Kodierung (Stunde, Wochentag)
    Gibt eine **neue** DataFrame‑Kopie zurück.
    """
    df = df.copy()

    # ⇒ Gängige technische Indikatoren ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
    df["sma_10"]   = _sma(df["Close"], 10)
    df["ema_20"]   = _ema(df["Close"], 20)
    df["rsi_14"]   = _rsi(df["Close"], 14)
    df["atr_14"]   = _atr(df["high"], df["low"], df["Close"], 14)
    df["bb_pct_b"] = _bollinger_pct_b(df["Close"], 20, 2.0)

    # ⇒ Momentum‑Lags (1‑3 Kerzen) ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
    for lag in (1, 2, 3):
        df[f"mom_{lag}"] = df["Close"].pct_change(lag)

    # ⇒ Zeitfeat.: Stunde & Wochentag als Sin/Cos ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    hours      = df["timestamp"].dt.hour
    weekdays   = df["timestamp"].dt.weekday
    df["hour_sin"]   = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * hours / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * weekdays / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * weekdays / 7)

    # N/A‑Werte erst ganz am Ende (lassen Modell entscheiden → maskieren oder 0)
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)

    return df
