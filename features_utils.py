# ────────────────────────────────────────────────────────────
#   ▸ load_ticks            : CSV/Parquet → DataFrame
#   ▸ make_hourly_bars      : Tick-→OHLCV-Resampling (1-H)
#   ▸ triple_barrier_label  : up/down/neutral   (−1/0/+1)
#   ▸ leak_filter           : entfernt Bars, die das zukünftige
#                             Horizon-Fenster überschneiden
# ────────────────────────────────────────────────────────────
# features_utils.py  ──────────────────────────────────────────────
from pathlib import Path          # <─  neu
from datetime import timedelta    # <─  neu
import pandas as pd               # <─  neu
import numpy as np                # <─  neu


def load_ticks(path: Path) -> pd.DataFrame:
    df = (
        pd.read_csv(path, parse_dates=["Time"])    # ← Time → Datum
          .rename(columns={
              "Time":      "TimeStamp",
              "Tick_Bid":  "Bid",
              "Tick_Ask":  "Ask",
              "Tick_Last": "Last",          # ← HIER umbenennen!
          })
          .set_index("TimeStamp")
          .sort_index()
    )
    return df

# ------------------------------------------------------------------
# 1)  make_hourly_bars  – stellt sicher, dass ‚Close‘ existiert
# ------------------------------------------------------------------
def make_hourly_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert Tick-Data (Bid/Ask/Last) zu 1-Stunden-OHLC-Bars.
    ▸  liefert Spalten: TimeStamp, Open, High, Low, Close, Volume
    """
    # Wir nehmen den Letztpreis (oder Bid, falls es kein Last gibt)
    price_col = "Last" if "Last" in ticks.columns else "Bid"
    
    ohlc = ticks[price_col].resample("1h").ohlc()
    ohlc.columns = ["Open", "High", "Low", "Close"]        # << Großschreibung!
    
    vol  = ticks[price_col].resample("1h").size().rename("Volume")
    bars = pd.concat([ohlc, vol], axis=1).dropna().reset_index()

    bars.rename(columns={"index": "TimeStamp"}, inplace=True)
    return bars


