# prepare_entry.py  – Labels „lokales Hoch / Tief in den nächsten 15 min“
import pandas as pd
from pathlib import Path
from features_utils import load_ticks, make_minute_bars      # 1‑Min‑Bars gen
RAW_TICKS = Path("data/RawTickData3.txt")
OUT_CSV   = Path("data/longtrend_entry.csv")

WIN_MIN   = 15                     # Fensterbreite
def main():
    ticks = load_ticks(RAW_TICKS)
    bars  = make_minute_bars(ticks)                # 1‑Min‑OHLCV
    bars["close_fwd_max"] = bars["Close"].rolling(WIN_MIN,  center=False).max().shift(-WIN_MIN)
    bars["close_fwd_min"] = bars["Close"].rolling(WIN_MIN,  center=False).min().shift(-WIN_MIN)

    cond_high = bars["close_fwd_max"] - bars["Close"] <= 0      # aktueller Close ist das Hoch des Fensters
    cond_low  = bars["Close"] - bars["close_fwd_min"] <= 0      # aktueller Close ist das Tief des Fensters

    bars["label_entry"] = 0
    bars.loc[cond_high, "label_entry"] = 1      # Hoch   → Short‑Setup
    bars.loc[cond_low,  "label_entry"] = 2      # Tief   → Long‑Setup

    bars.drop(columns=["close_fwd_*"], inplace=True, errors="ignore")
    bars.to_csv(OUT_CSV, index=False)
    print(f"✅ Gespeichert: {OUT_CSV}  ({len(bars)} Zeilen)")

if __name__ == "__main__":
    main()
