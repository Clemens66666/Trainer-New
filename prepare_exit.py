# prepare_exit.py – Label „Rückschlag ≥ Thr innerhalb n Bars“
import pandas as pd
from pathlib import Path
from features_utils import load_ticks, make_hourly_bars
RAW_TICKS = Path("data/RawTickData3.txt")
OUT_CSV   = Path("data/longtrend_exit.csv")

N_BARS = 12               # Horizont (n Stunden)
PCT_DN = 0.003            # ‑0.3 % Rückschlag

def main():
    ticks = load_ticks(RAW_TICKS)
    bars  = make_hourly_bars(ticks)

    # Minimaler Draw‑Down relativ zum Start‑Close der nächsten n Stunden
    forward_min = bars["Close"].rolling(N_BARS, center=False).min().shift(-N_BARS)
    draw_dn_pct = (bars["Close"] - forward_min) / bars["Close"]

    bars["label_exit"] = (draw_dn_pct >= PCT_DN).astype(int)
    bars.to_csv(OUT_CSV, index=False)
    print(f"✅ Gespeichert: {OUT_CSV}  ({len(bars)} Zeilen)")

if __name__ == "__main__":
    main()
