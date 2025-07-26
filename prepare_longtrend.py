# prepare_longtrend.py  –  Legacy-kompatible Pre‑Pipeline
import pandas as pd
from pathlib import Path
from features_utils import (      # ←  deine bestehenden Helfer
    load_ticks,
    make_hourly_bars,
)
from labeling import make_trend_labels

RAW_TXT   = Path("data/RawTickData3.txt")   # rohe Tickdatei
OUTPUT_CSV = Path("data/longtrend.csv")     # für Hybrid‑Trainer

# Label‑Parameter – identisch zu Legacy‑Trainer
HORIZON_H = 24            # 24‑Stunden‑Barriere
THR_UP    = 0.004         # +0.4 %
THR_DN    = 0.004         # −0.4 %

def main():
    print("▶  Ticks einlesen …")
    ticks = load_ticks(RAW_TXT)                 # parse_dates etc. in helper

    print("▶  Stündliche Bars bauen …")
    bars = make_hourly_bars(ticks)
    # ─── Spalten für labeling.py anpassen ────────────────
    # ── Spalten für Labeling + Feature‑Enrichment anpassen ──
    # 1) High/Low kleinschreiben, 2) price als Alias für Close hinzufügen,
    #    Close aber in der Tabelle belassen (wird für SMA etc. gebraucht)
    bars = (
        bars
        .rename(columns={"High": "high", "Low": "low"})
        .assign(price=bars["Close"])
        .set_index("TimeStamp")
    )
    
    print("▶  Trend-Labels berechnen …")
    # Trend-Labels berechnen und alle Trend-Spalten übernehmen …
    trend = make_trend_labels(bars)
    # Hängt side, beta, window & dc_phase an bars an
    bars = bars.join(trend)

    
    # ── Label aus dem Trend-Output generieren ────────────────
    # make_trend_labels liefert eine 'side' Spalte mit Werten {-1,0,1}
    print("▶  Label-Spalte aus Trend-Side erzeugen …")
    bars["label"] = bars["side"].map({-1: 0,  0: 0,  1: 1})

       # Entfernt die temporären Trend-Spalten (ignoriert fehlende Spalten)
    bars = bars.drop(
        columns=["side", "beta", "window", "dc_phase"],
        errors="ignore"
    )
    # Einheitliche Zeitspalte
    # ── Index zurückholen und in 'timestamp' umbenennen ─────────
    # Wir setzen den bisherigen Index (TimeStamp) als Spalte ab
    bars = bars.reset_index()
    # Manche Helper nennen sie 'TimeStamp', andere 'Time'
    bars = bars.rename(columns={"TimeStamp": "timestamp", "Time": "timestamp"})

    # Speichern
    bars.to_csv(OUTPUT_CSV, index=False)
    print(f"✅  Gespeichert: {OUTPUT_CSV}  ({len(bars)} Zeilen)")

if __name__ == "__main__":
    main()
