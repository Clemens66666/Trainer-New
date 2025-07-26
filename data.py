from pathlib import Path
import pandas as pd

def read_raw_csv(path):
    """
    Liest Long‑Trend‑ oder Tick‑CSV ein und sorgt dafür,
    dass am Ende *immer* eine Spalte `timestamp` existiert.
    """
    # 1. Kandidaten prüfen
    tmp = pd.read_csv(path, nrows=0).columns
    times = [c for c in ("timestamp", "Time") if c in tmp]

    # 2. Datei einlesen, nur vorhandene Zeitspalte parsen
    df = pd.read_csv(path, parse_dates=times)

    # 3. Auf einheitlichen Namen bringen
    if "Time" in df.columns and "timestamp" not in df.columns:
        df.rename(columns={"Time": "timestamp"}, inplace=True)

    # 4. Sortieren & zurück
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
