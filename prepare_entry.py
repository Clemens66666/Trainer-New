import pandas as pd
from .labeling import make_labels_entry          # ⇦ analog zu longtrend
from .features import add_time_features, make_indicators

def prepare_entry(df_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    1.  Bereitet Roh-Tickdaten für das Entry-Modell auf
    2.  Setzt Binary-Labels via `make_labels_entry`
    3.  Liefert ein sauberes DataFrame mit Features + label-Spalte
    """
    # ---------- 1) Grund-Preprocessing ----------
    df = df_raw.copy()

    # Zeit-basierte Features (gleiche Funktion wie in prepare_longtrend)
    if cfg["features"].get("time_features", True):
        df = add_time_features(df)

    # Technische Indikatoren
    inds = cfg["features"]["indicators"]
    df = make_indicators(df, inds)

    # Window-Aggregationen (z. B. rolling mean/vol)
    for w in cfg["features"]["window_sizes"]:
        df[f"ret_mean_{w}"] = df["close"].pct_change().rolling(w).mean()
        df[f"ret_std_{w}"]  = df["close"].pct_change().rolling(w).std()

    # ---------- 2) Label-Erstellung ----------
    lbl_cfg   = cfg["entry"]                       # ATR-Window, tp/sl/horizon …
    df["label"] = make_labels_entry(               # TRUE = Entry-Signal
        close       = df["close"].to_numpy(),
        atr         = df["atr_14"].to_numpy(),     # schon in make_indicators
        atr_window  = lbl_cfg["atr_window"],
        tp_mul      = lbl_cfg["tp_mul"],
        sl_mul      = lbl_cfg["sl_mul"],
        horizon     = lbl_cfg["horizon"],
        execute_p   = lbl_cfg["execute_p"],
    )

    # ---------- 3) NaN-Bereinigung ----------
    df = df.dropna().reset_index(drop=True)

    return df
