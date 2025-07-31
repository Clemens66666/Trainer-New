import pandas as pd
from .labeling import make_labels_exit           # ⇦ analog zu longtrend
from .features import add_time_features, make_indicators

def prepare_exit(df_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    1.  Feature-Engineering auf Roh-Ticks
    2.  Binäre Exit-Labels via `make_labels_exit`
    3.  Rückgabe: DataFrame mit allen Features + label
    """
    df = df_raw.copy()

    # Basis-Feature-Set wiederverwenden
    if cfg["features"].get("time_features", True):
        df = add_time_features(df)
    df = make_indicators(df, cfg["features"]["indicators"])

    for w in cfg["features"]["window_sizes"]:
        df[f"ret_mean_{w}"] = df["close"].pct_change().rolling(w).mean()
        df[f"ret_std_{w}"]  = df["close"].pct_change().rolling(w).std()

    # Labeling-Parameter
    lbl_cfg = cfg["exit"]                         # max_horizon, trail_atr_mul, …

    df["label"] = make_labels_exit(
        close          = df["close"].to_numpy(),
        atr            = df["atr_14"].to_numpy(),
        max_horizon    = lbl_cfg["max_horizon"],
        trail_atr_mul  = lbl_cfg["trail_atr_mul"],
    )

    return df.dropna().reset_index(drop=True)
