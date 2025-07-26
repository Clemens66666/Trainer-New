# validate.py
import argparse, json, joblib, pickle, yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import metrics

# lokale Utils/Module
from utils.features import make_features               # dein Feature‑Generator
from utils.labeling import make_labels                 # falls du Labels neu brauchst
from trainers.hybrid_longtrend_trainer import (
    HybridLongTrendTrainer,
)                                                      # nur um schnell das Ensemble‑Objekt zu rekonstruieren


def load_oos_file(path: Path) -> pd.DataFrame:
    """Einfacher Loader für Raw‑Tick‑Text: erwartet csv‑ähnliche Spaltenüberschrift"""
    df = pd.read_csv(path, sep=",")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    return df.reset_index(drop=True)


def evaluate(y_true, y_pred, prices):
    """Berechnet alle Metriken in einem Rutsch"""
    bin_pred = (y_pred >= 0.5).astype(int)

    # Klassische Kennzahlen
    ll   = metrics.log_loss(y_true, y_pred)
    acc  = metrics.accuracy_score(y_true, bin_pred)
    prec = metrics.precision_score(y_true, bin_pred)
    rec  = metrics.recall_score(y_true, bin_pred)
    f1   = metrics.f1_score(y_true, bin_pred)
    auc  = metrics.roc_auc_score(y_true, y_pred)

    # PnL‑basierte Kennzahlen
    pnl      = np.where(bin_pred == 1, prices.shift(-1) - prices, 0.0)  # sehr simpel – ersetze gern
    wins     = pnl[pnl > 0]
    losses   = pnl[pnl < 0]
    hit_rate = (pnl > 0).mean()

    avg_win  = wins.mean()  if len(wins)   else 0
    avg_loss = losses.mean() if len(losses) else 0
    expectancy = hit_rate * avg_win + (1 - hit_rate) * avg_loss

    sharpe = pnl.mean() / (pnl.std(ddof=1) + 1e-8)

    return {
        "log_loss": ll,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "hit_rate": hit_rate,
        "avg_return": pnl.mean(),
        "expectancy": expectancy,
        "sharpe": sharpe,
    }


def main(args):
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    # ---- 1. Ensemble rekonstruieren ----
    with open(model_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    # kleiner Trick: wir instanziieren den Trainer nur,
    # um genau dieselbe Pipeline/Preprocessing wiederzubekommen
    trainer = HybridLongTrendTrainer(cfg)
    trainer.load(model_dir)                     # erwartet .pt/.pkl/.joblib in model_dir

    results = {}

    for fname in ["RawTickDataTestData.txt", "RawTickDataOOS.txt"]:
        df_raw = load_oos_file(data_dir / fname)

        # ---- 2. Feature Engineering (identisch zum Training) ----
        X_feat = make_features(df_raw, cfg["features"])
        y_true = make_labels(df_raw, cfg["trend"]) if "label" not in df_raw else df_raw["label"]
        prices = df_raw["price"] if "price" in df_raw else df_raw["Close"]

        # ---- 3. Vorhersage ----
        with torch.no_grad():
            y_pred = trainer.predict_proba(X_feat)   # liefert Wahrscheinlichkeiten in numpy

        # ---- 4. Metriken ----
        results[fname] = evaluate(y_true.values, y_pred, prices.values)

    # hübsch ausgeben
    print("\nValidation metrics")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="data",  help="Pfad zu Raw OOS‑Files")
    p.add_argument("--model_dir", default="models/hybrid_best", help="Ordner mit ft_ssl.pt, rf.pkl etc.")
    main(p.parse_args())
