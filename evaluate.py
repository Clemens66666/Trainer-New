#!/usr/bin/env python
# evaluate_plus.py
"""
Generische Evaluations-Pipeline für ML-Trading-Modelle

Funktionen
----------
* Ladet Tick- oder Kerz­endaten, resampelt optional in Minuten-Bars
* Erzeugt Trend-Labels (Directional-Change-Methode)
* Berechnet technische Features (nutzt utils.features.enrich)
* Baut Sequenzen und ruft passende Modell-Komponenten auf
* Aggregiert Vorhersagen, führt Meta-Ensemble aus (falls vorhanden)
* Gibt Klassifikations- sowie Trading-Metriken aus
* Erstellt optionale Visualisierungen und Feature-Analysen
* Unterstützt mehrere --model-dir Angaben für Modell­vergleich

Abhängigkeiten: pandas, numpy, PyYAML, scikit-learn, torch, rtdl, matplotlib,
seaborn (nur für Plots), joblib (Pickle), tqdm (Progress-Bar)

Beispiel
--------
python evaluate_plus.py \
    --model-dir models/hybrid_20250802 \
    --test-file data/rawtickdata3.txt \
    --seq-len 24 --bar-min 60 \
    --plots
"""
# ---------------------------------------------------------------------------

import argparse, json, yaml, sys, inspect, pickle, math, textwrap, warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve, auc
)
# ------------- NEW IMPORTS --------------------
from trainers.hybrid_longtrend_trainer import extract_regime_features
import json
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Suppress noisy warnings (optional)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# -----------------------------  Helper-Funktionen  -------------------------
# ---------------------------------------------------------------------------

def json_or_yaml(
    file_path: Path,
    yaml_keys: List[str],
    cfg_fallback: Dict[str, Any],
    encoding: str = "utf-8"
) -> Dict[str, Any]:
    """
    1) Wenn *file_path* existiert -> JSON laden
    2) Sonst versuche die Keys im globalen YAML-Dict cfg_fallback zu finden
    3) Fehlt alles: gib leeres Dict zurück
    """
    if file_path.exists():
        with open(file_path, "r", encoding=encoding) as f:
            return json.load(f)
    # YAML-Pfad im Fallback nachschlagen
    cur = cfg_fallback
    for k in yaml_keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return {}
    return cur if isinstance(cur, dict) else {}


def adapt_ft_cfg(ft_cfg: Dict[str, Any], cls) -> Dict[str, Any]:
    """
    Entfernt Keys, die der FTTransformer (oder Ersatz-Klasse) nicht kennt,
    und mappt veraltete Namen (z.B. d_model -> d_token).
    """
    if not ft_cfg:
        return {}
    sig = inspect.signature(cls.__init__)
    valid_keys = set(sig.parameters.keys())
    mapping = {"d_model": "d_token"}  # Beispiel-Alias
    adapted = {}
    for k, v in ft_cfg.items():
        k2 = mapping.get(k, k)
        if k2 in valid_keys:
            adapted[k2] = v
    return adapted

def filter_cfg(cfg: dict, cls):
     """
     Entfernt alle Keys, die der Ziel-Klasse nicht kennt – so vermeiden wir
     TypeErrors wie 'unexpected keyword'.
     """
     if not cfg:
         return {}
     valid = set(inspect.signature(cls.__init__).parameters)
     return {k: v for k, v in cfg.items() if k in valid}

def mean_preds(pred_lists: List[np.ndarray]) -> np.ndarray:
    """Mittelt eine Liste gleicher NumPy-Arrays (axis=0)"""
    if not pred_lists:
        return np.array([])
    stacked = np.vstack(pred_lists)
    return stacked.mean(axis=0)


def load_pickle(path: Path):
    """Kompatibles Laden für Pickles / joblib"""
    if not path.exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


# ------------------  Daten-Utilities: Tick → Bar & Labeling  ---------------

# ---------------------------------------------------------------------------
# Daten-Utility: Tick-Datei ➜ Minuten-Bars
# ---------------------------------------------------------------------------
def iter_ticks_as_bars(
    tick_file: Path,
    bar_min: int,
    chunksize: int = 2_000_000,
):
    """
    Liest eine große Tick-Datei stückweise ein und resampelt in OHLCV-Bars.
    Erkennt viele verschiedene Kopfzeilen-Schreibweisen.

    Erwartete Minimal-Spalten nach Umbenennung:
        timestamp • price • volume
    Preis_bid / price_ask werden optional als Extra-Features belassen.
    """
    # ------------------------------------------------------------
    # 1) Lesestrategie: zuerst mit Standard-usecols, sonst Fallback
    # ------------------------------------------------------------
    read_kwargs = dict(
        chunksize=chunksize,
        low_memory=False,
    )

    try:
        chunk_iter = pd.read_csv(
            tick_file,
            usecols=["timestamp", "price", "volume"],
            parse_dates=["timestamp"],
            dtype={"price": "float64", "volume": "float64"},
            **read_kwargs,
        )
    except ValueError:
        # Header weicht ab → ohne usecols lesen, später umbenennen
        chunk_iter = pd.read_csv(tick_file, **read_kwargs)

    # ------------------------------------------------------------
    # 2) Jeder Chunk wird vereinheitlicht und in Bars gewandelt
    # ------------------------------------------------------------
    for chunk in chunk_iter:
        # ---------- a) Spalten umbenennen ----------
        rename_map = {
            # Zeitstempel-Aliase
            "timestamp": "timestamp", "time": "timestamp",
            "datetime": "timestamp", "date": "timestamp",
            "datime": "timestamp",

            # Preis-Aliase
            "price": "price", "close": "price", "last": "price",
            "bid": "price", "ask": "price",
            "tick_last": "price", "tick_bid": "price_bid",
            "tick_ask": "price_ask",

            # Volumen-Aliase
            "volume": "volume", "size": "volume",
            "qty": "volume", "tick_volume": "volume",
        }

        chunk.rename(
            columns={
                c: rename_map[c.lower()]
                for c in chunk.columns
                if c.lower() in rename_map
            },
            inplace=True,
        )

        # Pflicht-Spalten prüfen
        if "timestamp" not in chunk.columns:
            raise ValueError(
                f"Keine Zeitstempel-Spalte in {tick_file} gefunden!"
            )
        if "price" not in chunk.columns:
            raise ValueError(
                f"Keine Preis-Spalte in {tick_file} gefunden!"
            )
        # Volume not mandatory for price computations; set 0 if absent
        if "volume" not in chunk.columns:
            chunk["volume"] = 0.0

        # ---------- b) Zeitstempel & Index ----------
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        chunk = chunk.set_index("timestamp").sort_index()
        chunk = chunk[~chunk.index.duplicated(keep="last")]

        # ---------- c) Numerische Spalten in float ----------
        for num_col in ["price", "volume", "price_bid", "price_ask"]:
            if num_col in chunk.columns:
                chunk[num_col] = pd.to_numeric(
                    chunk[num_col], errors="coerce"
                )

        # ---------- d) Resampling in Minuten-Bars ----------
        bar = chunk.resample(f"{bar_min}min").agg(
            price=("price", "last"),
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("volume", "sum"),
            # price_bid/ask optional:
            price_bid=("price_bid", "last") if "price_bid" in chunk else np.nan,
            price_ask=("price_ask", "last") if "price_ask" in chunk else np.nan,
        )

        # Nur vollständige Bars behalten
        bar = bar.dropna(subset=["price", "open", "high", "low", "close"])
        if not bar.empty:
            yield bar



# -----------  Trend-Labeling (Directional-Change + Trend-Scan)  ------------

def _directional_change(price: pd.Series, dc_thres: float) -> pd.Series:
    """
    Primitive DC-Phase-Erkennung: +1 = Up, -1 = Down, 0 = Idle
    """
    price = pd.to_numeric(price, errors="coerce").astype(np.float64)
    ref = price.iloc[0]
    phase = np.zeros(len(price), dtype=int)
    dc = np.zeros(len(price), dtype=int)

    last_ext = ref
    direction = 0  # 1 := up, -1 := down, 0 := neutral
    for i, p in enumerate(price):
        move = (p / last_ext - 1) * 100  # Prozent
        if direction >= 0 and move <= -dc_thres:
            direction = -1
            last_ext = p
            dc[i] = direction
        elif direction <= 0 and move >= dc_thres:
            direction = 1
            last_ext = p
            dc[i] = direction
        phase[i] = direction
    return pd.Series(phase, index=price.index, name="dc_phase")


def make_trend_labels(
    df: pd.DataFrame,
    dc_thres: float = 0.5,
    w_list: List[int] = (5, 15, 30),
    tau: int = 1
) -> pd.DataFrame:
    """
    1. Directional-Change-Phase bestimmen
    2. Trend-Scan = Mehrheit der DC-Phasen in gleitenden Fenstern w_list
    """
    if "price" not in df.columns:
        raise KeyError("'price' Spalte fehlt für Trend-Labeling")
    trend = pd.DataFrame(index=df.index)
    trend["dc_phase"] = _directional_change(df["price"], dc_thres)
    # Trend-Scan: Mehrheit >= tau Fenster müssen up ODER down sein
    for w in w_list:
        trend[f"scan{w}"] = trend["dc_phase"].rolling(w, min_periods=w) \
            .apply(lambda arr: math.copysign(1, arr.sum())
                   if abs(arr.sum()) >= tau else 0, raw=True)
    # Endgültiger Trend = Durchschnitt der Fenster
    scans = trend[[f"scan{w}" for w in w_list]]
    trend["trend_side"] = scans.mean(axis=1).apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return trend


# ---------------  Feature-Engineering Wrapper (example)  -------------------

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ruft utils.features.enrich auf.
    Fügt vorab eine 'timestamp'-Spalte hinzu, wenn der Zeitstempel nur im Index steckt,
    denn enrich() erwartet diese Spalte.
    """
    # timestamp-Spalte sicherstellen
    if "timestamp" not in df.columns:
        # Timestamp als Spalte *hinzufügen*, aber Index unverändert lassen
        df = df.copy()
        df["timestamp"] = df.index

    from utils.features import enrich  # Projektfunktion
    feat = enrich(df.copy())

    # Basis-OHLCV ggf. nachziehen
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns and col not in feat.columns:
            feat[col] = df[col]
    return feat



# ---------------------------------------------------------------------------
# ------------------------------  Haupt-Routine  ----------------------------
# ---------------------------------------------------------------------------
def evaluate_one_model(
    model_dir: Path,
    test_file: Path,
    seq_len: int,
    bar_min: int,
    plots: bool = False,
) -> Dict[str, Any]:
    """
    Führt Evaluation für ein Modellverzeichnis durch
    und gibt ein Dict mit allen Metriken + Plot-Pfaden zurück.
    """

    # ---------------- Konfiguration ----------------
    cfg_yaml = yaml.safe_load(open(model_dir / "config.yaml", encoding="utf-8"))

    def J(json_name: str, yaml_path: List[str], default={}):
        return json_or_yaml(model_dir / json_name, yaml_path, cfg_yaml) or default

    # ---------------- Temperature-Scaler laden ----------------
    T = 1.0
    temp_path = model_dir / "temp_scaler.pt"
    if temp_path.exists():
        T = torch.load(temp_path)["temperature"]
        print(f"🔸 Temperature Scaling aktiv (T={T:.3f})")

    # ------------- Modell-Komponenten laden ----------
    from trainers.hybrid_longtrend_trainer import (
        FTWrapped, SimpleCNN, MetaTransformer
    )

    # FT-Transformer
    ft_cfg = adapt_ft_cfg(J("ft_cfg.json", ["model", "ft_params"]), FTWrapped)
    ft_wrap = None
    if ft_cfg:
        ft_backbone = FTWrapped(**ft_cfg)
        ft_backbone.load_state_dict(torch.load(model_dir / "ft.pt", map_location="cpu"))
        ft_backbone.eval()
        ft_wrap = FTWrapped(ft_backbone)
        ft_wrap.load_state_dict(torch.load(model_dir / "ft_wrap.pt", map_location="cpu"))
        ft_wrap.eval()

    # CNN
    cnn_cfg = filter_cfg(J("cnn_cfg.json", ["cnn"]), SimpleCNN)
    cnn, n_cnn_feat = None, None
    if cnn_cfg:
        n_cnn_feat = cnn_cfg["n_feat"]
        cnn = SimpleCNN(**cnn_cfg)
        cnn.load_state_dict(torch.load(model_dir / "cnn.pt", map_location="cpu"))
        cnn.eval()
    elif (model_dir / "cnn.pt").exists():
        state = torch.load(model_dir / "cnn.pt", map_location="cpu")
        n_cnn_feat = state["net.0.weight"].shape[1]
        cnn = SimpleCNN(n_feat=n_cnn_feat)
        cnn.load_state_dict(state)
        cnn.eval()

    # Klassische Modelle
    rf_list = load_pickle(model_dir / "rf.pkl") or []
    lgb_list = load_pickle(model_dir / "lgb.pkl") or []
    xgb_list = load_pickle(model_dir / "xgb.pkl") or []

    # Meta-Transformer
    meta_cfg = filter_cfg(J("meta_cfg.json", ["meta"]), MetaTransformer)
    meta_model = None
    if meta_cfg:
        meta_model = MetaTransformer(**meta_cfg)
        meta_model.load_state_dict(torch.load(model_dir / "meta.pt", map_location="cpu"))
        meta_model.eval()


    # ---------------- Flag: Regime-Features? --------------------
    use_regime = cfg_yaml.get("meta", {}).get("use_regime", False)

    # ---------------- Daten-Iterator vorbereiten ----------------
    data_iter = (
        iter_ticks_as_bars(test_file, bar_min)
        if "tick" in test_file.name.lower()
        else [pd.read_csv(test_file, parse_dates=["timestamp"]).set_index("timestamp")]
    )

    # ---------------- Evaluation ----------------
    seq_scaler = StandardScaler()
    y_true_all, y_prob_all = [], []
    comp_probs = {"ft": [], "cnn": [], "rf": [], "lgb": [], "xgb": []}

    for df_raw in data_iter:
        # ------- Spalten harmonisieren + Preis-NaNs füllen -------
        df_raw.rename(columns={c: "Close" for c in df_raw.columns if c.lower() == "close"}, inplace=True)
        if "price" not in df_raw.columns:
            df_raw["price"] = df_raw["Close"]
        df_raw["price"].replace(0, np.nan, inplace=True)
        df_raw["price"].ffill(inplace=True)

        # ---------------- Trend-Label ----------------
        trend = make_trend_labels(
            df_raw,
            cfg_yaml.get("dc_thres", 0.5),
            cfg_yaml.get("w_list", [5, 15, 30]),
            cfg_yaml.get("tau", 1),
        )
        df_raw["label"] = (trend["trend_side"] == 1).astype(int)


        # ------------- Regime-Features (ATR, %B) -------------
        if use_regime:
            regime_df = extract_regime_features(df_raw)
            regime_df = regime_df.loc[df_raw.index]         # align index

        # ---------------- Features -------------------
        df_feat = enrich_features(df_raw).dropna()

        # ---- exakte Trainings-Spalten laden ----
        feat_cols_file = model_dir / "feature_cols.json"
        if feat_cols_file.exists():
            feat_cols = json.load(open(feat_cols_file))
            for col in feat_cols:
                if col not in df_feat.columns:
                    df_feat[col] = 0.0
            df_feat = df_feat[feat_cols]

        # ---------------- Sequenzen ------------------
        # ---------------- Sequenzen ------------------
        feat_arr = df_feat.to_numpy(dtype=np.float32)
        lab_arr  = df_raw.loc[df_feat.index, "label"].to_numpy(np.float32)

        # Keine Sequenzen möglich? -> nächste Datei
        if len(feat_arr) <= seq_len:
            continue

        seqs   = [feat_arr[i - seq_len : i] for i in range(seq_len, len(feat_arr))]
        labels = lab_arr[seq_len:]

        if use_regime:
            reg_arr = regime_df.to_numpy(dtype=np.float32)
            reg_seqs = reg_arr[seq_len:]           # gleiche Länge wie labels

        # Scaler auf exakt derselben Flatten-Form fitten, die wir später transformieren
        seq_flat = np.reshape(seqs, (len(seqs), -1))
        seq_scaler.partial_fit(seq_flat)

        X_seq = torch.from_numpy(np.stack(seqs))                   # (N, seq_len, n_feat)
        y_seq = torch.from_numpy(labels.astype(np.float32))

        if use_regime:
            loader = DataLoader(
                TensorDataset(X_seq, y_seq,
                              torch.from_numpy(reg_seqs)),
                batch_size=512, shuffle=False)
        else:
            loader = DataLoader(TensorDataset(X_seq, y_seq),
                                batch_size=512, shuffle=False)

        # ---------------- Inferenz-Loop ---------------
        for xb, yb in loader:
            preds_components = []
    


            # Klassische Modelle
            xb_flat = seq_scaler.transform(xb.numpy().reshape(len(xb), -1))
            if rf_list:
                p_rf = mean_preds([m.predict_proba(xb_flat)[:, 1] for m in rf_list])
                comp_probs["rf"].append(p_rf); preds_components.append(p_rf)
            if lgb_list:
                p_lgb = mean_preds([m.predict(xb_flat) for m in lgb_list])
                comp_probs["lgb"].append(p_lgb); preds_components.append(p_lgb)
            if xgb_list:
                p_xgb = mean_preds([m.predict_proba(xb_flat)[:, 1] for m in xgb_list])
                comp_probs["xgb"].append(p_xgb); preds_components.append(p_xgb)

            # FT-Transformer  (mit Temperature-Scaling)
            if ft_wrap:
                logits_ft = ft_wrap(xb)
                if isinstance(logits_ft, dict):      # falls Wrapper Dict liefert
                    logits_ft = logits_ft["logits"]
                logits_ft = (logits_ft / T).detach().numpy().ravel()
                p_ft = 1 / (1 + np.exp(-logits_ft))      # Sigmoid
                comp_probs["ft"].append(p_ft); preds_components.append(p_ft)

            # CNN  (nur erste n_cnn_feat Features)
            if cnn:
                cnn_logits = cnn(xb[:, :, :n_cnn_feat].permute(0, 2, 1))
                cnn_logits = (cnn_logits / T).detach().numpy().ravel()
                p_cnn = 1 / (1 + np.exp(-cnn_logits))
                comp_probs["cnn"].append(p_cnn); preds_components.append(p_cnn)

            # Meta-Ensemble oder Durchschnitt
            if meta_model:
                stacked = np.vstack(preds_components).T.astype(np.float32)
                if use_regime:
                    # Regime-Features für dieses Batch anhängen
                    rb = loader.dataset.tensors[2]            # (N,2)
                    rb_batch = rb[loader.dataset.tensors[1]   # Zugriff via idx
                                          == yb].numpy()
                    stacked = np.hstack([stacked, rb_batch])
                p_meta = torch.sigmoid(
                    meta_model(torch.from_numpy(stacked))
                ).detach().numpy().ravel()
            else:
                p_meta = mean_preds(preds_components)

            y_true_all.append(yb.numpy())
            y_prob_all.append(p_meta)

    # ---------------- Kennzahlen ----------------
    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }
    cm = confusion_matrix(y_true, y_pred)
    cls = classification_report(y_true, y_pred, target_names=["no-trend", "up-trend"], output_dict=True)
    metrics.update({
        "precision": cls["up-trend"]["precision"],
        "recall":    cls["up-trend"]["recall"],
        "f1":        cls["up-trend"]["f1-score"],
        "confusion_matrix": cm.tolist(),
    })

    # -------------- Plots (optional) --------------
    fig_paths = {}
    if plots:
        import matplotlib.pyplot as plt, seaborn as sns, os
        out_dir = model_dir / "eval_figs"; out_dir.mkdir(exist_ok=True)

        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["no", "up"], yticklabels=["no", "up"])
        plt.title("Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Pred")
        cm_path = out_dir / "confusion_matrix.png"
        plt.tight_layout(); plt.savefig(cm_path); plt.close()
        fig_paths["confusion_matrix"] = str(cm_path)

        from sklearn.metrics import RocCurveDisplay, precision_recall_curve, auc
        RocCurveDisplay.from_predictions(y_true, y_prob)
        plt.title(f"ROC (AUC={metrics['roc_auc']:.3f})")
        roc_path = out_dir / "roc_curve.png"
        plt.tight_layout(); plt.savefig(roc_path); plt.close()
        fig_paths["roc_curve"] = str(roc_path)

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec, prec)
        plt.figure(); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"PR-Curve (AUC={pr_auc:.3f})")
        pr_path = out_dir / "pr_curve.png"
        plt.tight_layout(); plt.savefig(pr_path); plt.close()
        fig_paths["pr_curve"] = str(pr_path)

        metrics["pr_auc"] = pr_auc
        metrics["figures"] = fig_paths

    return metrics

# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__)
    )
    p.add_argument("--model-dir", required=True, nargs="+",
                   help="Pfad(e) zu trainierten Modell­verzeichnissen")
    p.add_argument("--test-file", required=True,
                   help="Testdatei (Ticks oder Kerzen-CSV/TXT)")
    p.add_argument("--seq-len", type=int, default=24,
                   help="Sequenz­länge für Zeitreihen­modelle")
    p.add_argument("--bar-min", type=int, default=60,
                   help="Resampling-Intervall in Minuten für Tick-Dateien")
    p.add_argument("--plots", action="store_true",
                   help="Speichert PNG-Plots in <model>/eval_figs")
    return p.parse_args()


def main():
    args = parse_args()
    results = {}
    for md in args.model_dir:
        md_path = Path(md)
        if not md_path.exists():
            print(f"[WARN] Model-Dir {md} existiert nicht – übersprungen.")
            continue
        print(f"\n⏳  Evaluating {md_path} ...")
        res = evaluate_one_model(
            md_path, Path(args.test_file),
            seq_len=args.seq_len, bar_min=args.bar_min,
            plots=args.plots
        )
        results[md] = res
        # Kurze Übersicht
        print(f"✔  {md}: acc={res['accuracy']:.3f}, "
              f"AUC={res['roc_auc']:.3f}, F1={res['f1']:.3f}")

    # Vergleichstabelle
    if len(results) > 1:
        print("\n===== Modellvergleich =====")
        header = f"{'Model':35s} |  Acc   |  AUC   |  F1"
        print(header); print("-"*len(header))
        for k, v in results.items():
            print(f"{k:35s} | {v['accuracy']:.4f} | "
                  f"{v['roc_auc']:.4f} | {v['f1']:.4f}")


if __name__ == "__main__":
    main()
