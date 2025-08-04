from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import yaml, joblib, numpy as np
from sklearn.model_selection import TimeSeriesSplit, KFold
import optuna, logging, random, os, torch   # ← torch u. os ergänzt

log = logging.getLogger(__name__)

class BaseTrainer(ABC):
    """Gemeinsames Grundgerüst – alle Trainer erben hiervon."""

    def __init__(self, cfg_path: str | Path):
        # ── Konfig einlesen ───────────────────────────────────
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # ── Seed & Zufälligkeit ───────────────────────────────
        self.seed = self.cfg.get("seed", 42)
        np.random.seed(self.seed)
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        # ── CV-Strategie ──────────────────────────────────────
        cv_cfg = self.cfg["cv"]
        if cv_cfg["type"] == "timeseries":
            self.cv = TimeSeriesSplit(**cv_cfg["params"])
        elif cv_cfg["type"] == "kfold":
            self.cv = KFold(**cv_cfg["params"])
        else:
            raise ValueError(f"Unknown CV type: {cv_cfg['type']}")

        # ── Modell-Tag & Device ───────────────────────────────
        self.model_tag = self.cfg["model"]["type"]          # "hybrid_longtrend", "entry", "exit"
        self.device    = torch.device("cpu")                # alles läuft auf CPU

        # ── Ausgabeordner vorbereiten ─────────────────────────
        #   Vermeidet Überschreiben zwischen den Trainern
        for p in ["opt_ft_tmp", "ft_final", "meta_runs"]:
            os.makedirs(f"{self.model_tag}_{p}", exist_ok=True)

        # ── Dateiname für finales Pickle ──────────────────────
        self.out_path = Path(self.cfg["model"].get("out_path",
                           f"{self.model_tag}_model.pkl"))

    # ─────────────────── Hooks für Subklassen ──────────────────────
    @abstractmethod
    def load_data(self): ...
    @abstractmethod
    def build_features(self, X): ...
    @abstractmethod
    def optimize(self, X, y): ...
    @abstractmethod
    def train_final(self, X, y, best_params): ...

    # ─────────────────── Convenience ───────────────────────────────
    def save_model(self, model):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.out_path)
        log.info("✅ Modell gespeichert unter %s", self.out_path)

    # ─────────────────── High-Level-Pipeline ───────────────────────
    def run(self):
        log.info("🚀 Starte Trainer %s …", self.__class__.__name__)
        X, y        = self.load_data()
        X_feat      = self.build_features(X)
        study       = self.optimize(X_feat, y)
        final_model = self.train_final(X_feat, y, study.best_params)
        self.save_model(final_model)
        return final_model, study
