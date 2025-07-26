from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import optuna, numpy as np
from .base import BaseTrainer
from utils.data import read_raw_csv

class LongTrendTrainer(BaseTrainer):
    def load_data(self):
        df = read_raw_csv(Path(self.cfg["data"]["raw_dir"]) /
                          self.cfg["data"]["longtrend_file"])
        y  = df.pop("label").values
        return df, y

    def build_features(self, X):   # z. B. Rolling-Stats …
        return X.select_dtypes("number").values

    def optimize(self, X, y):
        def obj(trial):
            lr  = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            n   = trial.suggest_int("n_estimators", 100, 400)
            md  = trial.suggest_int("max_depth", 3, 8)
            m   = GradientBoostingClassifier(learning_rate=lr,
                                             n_estimators=n,
                                             max_depth=md)
            ll  = -cross_val_score(m, X, y, cv=5,
                                   scoring="neg_log_loss").mean()
            return ll
        study = optuna.create_study(direction="minimize")
        study.optimize(obj, n_trials=self.cfg["optuna"]["n_trials"])
        return study

    def train_final(self, X, y, best_params):
        model = GradientBoostingClassifier(**best_params)
        return model.fit(X, y)
