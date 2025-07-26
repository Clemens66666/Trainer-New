# utils/purged_cv.py
from sklearn.model_selection import KFold

class PurgedKFold(KFold):
    """
    Minimal‑Ersatz für mlfinlab.cross_validation.PurgedKFold.
    - API kompatibel: init(n_splits, embargo_pct, **kw)
    - Kein echtes Purging / Embargo‑Handling; setzt aber dieselben Attribute,
      damit downstream‑Code nicht bricht.
    """
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.0, **kwargs):
        super().__init__(n_splits=n_splits, shuffle=False)
        self.embargo_pct = embargo_pct
        # alle weiteren kwargs verschwinden im **kwargs, damit Aufrufe identisch bleiben.
