"""Calibrated stacking of the component scorers.

s(c|x) = sigmoid(w^T [s_knn, s_cls, s_txt, s_meta, ...] + b), fitted on the
validation split by logistic regression over each image's candidate set
(the union of every scorer's top candidates, a few hundred categories).
A gradient-boosted alternative is provided for comparison.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

FEATURES: Tuple[str, ...] = (
    "knn",            # similarity-weighted k-NN vote
    "knn_maxsim",     # best similarity among neighbours carrying the category
    "knn_frac",       # fraction of the K neighbours carrying it
    "cls",            # supervised head probability (0 if not a head category)
    "in_head",        # category has a head output
    "txt",            # cosine(image, category text prototype)
    "txt_rel",        # txt minus the best txt score for this image
    "meta",           # cosine(title+description text, category prototype)
    "meta_rel",       # meta minus the best meta score for this image
    "lex",            # fraction of category-name tokens found in title/description
    "year_match",     # category year == photo year
    "year_mismatch",  # category year != photo year
    "geo",            # distance-weighted vote of nearby geotagged images
    "log_prior",      # log(1 + training frequency)
)


class FusionModel:
    def __init__(self, kind: str, model, feature_names: Sequence[str] = FEATURES):
        self.kind = kind
        self.model = model
        self.feature_names = list(feature_names)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, kind: str = "logreg", C: float = 1.0,
            seed: int = 0) -> "FusionModel":
        if len(np.unique(y)) < 2:
            raise ValueError("fusion training data needs both positive and negative candidates")
        if kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            model = Pipeline([("scale", StandardScaler()),
                              ("lr", LogisticRegression(C=C, max_iter=2000))])
        elif kind == "gbdt":
            from sklearn.ensemble import HistGradientBoostingClassifier

            model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                                   l2_regularization=1.0, early_stopping=True,
                                                   validation_fraction=0.1, random_state=seed)
        else:
            raise ValueError("unknown fusion kind %r" % kind)
        model.fit(X, y)
        return cls(kind, model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.zeros(0, np.float32)
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    def coefficients(self) -> Dict[str, float]:
        """Standardised logistic-regression weights (transparency report)."""
        if self.kind != "logreg":
            return {}
        lr = self.model.named_steps["lr"]
        out = {name: float(w) for name, w in zip(self.feature_names, lr.coef_[0])}
        out["(intercept)"] = float(lr.intercept_[0])
        return out

    def contributions(self, x: np.ndarray) -> List[Tuple[str, float]]:
        """Per-feature contribution to the logit for one candidate (logreg only)."""
        if self.kind != "logreg":
            return []
        scaler = self.model.named_steps["scale"]
        lr = self.model.named_steps["lr"]
        z = (x - scaler.mean_) / scaler.scale_
        contrib = z * lr.coef_[0]
        return sorted(zip(self.feature_names, contrib.tolist()), key=lambda t: -abs(t[1]))

    def save(self, path: str) -> None:
        import joblib

        joblib.dump({"kind": self.kind, "model": self.model, "features": self.feature_names}, path)

    @classmethod
    def load(cls, path: str) -> "FusionModel":
        import joblib

        d = joblib.load(path)
        if list(d["features"]) != list(FEATURES):
            raise RuntimeError("fusion model was trained with different features; re-run s07_fit_fusion.py")
        return cls(d["kind"], d["model"], d["features"])
