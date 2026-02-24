import os
import joblib
import shap
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.config.config import CONFIG


class ShapTextExplainer:

    def __init__(self):
        self.models = {}
        self.bg = {}
        self.explainers = {}

    def fit(self, train_df):

        X = train_df[CONFIG["text_col_clean"]].astype(str).values

        for name, col in CONFIG["targets"].items():

            y = train_df[col].astype(str).values

            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=2000))
            ])

            pipe.fit(X, y)

            self.models[name] = pipe

            n = min(200, len(X))
            self.bg[name] = list(X[:n])

        return self

    def save(self):
        os.makedirs(CONFIG["shap_dir"], exist_ok=True)
        joblib.dump(
            {"models": self.models, "bg": self.bg},
            os.path.join(CONFIG["shap_dir"], "shap_models.joblib")
        )

    def load(self):
        obj = joblib.load(os.path.join(CONFIG["shap_dir"], "shap_models.joblib"))
        self.models = obj["models"]
        self.bg = obj["bg"]
        return self

    def explain(self, text, task="category", top_k=10):

        pipe = self.models[task]

        if task not in self.explainers:
            masker = shap.maskers.Text(None)
            self.explainers[task] = shap.Explainer(pipe.predict_proba, masker=masker)

        exp = self.explainers[task]([str(text)])

        vals = exp.values[0]
        toks = exp.data[0]

        if vals.ndim == 2:
            vals = vals[:, 0]

        pairs = [(t, float(v)) for t, v in zip(toks, vals) if str(t).strip()]

        pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        return [{"token": t, "impact": v} for t, v in pairs[:top_k]]