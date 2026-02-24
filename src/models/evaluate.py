import os
import json
import numpy as np
import mlflow
from sklearn.metrics import f1_score
from src.config.config import CONFIG


class ModelEvaluator:

    def run(self, model, X_test, y_test):

        p_cat, p_pri, p_sent = model.predict(X_test, batch_size=CONFIG["batch_size"], verbose=0)

        y_pred = {
            "category": np.argmax(p_cat, axis=1),
            "priority": np.argmax(p_pri, axis=1),
            "sentiment": np.argmax(p_sent, axis=1)
        }

        metrics = {}
        for k in y_test.keys():
            metrics[f"{k}_f1_macro"] = float(f1_score(y_test[k], y_pred[k], average="macro"))

        os.makedirs(CONFIG["model_dir"], exist_ok=True)

        metrics_path = os.path.join(CONFIG["model_dir"], "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f)

        try:
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            mlflow.log_artifact(metrics_path, artifact_path="artifacts")
        except Exception:
            pass

        return metrics