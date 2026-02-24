import os
import json
import mlflow
from src.config.config import CONFIG


class ModelExporter:

    def run(self, feature_builder):

        os.makedirs(CONFIG["model_dir"], exist_ok=True)

        label_maps = {name: enc.classes_.tolist() for name, enc in feature_builder.encoders.items()}

        out_path = os.path.join(CONFIG["model_dir"], "label_maps.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(label_maps, f)

        try:
            mlflow.log_artifact(out_path, artifact_path="artifacts")
        except Exception:
            pass

        return label_maps