import os
import json
import mlflow
import mlflow.tensorflow
from src.config.config import CONFIG
from src.models.multitask_tf import MultiTaskModel


class ModelTrainer:

    def run(self, X_train, y_train, X_val, y_val, num_classes):

        mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
        mlflow.set_experiment(CONFIG["mlflow_experiment"])

        model = MultiTaskModel().build(num_classes)

        with mlflow.start_run(run_name=CONFIG["mlflow_run_name"]):

            mlflow.log_params({
                "epochs": CONFIG["epochs"],
                "batch_size": CONFIG["batch_size"],
                "lr": CONFIG["lr"],
                "max_vocab": CONFIG["max_vocab"],
                "max_len": CONFIG["max_len"],
                "num_category": num_classes["category"],
                "num_priority": num_classes["priority"],
                "num_sentiment": num_classes["sentiment"]
            })

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=CONFIG["epochs"],
                batch_size=CONFIG["batch_size"],
                verbose=1
            )

            os.makedirs(CONFIG["model_dir"], exist_ok=True)

            model_path = os.path.join(CONFIG["model_dir"], "model.keras")
            model.save(model_path)

            history_path = os.path.join(CONFIG["model_dir"], "history.json")
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history.history, f)

            for k, v in history.history.items():
                if len(v) > 0:
                    mlflow.log_metric(k, float(v[-1]))

            mlflow.log_artifact(history_path, artifact_path="artifacts")
            mlflow.log_artifact(model_path, artifact_path="artifacts")

            try:
                mlflow.tensorflow.log_model(model, artifact_path="model")
            except Exception:
                pass

        return model