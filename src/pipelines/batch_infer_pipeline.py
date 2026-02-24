import os
import numpy as np
import tensorflow as tf
from src.config.config import CONFIG
from src.features.tokenizer import TextTokenizer
from src.data.cleaning import DataCleaning
from src.utils.common import load_json
import keras


class BatchInferPipeline:

    def run(self, df):

        tok = TextTokenizer().load()

        model = keras.models.load_model(
            os.path.join(CONFIG["model_dir"], "model.keras")
        )

        labels = load_json(
            os.path.join(CONFIG["model_dir"], "label_maps.json")
        )

        cleaner = DataCleaning()

        texts = df[
            CONFIG["text_col_raw"]
        ].astype(str).apply(cleaner.clean).values

        X = tok.transform(texts)

        p_cat, p_pri, p_sent = model.predict(
            X,
            batch_size=CONFIG["batch_size"],
            verbose=0
        )

        out = df.copy()

        out["pred_category"] = [
            labels["category"][i]
            for i in np.argmax(p_cat, axis=1)
        ]

        out["pred_priority"] = [
            labels["priority"][i]
            for i in np.argmax(p_pri, axis=1)
        ]

        out["pred_sentiment"] = [
            labels["sentiment"][i]
            for i in np.argmax(p_sent, axis=1)
        ]

        out["conf_category"] = np.max(p_cat, axis=1)
        out["conf_priority"] = np.max(p_pri, axis=1)
        out["conf_sentiment"] = np.max(p_sent, axis=1)

        os.makedirs(CONFIG["reports_dir"], exist_ok=True)

        out_path = os.path.join(
            CONFIG["reports_dir"],
            "batch_predictions.csv"
        )

        out.to_csv(out_path, index=False)

        return out_path