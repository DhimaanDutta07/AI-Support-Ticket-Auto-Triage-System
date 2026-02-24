import os
import pandas as pd
from datetime import datetime
from src.config.config import CONFIG


class InferenceLogger:

    def __init__(self, path=None):
        os.makedirs(CONFIG["logs_dir"], exist_ok=True)
        self.path = path or os.path.join(CONFIG["logs_dir"], "inference_logs.csv")

    def append(self, rows):

        df = pd.DataFrame(rows)
        df["logged_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        if os.path.exists(self.path):
            old = pd.read_csv(self.path)
            df = pd.concat([old, df], ignore_index=True)

        df.to_csv(self.path, index=False)
        return self.path