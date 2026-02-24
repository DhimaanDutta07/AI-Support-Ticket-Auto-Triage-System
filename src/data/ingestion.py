import pandas as pd
from src.config.config import CONFIG


class DataIngestion:

    def run(self):
        return pd.read_csv(CONFIG["raw_data_path"])