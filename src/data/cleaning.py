import re
from src.config.config import CONFIG


class DataCleaning:

    def clean(self, text):

        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z ]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def run(self, df):

        df[CONFIG["text_col_clean"]] = df[CONFIG["text_col_raw"]].apply(self.clean)
        return df