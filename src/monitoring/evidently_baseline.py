import os
from src.config.config import CONFIG


class EvidentlyBaseline:

    def run(self, df):

        os.makedirs(CONFIG["evidently_dir"], exist_ok=True)

        base = df.copy()
        base["text_len"] = df[CONFIG["text_col_clean"]].astype(str).str.len()

        out_path = os.path.join(CONFIG["evidently_dir"], "monitoring_reference.csv")
        base.to_csv(out_path, index=False)

        return out_path