import os
from src.config.config import CONFIG


class EvidentlyCurrent:

    def run(self, df):

        os.makedirs(CONFIG["logs_dir"], exist_ok=True)

        cur = df.copy()
        cur["text_len"] = df[CONFIG["text_col_clean"]].astype(str).str.len()

        out_path = os.path.join(CONFIG["logs_dir"], "monitoring_current.csv")
        cur.to_csv(out_path, index=False)

        return out_path