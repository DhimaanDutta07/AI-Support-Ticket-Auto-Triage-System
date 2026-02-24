import os
import pandas as pd
from src.config.config import CONFIG
from evidently import Report
from evidently.presets import DataSummaryPreset, DataDriftPreset


class EvidentlyReport:

    def __init__(self, curr_path=None, ref_path=None):
        self.curr_path = curr_path or os.path.join(CONFIG["logs_dir"], "monitoring_current.csv")
        self.ref_path = ref_path or os.path.join(CONFIG["evidently_dir"], "monitoring_reference.csv")

    def get_report(self):

        ref_ds = pd.read_csv(self.ref_path)
        cur_ds = pd.read_csv(self.curr_path)

        rep = Report([DataSummaryPreset(), DataDriftPreset()])
        snap = rep.run(cur_ds, ref_ds)

        out_dir = os.path.join(CONFIG["evidently_dir"], "monitoring")
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, "evidently_report.html")
        snap.save_html(out_path)

        return out_path