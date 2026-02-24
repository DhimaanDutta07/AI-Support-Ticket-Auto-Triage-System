from src.monitoring.evidently_baseline import EvidentlyBaseline
from src.monitoring.evidently_drift import EvidentlyCurrent
from src.monitoring.evidently_report import EvidentlyReport


class MonitoringPipeline:

    def run(self, reference_df, current_df):

        ref_path = EvidentlyBaseline().run(reference_df)
        curr_path = EvidentlyCurrent().run(current_df)
        return EvidentlyReport(curr_path=curr_path, ref_path=ref_path).get_report()