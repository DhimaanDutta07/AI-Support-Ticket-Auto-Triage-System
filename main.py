from src.pipelines.train_pipeline import TrainPipeline
import pandas as pd
from src.data.ingestion import DataIngestion
from src.data.cleaning import DataCleaning
from src.pipelines.monitoring_pipeline import MonitoringPipeline

if __name__ == "__main__":
    TrainPipeline().run()

    df = DataIngestion().run()
    df = DataCleaning().run(df)

    reference_df = df.sample(20000, random_state=42)
    current_df = df.sample(5000, random_state=1)

    path = MonitoringPipeline().run(
        reference_df,
        current_df
    )
    print(path)