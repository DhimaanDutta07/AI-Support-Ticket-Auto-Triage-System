from sklearn.model_selection import train_test_split
from src.config.config import CONFIG


class DataSplitting:

    def run(self, df):

        test_val = CONFIG["test_size"] + CONFIG["val_size"]

        train_df, temp_df = train_test_split(
            df,
            test_size=test_val,
            random_state=CONFIG["random_state"],
            stratify=df[CONFIG["targets"]["category"]]
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=CONFIG["test_size"] / test_val,
            random_state=CONFIG["random_state"],
            stratify=temp_df[CONFIG["targets"]["category"]]
        )

        return train_df, val_df, test_df