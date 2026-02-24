from sklearn.preprocessing import LabelEncoder
from src.config.config import CONFIG
from src.features.tokenizer import TextTokenizer


class FeatureBuilder:

    def __init__(self):
        self.tokenizer = TextTokenizer()
        self.encoders = {}

    def fit(self, train_df):

        self.tokenizer.fit(train_df[CONFIG["text_col_clean"]].astype(str).values)
        self.tokenizer.save()

        for name, col in CONFIG["targets"].items():
            le = LabelEncoder()
            le.fit(train_df[col].astype(str).values)
            self.encoders[name] = le

        return self

    def transform(self, df):

        X = self.tokenizer.transform(df[CONFIG["text_col_clean"]].astype(str).values)

        y = {}
        for name, col in CONFIG["targets"].items():
            y[name] = self.encoders[name].transform(df[col].astype(str).values)

        return X, y

    def get_num_classes(self):
        return {k: len(v.classes_) for k, v in self.encoders.items()}