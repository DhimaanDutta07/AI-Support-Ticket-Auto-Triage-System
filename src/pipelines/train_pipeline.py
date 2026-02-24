from src.data.ingestion import DataIngestion
from src.data.validation import DataValidation
from src.data.cleaning import DataCleaning
from src.data.splitting import DataSplitting
from src.features.build_features import FeatureBuilder
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.models.export import ModelExporter
from src.explain.shap_text import ShapTextExplainer
from src.monitoring.evidently_baseline import EvidentlyBaseline


class TrainPipeline:

    def run(self):

        df = DataIngestion().run()
        df = DataValidation().run(df)
        df = DataCleaning().run(df)

        train_df, val_df, test_df = DataSplitting().run(df)

        fb = FeatureBuilder().fit(train_df)

        X_train, y_train = fb.transform(train_df)
        X_val, y_val = fb.transform(val_df)
        X_test, y_test = fb.transform(test_df)

        model = ModelTrainer().run(X_train, y_train, X_val, y_val, fb.get_num_classes())
        metrics = ModelEvaluator().run(model, X_test, y_test)

        ModelExporter().run(fb)

        ShapTextExplainer().fit(train_df).save()
        EvidentlyBaseline().run(train_df)

        return metrics