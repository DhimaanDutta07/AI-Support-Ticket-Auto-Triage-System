def test_import():

    from src.pipelines.train_pipeline import TrainPipeline

    assert TrainPipeline is not None