CONFIG = {
    "raw_data_path": "data/raw/support_tickets.csv",
    "processed_dir": "data/processed",
    "model_dir": "artifacts/model",
    "tokenizer_dir": "artifacts/tokenizer",
    "shap_dir": "artifacts/shap",
    "evidently_dir": "artifacts/evidently",
    "reports_dir": "artifacts/reports",
    "logs_dir": "logs",
    "random_state": 42,
    "test_size": 0.10,
    "val_size": 0.10,
    "text_col_raw": "text",
    "text_col_clean": "clean_text",
    "targets": {
        "category": "category",
        "priority": "priority",
        "sentiment": "sentiment"
    },
    "max_vocab": 20000,
    "max_len": 80,
    "batch_size": 64,
    "epochs": 5,
    "lr": 1e-3,
    "mlflow_tracking_uri": "file:./mlruns",
    "mlflow_experiment": "support_ticket_triage",
    "mlflow_run_name": "multitask_lstm"
}