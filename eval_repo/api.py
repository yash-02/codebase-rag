"""Thin HTTP-style handlers wrapping the training/inference pipeline."""
from pipeline import run_training_pipeline, run_inference


def predict_endpoint(sample):
    """Handle a prediction request by running inference on one sample."""
    return run_inference("models/churn_model.joblib", sample)


def retrain_endpoint(path: str):
    """Handle a retrain request by re-running the full training pipeline."""
    return run_training_pipeline(path, target_col="Churn", model_out_path="models/churn_model.joblib")
