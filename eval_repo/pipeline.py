"""End-to-end training and inference pipeline, tying data_utils + model together."""
import joblib

from data_utils import prepare_training_data
from model import ModelTrainer, evaluate_model


def run_training_pipeline(path: str, target_col: str, model_out_path: str) -> dict:
    """Prepare data, train a model, evaluate it, and save it to disk."""
    X_train, X_test, y_train, y_test = prepare_training_data(path, target_col)
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    trainer.save(model_out_path)
    return evaluate_model(trainer.model, X_test, y_test)


def run_inference(model_path: str, sample):
    """Load a saved model and predict the churn probability for one sample."""
    model = joblib.load(model_path)
    return model.predict_proba(sample)[0][1]
