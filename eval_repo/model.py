"""Model training and evaluation."""
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


class ModelTrainer:
    """Wraps a RandomForestClassifier with save/load helpers."""

    def __init__(self, n_estimators: int = 200):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    def train(self, X_train, y_train):
        """Fit the underlying classifier on the training split."""
        self.model.fit(X_train, y_train)
        return self.model

    def save(self, path: str):
        """Persist the trained model to disk with joblib."""
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load a previously trained model from disk."""
        self.model = joblib.load(path)
        return self.model


def evaluate_model(model, X_test, y_test) -> dict:
    """Score a trained model on held-out data, returning accuracy and F1."""
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }


def cross_validate(trainer: ModelTrainer, folds: list) -> list:
    """Run evaluate_model across a list of (X_train, y_train, X_test, y_test) folds."""
    scores = []
    for X_train, y_train, X_test, y_test in folds:
        trainer.train(X_train, y_train)
        scores.append(evaluate_model(trainer.model, X_test, y_test))
    return scores
