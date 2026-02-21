"""
Prediction module.

Loads saved model artifacts and provides prediction functions for
single queries and batch predictions.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


class RAGPredictor:
    """Load trained models and make predictions."""

    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.features = None
        self.metrics = None
        self._load_all()

    def _load_all(self):
        """Load all saved model artifacts."""
        # Feature list
        feat_path = os.path.join(MODELS_DIR, "feature_list.json")
        if os.path.exists(feat_path):
            with open(feat_path) as f:
                self.features = json.load(f)

        # Models
        model_files = {
            "correctness": "xgb_correctness.joblib",
            "hallucination": "xgb_hallucination.joblib",
            "faithfulness": "xgb_faithfulness.joblib",
            "latency": "xgb_latency.joblib",
            "cost": "xgb_cost.joblib",
            "multiclass": "xgb_multiclass.joblib",
            "task_classifier": "tfidf_task_classifier.joblib",
        }
        for name, filename in model_files.items():
            path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)

        # Label encoders
        encoder_files = {
            "correctness": "le_correctness.joblib",
            "faithfulness": "le_faithfulness.joblib",
            "task_type": "le_task_type.joblib",
        }
        for name, filename in encoder_files.items():
            path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(path):
                self.encoders[name] = joblib.load(path)

        # Special feature lists for regression (leakage-safe)
        for name in ["latency_features", "cost_features"]:
            path = os.path.join(MODELS_DIR, f"{name}.joblib")
            if os.path.exists(path):
                self.models[f"{name}_list"] = joblib.load(path)

        # Metrics
        metrics_path = os.path.join(MODELS_DIR, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self.metrics = json.load(f)

        loaded = [k for k in self.models if not k.endswith("_list")]
        print(f"  Loaded {len(loaded)} models: {loaded}")

    def predict_correctness(self, X: pd.DataFrame) -> dict:
        """Predict binary correctness probability.

        Returns:
            Dict with 'prediction' (0/1) and 'probability' (float).
        """
        clf = self.models["correctness"]
        pred = clf.predict(X)
        proba = clf.predict_proba(X)[:, 1]
        return {"prediction": pred.tolist(), "probability": proba.tolist()}

    def predict_hallucination(self, X: pd.DataFrame) -> dict:
        """Predict hallucination risk.

        Returns:
            Dict with 'prediction' (0/1) and 'probability' (float).
        """
        clf = self.models["hallucination"]
        pred = clf.predict(X)
        proba = clf.predict_proba(X)[:, 1]
        return {"prediction": pred.tolist(), "probability": proba.tolist()}

    def predict_latency(self, X: pd.DataFrame) -> dict:
        """Predict latency in ms."""
        feat = self.models.get("latency_features_list", self.features)
        cols = [c for c in feat if c in X.columns]
        reg = self.models["latency"]
        pred = reg.predict(X[cols])
        return {"predicted_ms": pred.tolist()}

    def predict_cost(self, X: pd.DataFrame) -> dict:
        """Predict cost in USD."""
        feat = self.models.get("cost_features_list", self.features)
        cols = [c for c in feat if c in X.columns]
        reg = self.models["cost"]
        pred = reg.predict(X[cols])
        return {"predicted_usd": pred.tolist()}

    def predict_all(self, X: pd.DataFrame) -> dict:
        """Run all predictions on input features.

        Returns:
            Combined dict with correctness, hallucination, latency, cost.
        """
        return {
            "correctness": self.predict_correctness(X),
            "hallucination": self.predict_hallucination(X),
            "latency": self.predict_latency(X),
            "cost": self.predict_cost(X),
        }

    def get_metrics(self) -> dict:
        """Return saved model evaluation metrics."""
        return self.metrics or {}

    def classify_query(self, query_text: str) -> str:
        """Classify a query text into task_type."""
        clf = self.models["task_classifier"]
        le = self.encoders["task_type"]
        pred = clf.predict([query_text])
        return le.inverse_transform(pred)[0]


def load_predictor() -> RAGPredictor:
    """Convenience function to load the predictor."""
    return RAGPredictor()
