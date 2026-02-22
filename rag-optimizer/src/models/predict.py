"""
Prediction module — v2 with SHAP, calibration, caching, and structured logging.

Loads saved model artifacts at startup and provides prediction functions
with real-time SHAP explanations, calibrated probabilities, and JSON logging.
"""

import os
import json
import time
import logging
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# ── Structured Logging Setup ──

os.makedirs(LOGS_DIR, exist_ok=True)

# JSON-lines logger for predictions
_pred_logger = logging.getLogger("rag_predictions")
_pred_logger.setLevel(logging.INFO)
_pred_logger.propagate = False

if not _pred_logger.handlers:
    _handler = logging.FileHandler(
        os.path.join(LOGS_DIR, "predictions.jsonl"), encoding="utf-8"
    )
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _pred_logger.addHandler(_handler)


def _log_prediction(input_config: dict, predictions: dict, latency_ms: float, warnings: list):
    """Log a prediction to the JSONL file."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_hash": hashlib.md5(json.dumps(input_config, sort_keys=True, default=str).encode()).hexdigest()[:12],
        "predictions": predictions,
        "latency_ms": round(latency_ms, 2),
        "warnings": warnings,
    }
    try:
        _pred_logger.info(json.dumps(entry, default=str))
    except Exception:
        pass  # Never let logging break predictions


class RAGPredictor:
    """Load trained models and make predictions with SHAP, calibration, and caching."""

    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.features = None
        self.metrics = None
        self.thresholds = {}
        self.shap_explainers = {}
        self._categories = None
        self._load_all()

    def _load_all(self):
        """Load all saved model artifacts at startup (cached, no per-request I/O)."""
        # Feature list
        feat_path = os.path.join(MODELS_DIR, "feature_list.json")
        if os.path.exists(feat_path):
            with open(feat_path) as f:
                self.features = json.load(f)

        # One-hot categories (cached at startup)
        cat_path = os.path.join(MODELS_DIR, "onehot_categories.json")
        if os.path.exists(cat_path):
            with open(cat_path) as f:
                self._categories = json.load(f)

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

        # Calibrated models (prefer these for probability estimates)
        for name in ["correctness", "hallucination"]:
            cal_path = os.path.join(MODELS_DIR, f"xgb_{name}_calibrated.joblib")
            if os.path.exists(cal_path):
                self.models[f"{name}_calibrated"] = joblib.load(cal_path)

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

        # Thresholds (cached at startup — no more per-request disk reads)
        for name in ["correctness", "hallucination"]:
            thresh_path = os.path.join(MODELS_DIR, f"threshold_{name}.joblib")
            if os.path.exists(thresh_path):
                self.thresholds[name] = joblib.load(thresh_path)
            else:
                self.thresholds[name] = 0.5

        # Metrics
        metrics_path = os.path.join(MODELS_DIR, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self.metrics = json.load(f)

        # SHAP explainers (created once at startup)
        self._init_shap_explainers()

        loaded = [k for k in self.models if not k.endswith("_list")]
        print(f"  Loaded {len(loaded)} models: {loaded}")
        print(f"  Thresholds: {self.thresholds}")
        print(f"  SHAP explainers: {list(self.shap_explainers.keys())}")

    def _init_shap_explainers(self):
        """Initialize SHAP TreeExplainers for tree-based models."""
        try:
            import shap
            for name in ["correctness", "hallucination"]:
                if name in self.models:
                    self.shap_explainers[name] = shap.TreeExplainer(self.models[name])
        except ImportError:
            print("  ⚠ SHAP not installed — explainability disabled")
        except Exception as e:
            print(f"  ⚠ SHAP init failed: {e}")

    def explain_prediction(self, X: pd.DataFrame, model_name: str, top_n: int = 10) -> list[dict]:
        """Compute SHAP explanations for a prediction.

        Args:
            X: Feature DataFrame (single row).
            model_name: Which model to explain ("correctness" or "hallucination").
            top_n: Number of top contributing features to return.

        Returns:
            List of dicts with 'feature', 'value', and 'impact' keys,
            sorted by absolute impact descending.
        """
        if model_name not in self.shap_explainers:
            return []

        try:
            explainer = self.shap_explainers[model_name]
            shap_values = explainer.shap_values(X)

            # Handle binary classification (may return list of [class_0, class_1])
            if isinstance(shap_values, list):
                sv = shap_values[1]  # class 1 (positive) explanations
            else:
                sv = shap_values

            sv_row = sv[0] if sv.ndim > 1 else sv
            feature_names = self.features or list(X.columns)

            # Get top-N by absolute impact
            abs_impacts = np.abs(sv_row)
            top_idx = np.argsort(abs_impacts)[::-1][:top_n]

            explanations = []
            for idx in top_idx:
                if idx < len(feature_names):
                    explanations.append({
                        "feature": feature_names[idx],
                        "value": round(float(X.iloc[0, idx]), 4),
                        "impact": round(float(sv_row[idx]), 4),
                    })
            return explanations
        except Exception:
            return []

    def predict_correctness(self, X: pd.DataFrame) -> dict:
        """Predict binary correctness with calibrated probabilities and threshold.

        Returns:
            Dict with 'prediction', 'probability', 'calibrated', 'label'.
        """
        # Use calibrated model if available, raw otherwise
        calibrated = f"correctness_calibrated" in self.models
        clf = self.models.get("correctness_calibrated", self.models["correctness"])

        proba = float(clf.predict_proba(X)[:, 1][0])
        threshold = self.thresholds.get("correctness", 0.5)
        label = "CORRECT" if proba >= threshold else "INCORRECT"

        return {
            "prediction": 1 if proba >= threshold else 0,
            "probability": round(proba, 4),
            "calibrated": calibrated,
            "label": label,
            "threshold": round(float(threshold), 3),
        }

    def predict_hallucination(self, X: pd.DataFrame) -> dict:
        """Predict hallucination risk with calibrated probabilities and threshold.

        Returns:
            Dict with 'prediction', 'probability', 'calibrated', 'label'.
        """
        calibrated = "hallucination_calibrated" in self.models
        clf = self.models.get("hallucination_calibrated", self.models["hallucination"])

        proba = float(clf.predict_proba(X)[:, 1][0])
        threshold = self.thresholds.get("hallucination", 0.5)
        label = "HIGH" if proba >= threshold else "LOW"

        return {
            "prediction": 1 if proba >= threshold else 0,
            "probability": round(proba, 4),
            "calibrated": calibrated,
            "label": label,
            "threshold": round(float(threshold), 3),
        }

    def predict_latency(self, X: pd.DataFrame) -> dict:
        """Predict latency in ms."""
        feat = self.models.get("latency_features_list", self.features)
        cols = [c for c in feat if c in X.columns]
        reg = self.models["latency"]
        pred = float(reg.predict(X[cols])[0])
        return {"predicted_ms": round(pred, 2)}

    def predict_cost(self, X: pd.DataFrame) -> dict:
        """Predict cost in USD."""
        feat = self.models.get("cost_features_list", self.features)
        cols = [c for c in feat if c in X.columns]
        reg = self.models["cost"]
        pred = float(reg.predict(X[cols])[0])
        return {"predicted_usd": round(pred, 6)}

    def predict_all(self, X: pd.DataFrame, config: dict = None, explain: bool = False) -> dict:
        """Run all predictions with optional SHAP explanations and logging.

        Args:
            X: Feature DataFrame (single row).
            config: Original input config (for logging).
            explain: If True, include SHAP explanations.

        Returns:
            Combined dict with all predictions, explanations, and metadata.
        """
        t0 = time.perf_counter()

        result = {
            "correctness": self.predict_correctness(X),
            "hallucination": self.predict_hallucination(X),
            "latency": self.predict_latency(X),
            "cost": self.predict_cost(X),
        }

        if explain:
            result["shap_correctness"] = self.explain_prediction(X, "correctness")
            result["shap_hallucination"] = self.explain_prediction(X, "hallucination")

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Structured logging
        if config is not None:
            log_data = {
                "correctness": result["correctness"]["label"],
                "correctness_prob": result["correctness"]["probability"],
                "hallucination": result["hallucination"]["label"],
                "hallucination_prob": result["hallucination"]["probability"],
                "latency_ms": result["latency"]["predicted_ms"],
                "cost_usd": result["cost"]["predicted_usd"],
            }
            _log_prediction(config, log_data, elapsed_ms, [])

        return result

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
