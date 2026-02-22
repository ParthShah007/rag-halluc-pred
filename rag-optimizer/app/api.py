"""
RAG-Optimize: FastAPI REST API v3

Enhanced with SHAP explanations, calibrated probabilities, input validation,
and structured prediction logging.

Run with: uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

app = FastAPI(
    title="RAG-Optimize API",
    description="Intelligent RAG Configuration Recommender — predict correctness, "
                "hallucination risk, latency, and cost for any RAG configuration. "
                "Now with SHAP explanations and calibrated probabilities.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy model loading (cached at startup) ──
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        from src.models.predict import RAGPredictor
        _predictor = RAGPredictor()
    return _predictor


# ── Pydantic Schemas ──

class QueryRequest(BaseModel):
    query: str
    class Config:
        json_schema_extra = {"example": {"query": "What causes inflation?"}}


class PredictRequest(BaseModel):
    """Full config for prediction."""
    query: str = Field(default="What causes inflation?")
    domain: str = Field(default="finance")
    task_type: str = Field(default="factual")
    difficulty: str = Field(default="medium")
    generator_model: str = Field(default="gpt-4o")
    embedding_model: str = Field(default="text-embedding-3-large")
    reranker_model: str = Field(default="none")
    retrieval_strategy: str = Field(default="hybrid")
    chunking_strategy: str = Field(default="fixed_500")
    eval_mode: str = Field(default="llm_judge")
    stop_reason: str = Field(default="eos")
    top1_score: float = Field(default=0.75)
    mean_retrieved_score: float = Field(default=0.55)
    recall_at_5: float = Field(default=1.0)
    recall_at_10: float = Field(default=1.0)
    mrr_at_10: float = Field(default=0.6)
    n_retrieved_chunks: int = Field(default=5)
    temperature: float = Field(default=0.3)
    top_p: float = Field(default=0.9)
    prompt_tokens: int = Field(default=800)
    answer_tokens: int = Field(default=150)
    context_window_tokens: int = Field(default=8192)
    max_new_tokens: int = Field(default=1024)
    latency_ms_retrieval: float = Field(default=150.0)
    latency_ms_generation: float = Field(default=800.0)
    total_latency_ms: float = Field(default=950.0)
    total_cost_usd: float = Field(default=0.01)
    has_relevant_in_top5: int = Field(default=1)
    has_relevant_in_top10: int = Field(default=1)
    has_answer_in_corpus: int = Field(default=1)
    is_noanswer_probe: int = Field(default=0)
    used_long_context_window: int = Field(default=0)
    answered_without_retrieval: int = Field(default=0)
    explain: bool = Field(default=True, description="Include SHAP explanations in response")


class ShapFeature(BaseModel):
    feature: str
    value: float
    impact: float


class PredictResponse(BaseModel):
    correctness: str
    correctness_confidence: float
    correctness_calibrated: bool
    hallucination_risk: str
    hallucination_probability: float
    hallucination_calibrated: bool
    estimated_latency_ms: float
    estimated_cost_usd: float
    query_task_type: str
    shap_correctness: list[ShapFeature] = []
    shap_hallucination: list[ShapFeature] = []
    warnings: list[str] = []


class PredictionResponse(BaseModel):
    task_type: str
    message: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    shap_explainers: int
    calibrated_models: list[str]
    version: str


class MetricsResponse(BaseModel):
    classification: list
    regression: list


# ── Endpoints ──

@app.get("/", tags=["info"])
def root():
    return {
        "name": "RAG-Optimize API",
        "version": "3.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/metrics", "/predict", "/classify-query"],
        "features": ["shap_explanations", "calibrated_probabilities", "input_validation", "structured_logging"],
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    try:
        pred = get_predictor()
        n_models = len([k for k in pred.models if not k.endswith("_list")])
        calibrated = [k for k in pred.models if k.endswith("_calibrated")]
        return HealthResponse(
            status="healthy",
            models_loaded=n_models,
            shap_explainers=len(pred.shap_explainers),
            calibrated_models=calibrated,
            version="3.0.0",
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Models not available: {e}")


@app.get("/metrics", tags=["models"])
def get_metrics():
    try:
        pred = get_predictor()
        return pred.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse, tags=["predictions"])
def predict(request: PredictRequest):
    """Predict correctness, hallucination, latency, and cost for a full RAG config.

    Includes SHAP explanations showing which features most influenced each prediction,
    calibrated probability estimates, and input validation warnings.
    """
    try:
        from src.features.build_features import transform_single

        pred = get_predictor()
        config = request.model_dump()
        explain = config.pop("explain", True)

        X, validation_warnings = transform_single(config)

        # Run all predictions with SHAP
        results = pred.predict_all(X, config=config, explain=explain)

        # Task type
        task_type = pred.classify_query(config["query"])

        return PredictResponse(
            correctness=results["correctness"]["label"],
            correctness_confidence=results["correctness"]["probability"],
            correctness_calibrated=results["correctness"]["calibrated"],
            hallucination_risk=results["hallucination"]["label"],
            hallucination_probability=results["hallucination"]["probability"],
            hallucination_calibrated=results["hallucination"]["calibrated"],
            estimated_latency_ms=results["latency"]["predicted_ms"],
            estimated_cost_usd=results["cost"]["predicted_usd"],
            query_task_type=task_type,
            shap_correctness=[ShapFeature(**f) for f in results.get("shap_correctness", [])],
            shap_hallucination=[ShapFeature(**f) for f in results.get("shap_hallucination", [])],
            warnings=validation_warnings,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.post("/classify-query", response_model=PredictionResponse, tags=["predictions"])
def classify_query(request: QueryRequest):
    """Classify a query into its task type."""
    try:
        pred = get_predictor()
        task_type = pred.classify_query(request.query)
        return PredictionResponse(task_type=task_type, message=f"Classified as '{task_type}'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {e}")


@app.get("/models", tags=["models"])
def list_models():
    try:
        pred = get_predictor()
        models = [k for k in pred.models if not k.endswith("_list")]
        return {
            "models": models,
            "count": len(models),
            "calibrated": [k for k in models if k.endswith("_calibrated")],
            "shap_enabled": list(pred.shap_explainers.keys()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
