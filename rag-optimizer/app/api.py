"""
RAG-Optimize: FastAPI REST API v2

Run with: uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

app = FastAPI(
    title="RAG-Optimize API",
    description="Intelligent RAG Configuration Recommender — predict correctness, "
                "hallucination risk, latency, and cost for any RAG configuration.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy model loading ──
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
    top1_score: float = Field(default=0.75, ge=0, le=1)
    mean_retrieved_score: float = Field(default=0.55, ge=0, le=1)
    recall_at_5: float = Field(default=1.0)
    recall_at_10: float = Field(default=1.0)
    mrr_at_10: float = Field(default=0.6, ge=0, le=1)
    n_retrieved_chunks: int = Field(default=5, ge=1, le=50)
    temperature: float = Field(default=0.3, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
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


class PredictResponse(BaseModel):
    correctness: str
    correctness_confidence: float
    hallucination_risk: str
    hallucination_probability: float
    estimated_latency_ms: float
    estimated_cost_usd: float
    query_task_type: str


class PredictionResponse(BaseModel):
    task_type: str
    message: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    version: str


class MetricsResponse(BaseModel):
    classification: list
    regression: list


# ── Endpoints ──

@app.get("/", tags=["info"])
def root():
    return {
        "name": "RAG-Optimize API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/metrics", "/predict", "/classify-query"],
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    try:
        pred = get_predictor()
        n_models = len([k for k in pred.models if not k.endswith("_list")])
        return HealthResponse(status="healthy", models_loaded=n_models, version="2.0.0")
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
    """Predict correctness, hallucination, latency, and cost for a full RAG config."""
    try:
        import joblib
        from src.features.build_features import transform_single

        pred = get_predictor()
        config = request.model_dump()
        X = transform_single(config)

        # Correctness
        corr_model = pred.models.get("correctness")
        thresh_path = os.path.join(PROJECT_ROOT, "models", "threshold_correctness.joblib")
        threshold = joblib.load(thresh_path) if os.path.exists(thresh_path) else 0.5
        corr_proba = float(corr_model.predict_proba(X)[:, 1][0])

        # Hallucination
        hal_model = pred.models.get("hallucination")
        thresh_path_h = os.path.join(PROJECT_ROOT, "models", "threshold_hallucination.joblib")
        threshold_h = joblib.load(thresh_path_h) if os.path.exists(thresh_path_h) else 0.5
        hal_proba = float(hal_model.predict_proba(X)[:, 1][0])

        # Latency
        lat_feats = pred.models.get("latency_features_list", [])
        lat_cols = [c for c in lat_feats if c in X.columns]
        lat_val = float(pred.models["latency"].predict(X[lat_cols])[0])

        # Cost
        cost_feats = pred.models.get("cost_features_list", [])
        cost_cols = [c for c in cost_feats if c in X.columns]
        cost_val = float(pred.models["cost"].predict(X[cost_cols])[0])

        # Task type
        task_type = pred.classify_query(config["query"])

        return PredictResponse(
            correctness="CORRECT" if corr_proba >= threshold else "INCORRECT",
            correctness_confidence=round(corr_proba, 4),
            hallucination_risk="HIGH" if hal_proba >= threshold_h else "LOW",
            hallucination_probability=round(hal_proba, 4),
            estimated_latency_ms=round(lat_val, 2),
            estimated_cost_usd=round(cost_val, 6),
            query_task_type=task_type,
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
        return {"models": models, "count": len(models)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
