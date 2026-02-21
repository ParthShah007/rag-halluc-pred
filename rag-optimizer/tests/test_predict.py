"""Quick test to verify interactive prediction pipeline."""
import sys
sys.path.insert(0, ".")
from src.features.build_features import transform_single
import joblib

config = {
    "domain": "finance", "task_type": "factual", "difficulty": "medium",
    "generator_model": "gpt-4o", "embedding_model": "text-embedding-3-large",
    "reranker_model": "none", "retrieval_strategy": "hybrid",
    "chunking_strategy": "fixed_500", "eval_mode": "llm_judge", "stop_reason": "eos",
    "top1_score": 0.85, "mean_retrieved_score": 0.6, "recall_at_5": 1.0,
    "recall_at_10": 1.0, "mrr_at_10": 0.7, "n_retrieved_chunks": 5,
    "temperature": 0.3, "top_p": 0.9, "prompt_tokens": 800, "answer_tokens": 150,
    "context_window_tokens": 8192, "max_new_tokens": 1024,
    "latency_ms_retrieval": 150, "latency_ms_generation": 800,
    "total_latency_ms": 950, "total_cost_usd": 0.01,
    "has_relevant_in_top5": 1, "has_relevant_in_top10": 1,
    "has_answer_in_corpus": 1, "is_noanswer_probe": 0,
    "used_long_context_window": 0, "answered_without_retrieval": 0,
    "query": "What causes inflation?",
}

X = transform_single(config)
print(f"Feature vector shape: {X.shape}")

# Correctness
m = joblib.load("models/xgb_correctness.joblib")
t = joblib.load("models/threshold_correctness.joblib")
p = m.predict_proba(X)[:, 1][0]
label = "CORRECT" if p >= t else "INCORRECT"
print(f"Correctness: {p:.4f} (threshold={t:.3f}) -> {label}")

# Hallucination
mh = joblib.load("models/xgb_hallucination.joblib")
th = joblib.load("models/threshold_hallucination.joblib")
ph = mh.predict_proba(X)[:, 1][0]
hl = "HIGH RISK" if ph >= th else "LOW RISK"
print(f"Hallucination: {ph:.4f} (threshold={th:.3f}) -> {hl}")

# Latency
lat = joblib.load("models/xgb_latency.joblib")
lf = joblib.load("models/latency_features.joblib")
lc = [c for c in lf if c in X.columns]
print(f"Latency: {lat.predict(X[lc])[0]:.0f} ms")

# Cost
cost = joblib.load("models/xgb_cost.joblib")
cf = joblib.load("models/cost_features.joblib")
cc = [c for c in cf if c in X.columns]
print(f"Cost: ${cost.predict(X[cc])[0]:.5f}")

# Task type
from src.models.predict import RAGPredictor
pred = RAGPredictor()
print(f"Task type: {pred.classify_query('What causes inflation?')}")
print("\n=== ALL PREDICTIONS WORKED ===")
