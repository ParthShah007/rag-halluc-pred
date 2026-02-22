"""Quick test for API v3 enhancements."""
import requests
import json

BASE = "http://localhost:8000"

# Test 1: Health
print("=== HEALTH ===")
r = requests.get(f"{BASE}/health")
h = r.json()
print(json.dumps(h, indent=2))
assert h["status"] == "healthy"
assert h["models_loaded"] >= 7
assert h["shap_explainers"] >= 2
assert len(h["calibrated_models"]) >= 2
print("PASS: Health check\n")

# Test 2: Predict with SHAP
print("=== PREDICT WITH SHAP ===")
r = requests.post(f"{BASE}/predict", json={
    "query": "What causes inflation?",
    "generator_model": "gpt-4o",
    "retrieval_strategy": "hybrid",
    "top1_score": 0.85,
    "explain": True,
})
data = r.json()
print(f"Correctness: {data['correctness']} (conf={data['correctness_confidence']}, calibrated={data['correctness_calibrated']})")
print(f"Hallucination: {data['hallucination_risk']} (prob={data['hallucination_probability']}, calibrated={data['hallucination_calibrated']})")
print(f"Latency: {data['estimated_latency_ms']}ms")
print(f"Cost: ${data['estimated_cost_usd']}")
print(f"Task type: {data['query_task_type']}")
print(f"Warnings: {data['warnings']}")
print(f"SHAP correctness: {len(data['shap_correctness'])} features")
for f in data["shap_correctness"][:5]:
    print(f"  {f['feature']:30s}  val={f['value']:8.4f}  impact={f['impact']:+.4f}")
print(f"SHAP hallucination: {len(data['shap_hallucination'])} features")
for f in data["shap_hallucination"][:5]:
    print(f"  {f['feature']:30s}  val={f['value']:8.4f}  impact={f['impact']:+.4f}")
assert data["correctness_calibrated"] == True
assert data["hallucination_calibrated"] == True
assert len(data["shap_correctness"]) == 10
assert len(data["shap_hallucination"]) == 10
print("PASS: Predict with SHAP\n")

# Test 3: Predict with unseen categorical
print("=== UNSEEN CATEGORICAL ===")
r = requests.post(f"{BASE}/predict", json={
    "query": "What is quantum computing?",
    "generator_model": "claude-3-opus",
    "top1_score": 0.5,
    "explain": False,
})
data = r.json()
print(f"Warnings: {data['warnings']}")
has_unseen = any("unseen" in w.lower() for w in data["warnings"])
print(f"Detected unseen categorical: {has_unseen}")
assert has_unseen, "Should warn about unseen generator_model"
print("PASS: Unseen categorical handled\n")

# Test 4: Check logs
import os
log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "predictions.jsonl")
if os.path.exists(log_path):
    with open(log_path) as f:
        lines = f.readlines()
    print(f"=== LOGS: {len(lines)} prediction(s) logged ===")
    if lines:
        last = json.loads(lines[-1])
        print(f"Last log: timestamp={last['timestamp']}, latency={last['latency_ms']}ms")
        print(f"Predictions: {last['predictions']}")
    assert len(lines) >= 2, "Should have at least 2 logged predictions"
    print("PASS: Structured logging\n")

print("=" * 50)
print("ALL TESTS PASSED!")
