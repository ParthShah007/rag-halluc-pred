# 🚀 RAG-Optimize: Intelligent RAG Configuration Recommender

> **DataHack 2026** — End-to-end ML pipeline for predicting and optimizing RAG system configurations.

## Problem Statement

Given a RAG (Retrieval-Augmented Generation) evaluation dataset, build an ML system that:
1. **Predicts correctness** — Will this configuration produce a correct answer?
2. **Detects hallucinations** — Will the LLM make things up?
3. **Estimates latency & cost** — How fast/expensive will it be?
4. **Recommends optimal configs** — Best retrieval + generator + embedding combination.
5. **Explains predictions** — SHAP-based explainability for every recommendation.

## Architecture

```
data/raw/  →  src/data/  →  src/features/  →  src/models/  →  app/
 (4 CSVs)    (load+merge)   (engineer)       (train+save)   (Streamlit + FastAPI)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models (saves to models/)
python -m src.models.train

# 3. Launch interactive dashboard
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# 4. Launch REST API
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
rag-optimizer/
├── app/
│   ├── api.py                      # FastAPI REST API
│   └── streamlit_app.py            # Interactive Streamlit dashboard
├── src/
│   ├── data/load_data.py           # Load & merge all 4 tables
│   ├── features/build_features.py  # Feature engineering (80 features)
│   ├── models/train.py             # Train 7 models with SMOTE
│   └── models/predict.py           # Inference & predictions
├── models/                         # Saved model artifacts (.joblib)
├── data/raw/                       # Original 4 CSVs
├── notebooks/EDA.ipynb             # Exploratory analysis
├── tests/                          # Unit tests
├── Dockerfile                      # Container deployment
├── requirements.txt                # Python dependencies
└── README.md
```

## Models Trained (7 Total)

| # | Model | Task | Metric |
|---|-------|------|--------|
| 1 | XGBoost + SMOTE | Binary Correctness | AUC = 0.8503 |
| 2 | XGBoost + SMOTE | Multi-class Correctness | F1 = 0.5209 |
| 3 | XGBoost + SMOTE-ENN | Hallucination Detection | AUC = 0.7037 |
| 4 | XGBoost + SMOTE | Faithfulness | F1 = 0.5439 |
| 5 | XGBoost | Latency Regression | R² = 0.8632 |
| 6 | XGBoost | Cost Regression | R² = 0.9909 |
| 7 | TF-IDF + LogReg | Query Task Classification | NLP |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/metrics` | Model evaluation metrics |
| GET | `/models` | List available models |
| POST | `/predict` | **Full prediction** — correctness, hallucination, latency, cost |
| POST | `/classify-query` | Classify query text → task type |

### Example: POST `/predict`
```json
{
  "query": "What causes inflation?",
  "generator_model": "gpt-4o",
  "retrieval_strategy": "hybrid",
  "top1_score": 0.85,
  "recall_at_5": 1.0,
  "temperature": 0.3
}
```

## Dashboard Tabs

1. **📊 EDA Dashboard** — Interactive distributions, correlations, performance breakdowns
2. **🤖 Model Explorer** — Metrics comparison, feature importances
3. **🎯 Config Recommender** — Best configs, query classifier, safety rankings
4. **⚖️ Pareto Optimizer** — Cost/latency/accuracy tradeoff explorer with sliders
5. **🔮 Live Predictor** — Interactive form → real-time predictions with SHAP explanations

## Deployment

### Render (recommended)

**FastAPI:**
```
Build Command: pip install -r requirements.txt && python -m src.models.train
Start Command: uvicorn app.api:app --host 0.0.0.0 --port 10000
```

**Streamlit:**
```
Build Command: pip install -r requirements.txt
Start Command: streamlit run app/streamlit_app.py --server.port 10001 --server.address 0.0.0.0
```

### Docker
```bash
docker build -t rag-optimizer .
docker run -p 8501:8501 rag-optimizer
```

## Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, imbalanced-learn
- **Features**: 80 engineered features (numeric, flags, one-hot, ordinal, interactions, text)
- **Dashboard**: Streamlit, Plotly
- **API**: FastAPI, Uvicorn
- **Containerization**: Docker

## License

MIT
