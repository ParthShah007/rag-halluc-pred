# 🚀 RAG-Optimize: Intelligent RAG Configuration Recommender

> **DataHack 2026** — An end-to-end Machine Learning system that predicts, explains, and optimizes RAG (Retrieval-Augmented Generation) pipeline configurations.

---

## 📋 Table of Contents

- [What Is This?](#what-is-this)
- [Why Does This Exist?](#why-does-this-exist)
- [How It Works](#how-it-works)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [ML Models](#ml-models)
- [Dashboard](#dashboard)
- [REST API](#rest-api)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)

---

## What Is This?

RAG-Optimize is a **prediction + recommendation system** for RAG pipelines. Given any RAG configuration (which LLM, which embedding model, which retrieval strategy, etc.), the system predicts:

| Prediction | What It Answers | Model Type |
|------------|----------------|------------|
| **Correctness** | Will this config produce a correct answer? | Binary Classification (XGBoost) |
| **Hallucination Risk** | Will the LLM hallucinate / make things up? | Binary Classification (XGBoost) |
| **Latency** | How many milliseconds will this take? | Regression (XGBoost) |
| **Cost** | How much will this cost per query? | Regression (XGBoost) |
| **Faithfulness** | Is the answer faithful to the retrieved context? | Multi-class Classification |
| **Task Type** | What type of query is this? (factual, comparison, etc.) | NLP (TF-IDF + LogReg) |

It also provides **SHAP explanations** for every prediction — showing exactly which features pushed the prediction toward correct/incorrect.

---

## Why Does This Exist?

### The Problem

RAG systems have **hundreds of possible configurations**:
- Which LLM to use? (GPT-4o, GPT-4o-mini, Llama, Mixtral...)
- Which embedding model? (text-embedding-3-large, all-MiniLM, BGE...)
- Which retrieval strategy? (hybrid, dense, BM25, HyDE...)
- Which chunking strategy? (fixed_500, semantic, by_heading...)
- What temperature? How many chunks to retrieve? What context window size?

Each combination affects **accuracy, hallucination risk, latency, and cost** differently. Testing every combination manually is impractical — with just 5 options for each of 8 parameters, that's **390,625 possible configs**.

### The Solution

Instead of testing manually, we **train ML models** on 3,824 actual RAG evaluation runs to learn the patterns:
- "GPT-4o with hybrid retrieval and high similarity scores → usually correct"
- "BM25 with low MRR and no relevant chunks in top-5 → high hallucination risk"
- "Large context windows with many chunks → higher latency and cost"

The models learn these patterns from real data, and then predict outcomes for **any new configuration** in milliseconds.

---

## How It Works

### End-to-End Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│                                                                 │
│  eval_runs.csv ──┐                                              │
│  scenarios.csv ──┼──→ load_data.py ──→ Merged DataFrame (3824)  │
│  corpus_chunks ──┤                                              │
│  corpus_docs ────┘                                              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING                          │
│                                                                 │
│  build_features.py                                              │
│  ├─ 16 Numeric features (scores, tokens, latency, cost)        │
│  ├─  6 Flag features (has_relevant, has_answer, etc.)           │
│  ├─ 50 One-Hot encoded features (models, strategies, etc.)      │
│  ├─  1 Ordinal feature (difficulty)                             │
│  ├─  5 Interaction features (retrieval×difficulty, etc.)        │
│  └─  2 Text features (query length, word count)                 │
│  ═══════════════════════════════════════                         │
│  Total: 80 features                                             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MODEL TRAINING                             │
│                                                                 │
│  train.py (7 models)                                            │
│  ├─ SMOTE / SMOTE-ENN for class imbalance                       │
│  ├─ XGBoost, RF, LightGBM, CatBoost (best selected)            │
│  ├─ Optimal threshold tuning via PR curve                       │
│  └─ Saved to models/ as .joblib files                           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT LAYER                            │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────────────────┐       │
│  │   FastAPI (api.py)│    │  Streamlit (streamlit_app.py) │       │
│  │   /predict        │    │  5 interactive tabs           │       │
│  │   /health         │    │  Live predictions + SHAP      │       │
│  │   /classify-query │    │  EDA + Pareto optimizer        │       │
│  └──────────────────┘    └──────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Prediction Flow (what happens when you click "Predict")

```
User selects config in UI
        │
        ▼
transform_single(config)          ← Converts config dict → 80-feature vector
        │                            (one-hot encoding, interactions, etc.)
        ▼
model.predict_proba(X)            ← Each model predicts probabilities
        │
        ▼
threshold comparison              ← Optimized thresholds (not just 0.5)
        │                            Correctness: 0.215, Hallucination: 0.575
        ▼
Return: CORRECT/INCORRECT,
        LOW RISK/HIGH RISK,
        Latency (ms), Cost ($)
```

---

## Data Pipeline

### The 4 Input Datasets

| File | Rows | Description |
|------|------|-------------|
| `eval_runs.csv` | 3,824 | Every RAG evaluation run — the main table. Each row = one query processed through a specific RAG config, with quality metrics |
| `scenarios.csv` | 62 | Query scenarios — the questions being asked, their domain, difficulty, task type |
| `rag_corpus_chunks.csv` | 5,237 | Individual text chunks in the knowledge base |
| `rag_corpus_documents.csv` | 658 | Source documents with metadata (word count, type) |

### How They're Merged

```python
# In load_data.py:
eval_runs + scenarios     → joined on scenario_id (adds domain, difficulty, task_type)
         + documents      → joined on doc_id      (adds word_count, source_type)

# Result: 3,824 rows × 56 columns
```

### Train/Val/Test Split

The data comes **pre-split** via a `split` column:
- **Train**: 2,803 rows (73%) — used for model training
- **Val**: 355 rows (9%) — used for hyperparameter tuning
- **Test**: 666 rows (18%) — used for final evaluation (never seen during training)

---

## Feature Engineering

The `build_features.py` module transforms the 56 raw columns into **80 engineered features**:

### 1. Numeric Features (16)
These are direct numeric measurements from the RAG pipeline:

| Feature | Description | Why It Matters |
|---------|-------------|---------------|
| `top1_score` | Cosine similarity of the best matching chunk | Higher = retrieval found relevant content |
| `mean_retrieved_score` | Average similarity across all retrieved chunks | Quality of overall retrieval |
| `recall_at_5` | Were relevant chunks in top-5 results? | Key retrieval quality metric |
| `recall_at_10` | Were relevant chunks in top-10 results? | Broader retrieval quality |
| `mrr_at_10` | Mean Reciprocal Rank @ 10 | How early relevant chunks appear |
| `answer_tokens` | Number of tokens in the generated answer | Longer answers = higher cost |
| `prompt_tokens` | Number of tokens in the prompt | Affects cost directly |
| `n_retrieved_chunks` | How many chunks were retrieved | More chunks = more context |
| `context_window_tokens` | Max context window of the LLM | Limits how much context can be used |
| `max_new_tokens` | Max tokens the LLM can generate | Controls answer length |
| `latency_ms_retrieval` | Time to retrieve chunks (ms) | Retrieval speed |
| `latency_ms_generation` | Time for LLM generation (ms) | Generation speed |
| `total_latency_ms` | Total end-to-end time | Overall speed |
| `total_cost_usd` | Total cost per query | Direct cost |
| `temperature` | LLM temperature setting | Higher = more creative/risky |
| `top_p` | Nucleus sampling parameter | Controls output diversity |

### 2. Flag Features (6)
Binary (0/1) indicators:

| Feature | What It Means |
|---------|--------------|
| `used_long_context_window` | Was a large context window (>8k tokens) used? |
| `has_relevant_in_top5` | Is at least one gold chunk in the top-5 results? |
| `has_relevant_in_top10` | Is at least one gold chunk in the top-10 results? |
| `is_noanswer_probe` | Is this a "trick" query where the answer isn't in the corpus? |
| `has_answer_in_corpus` | Does the corpus actually contain the answer? |
| `answered_without_retrieval` | Did the LLM answer from its own knowledge? |

### 3. One-Hot Encoded Features (50)
Categorical variables converted to binary columns:

| Original Column | Example Values | # Columns Created |
|-----------------|---------------|-------------------|
| `generator_model` | gpt-4o, gpt-4o-mini, llama-3.1-70B, mixtral... | ~8 |
| `embedding_model` | text-embedding-3-large, all-MiniLM, BGE... | ~6 |
| `retrieval_strategy` | hybrid, dense, bm25, hyde | ~4 |
| `chunking_strategy` | fixed_500, semantic, by_heading, sliding_250 | ~4 |
| `reranker_model` | none, bge-reranker-base, cohere-rerank... | ~4 |
| `domain` | finance, healthcare, legal, technology... | ~8 |
| `task_type` | factual, comparison, multi_hop, summarization... | ~8 |
| `eval_mode` | llm_judge, exact_match, f1_overlap | ~3 |
| `stop_reason` | eos, max_tokens, length | ~3 |

### 4. Ordinal Feature (1)
- `difficulty_ord`: easy=0, medium=1, hard=2

### 5. Interaction Features (5)
Engineered combinations that capture non-linear relationships:

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `retrieval_x_difficulty` | top1_score × difficulty_ord | Good retrieval matters more on hard queries |
| `recall5_x_nchunks` | recall@5 × n_retrieved_chunks | Recall + volume interaction |
| `score_gap` | top1_score − mean_retrieved_score | Are results concentrated or spread? |
| `latency_ratio` | generation_latency / (retrieval_latency + 1) | Where is time being spent? |
| `cost_per_token` | cost / (prompt_tokens + answer_tokens + 1) | Cost efficiency |

### 6. Text Features (2)
- `query_len`: Character length of the query
- `query_word_count`: Word count of the query

---

## ML Models

### Why SMOTE?

The datasets are **heavily imbalanced**:
- Correctness: 74% correct vs 26% incorrect
- Hallucination: 82% safe vs 18% hallucinating

Without SMOTE, models just predict the majority class (always "CORRECT") and get high accuracy but miss all the interesting cases. **SMOTE generates synthetic examples** of the minority class to balance training.

We use **SMOTE-ENN** (SMOTE + Edited Nearest Neighbors) for the most imbalanced tasks — it both oversamples the minority AND removes noisy majority samples near the decision boundary.

### Why Optimized Thresholds?

By default, classifiers use 0.5 as the threshold. But when classes are imbalanced, this is wrong.

We use the **Precision-Recall curve** to find the threshold that maximizes F1-score:
- **Correctness threshold: 0.215** (lower → catches more incorrects)
- **Hallucination threshold: 0.575** (higher → reduces false alarms)

### Model Summaries

#### Model 1: Binary Correctness (is_correct)
- **Task**: Predict if a RAG config will produce a correct answer
- **Algorithm**: XGBoost with SMOTE-ENN
- **Results**: AUC = **0.8503**, F1-macro = **0.8133**, Accuracy = **86.3%**
- **Catches 60% of incorrect answers** while keeping 97% precision on correct ones

#### Model 2: Multi-class Correctness
- **Task**: Classify into correct / incorrect / partial
- **Algorithm**: XGBoost with SMOTE (all classes balanced to equal size)
- **Results**: F1-macro = **0.5209** (hard — "partial" is rare and ambiguous)

#### Model 3: Hallucination Detection
- **Task**: Predict if the LLM will hallucinate
- **Algorithm**: XGBoost with SMOTE-ENN + aggressive class weighting
- **Results**: AUC = **0.7037**, F1-macro = **0.6674**
- **Catches 45% of hallucinations** — hard task because hallucination signals are subtle

#### Model 4: Faithfulness Classification
- **Task**: Classify as faithful / unfaithful / unknown
- **Algorithm**: XGBoost with SMOTE
- **Results**: F1-macro = **0.5439**

#### Model 5: Latency Regression
- **Task**: Predict end-to-end latency in milliseconds
- **Algorithm**: XGBoost Regressor (no SMOTE — regression task)
- **Results**: R² = **0.8632**, RMSE = 180.7ms, MAE = 107.2ms
- **Excludes latency/cost features** to prevent data leakage

#### Model 6: Cost Regression
- **Task**: Predict cost per query in USD
- **Algorithm**: XGBoost Regressor
- **Results**: R² = **0.9909**, RMSE = $0.0001 — nearly perfect
- **Excludes cost features** to prevent leakage

#### Model 7: Query Task Type Classifier
- **Task**: Classify free-text queries into task types (factual, comparison, etc.)
- **Algorithm**: TF-IDF vectorizer + Logistic Regression
- **Purpose**: Auto-detect task type when user types a query

---

## Dashboard

The Streamlit dashboard has **5 interactive tabs**:

### Tab 1: 📊 EDA Dashboard
- **Key metrics**: Total runs, accuracy, hallucination rate, avg latency
- **Correctness distribution**: Pie chart + histogram
- **Performance breakdown**: Group by any config choice (model, strategy, etc.) and see accuracy/hallucination rates
- **Feature correlations**: Heatmap of all 16 numeric features

### Tab 2: 🤖 Model Explorer
- **Classification metrics table**: AUC, F1, accuracy, recall for each model
- **Regression metrics table**: RMSE, MAE, R² for latency and cost
- **Feature importance bar chart**: Top 20 most important features for the correctness model

### Tab 3: 🎯 Config Recommender
- **Top configs by accuracy**: Best-performing combinations of generator + retrieval + embedding
- **Safest configs**: Lowest hallucination rate combinations
- Filters to configs with ≥10 samples for statistical reliability

### Tab 4: ⚖️ Pareto Optimizer
- **Interactive scatter plots**: Accuracy vs Cost, Accuracy vs Latency
- **Sliders**: Set max acceptable cost and latency to find configs within budget
- **Recommendations**: Auto-identifies highest accuracy, cheapest, and fastest configs

### Tab 5: 🔮 Live Predictor
- **Interactive form**: Full RAG config with every parameter (model, scores, tokens, etc.)
- **One-click prediction**: Correctness, hallucination risk, latency, cost
- **SHAP explanation**: Bar chart showing which features pushed the prediction
- **Confidence bars**: Visual probability display

---

## REST API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info and available endpoints |
| `GET` | `/health` | Health check — shows number of loaded models |
| `GET` | `/metrics` | Full model evaluation metrics (AUC, F1, R², etc.) |
| `GET` | `/models` | List of all available models |
| `POST` | `/predict` | **Main endpoint** — full prediction for any RAG config |
| `POST` | `/classify-query` | Classify query text into task type |

### POST `/predict` — Example

**Request:**
```json
{
  "query": "What are the main causes of inflation?",
  "generator_model": "gpt-4o",
  "embedding_model": "text-embedding-3-large",
  "retrieval_strategy": "hybrid",
  "chunking_strategy": "fixed_500",
  "top1_score": 0.85,
  "mean_retrieved_score": 0.55,
  "recall_at_5": 1.0,
  "mrr_at_10": 0.6,
  "temperature": 0.3,
  "n_retrieved_chunks": 5
}
```

**Response:**
```json
{
  "correctness": "CORRECT",
  "correctness_confidence": 0.8734,
  "hallucination_risk": "LOW",
  "hallucination_probability": 0.1823,
  "estimated_latency_ms": 1245.67,
  "estimated_cost_usd": 0.0089,
  "query_task_type": "factual"
}
```

### Interactive API Docs

Visit `/docs` for the auto-generated Swagger UI where you can test all endpoints directly in the browser.

---

## Project Structure

```
rag-optimizer/
│
├── app/                              # Deployment layer
│   ├── api.py                        #   FastAPI REST API (6 endpoints)
│   └── streamlit_app.py              #   Streamlit dashboard (5 tabs)
│
├── src/                              # Core ML pipeline
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py              #   Load 4 CSVs + merge into 1 DataFrame
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py         #   Engineer 80 features from raw data
│   └── models/
│       ├── __init__.py
│       ├── train.py                  #   Train 7 models with SMOTE
│       └── predict.py                #   RAGPredictor class for inference
│
├── models/                           # Saved model artifacts
│   ├── xgb_correctness.joblib        #   Binary correctness classifier
│   ├── xgb_hallucination.joblib      #   Hallucination detector
│   ├── xgb_faithfulness.joblib       #   Faithfulness classifier
│   ├── xgb_multiclass.joblib         #   Multi-class correctness
│   ├── xgb_latency.joblib            #   Latency regressor
│   ├── xgb_cost.joblib               #   Cost regressor
│   ├── tfidf_task_classifier.joblib  #   Query task type classifier
│   ├── threshold_correctness.joblib  #   Optimized threshold (0.215)
│   ├── threshold_hallucination.joblib#   Optimized threshold (0.575)
│   ├── feature_list.json             #   80 feature column names
│   ├── onehot_categories.json        #   Valid categories for encoding
│   └── metrics.json                  #   Evaluation results
│
├── data/
│   └── raw/                          #   4 source CSV files
│       ├── eval_runs.csv
│       ├── scenarios.csv
│       ├── rag_corpus_chunks.csv
│       └── rag_corpus_documents.csv
│
├── tests/                            # Unit tests
│   ├── test_features.py
│   └── test_predict.py
│
├── notebooks/
│   └── EDA.ipynb                     #   Exploratory data analysis
│
├── docs/
│   └── data_dictionary.csv           #   Column definitions
│
├── Dockerfile                        # Docker deployment
├── Makefile                          # Automation commands
├── requirements.txt                  # Python dependencies
├── render.yaml                       # Render Blueprint (auto-deploy)
└── README.md                         # This file
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repo
git clone https://github.com/ParthShah007/rag-halluc-pred.git
cd rag-halluc-pred/rag-optimizer

# Install dependencies
pip install -r requirements.txt
```

### Train Models

```bash
python -m src.models.train
```

This will:
1. Load all 4 CSVs from `data/raw/`
2. Merge and engineer 80 features
3. Train 7 models with SMOTE class balancing
4. Save all artifacts to `models/`
5. Print evaluation metrics

### Run Locally

```bash
# Terminal 1: API server
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Dashboard
streamlit run app/streamlit_app.py
```

Then open:
- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

---

## Deployment

### Render (Cloud — Free Tier)

The repo includes a `render.yaml` Blueprint. To deploy:

1. Go to [dashboard.render.com/blueprints](https://dashboard.render.com/blueprints)
2. Click "New Blueprint Instance"
3. Connect GitHub → select `ParthShah007/rag-halluc-pred`
4. Render auto-detects the config and creates both services
5. Click "Apply"

**Any future `git push` auto-redeploys.**

### Docker

```bash
docker build -t rag-optimizer .
docker run -p 8501:8501 -p 8000:8000 rag-optimizer
```

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML Models** | XGBoost, LightGBM, CatBoost, Random Forest, scikit-learn |
| **Class Balancing** | SMOTE, SMOTE-ENN (imbalanced-learn) |
| **Explainability** | SHAP (TreeExplainer) |
| **Feature Engineering** | pandas, NumPy |
| **NLP** | TF-IDF (scikit-learn) |
| **Dashboard** | Streamlit, Plotly |
| **API** | FastAPI, Uvicorn |
| **Serialization** | joblib |
| **Containerization** | Docker |
| **Deployment** | Render |

---

## License

MIT

---

*Built for DataHack 2026*
