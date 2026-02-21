# ML Pipeline — Complete Technical Deep Dive

Everything that happens behind the scenes, from raw CSV to prediction.

---

## STAGE 1: Data Loading (`src/data/load_data.py`)

### What runs

```python
python -m src.models.train
# internally calls: load_and_merge() → 4 CSVs → 1 merged DataFrame
```

### The 4 raw datasets

| File | Rows | What it contains |
|------|------|-----------------|
| `eval_runs.csv` | 3,824 | **The main table.** Every row = one RAG query run. Contains: config used (model, embedding, retrieval strategy), quality metrics (correct/incorrect, hallucination flag), retrieval scores, token counts, latency, cost |
| `scenarios.csv` | 62 | The query scenarios — domain (finance/legal/health), difficulty (easy/medium/hard), task type (factual/comparison/multi_hop) |
| `rag_corpus_chunks.csv` | 5,237 | Individual text chunks in the knowledge base |
| `rag_corpus_documents.csv` | 658 | Source documents — word count, type, number of chunks |

### How merging works

```
eval_runs (3824 rows)
    │
    ├── LEFT JOIN scenarios ON scenario_id
    │   → adds: domain, difficulty, task_type, gold_chunk_ids
    │
    └── LEFT JOIN documents ON doc_id (extracted from doc_ids_used)
        → adds: source_type, word_count, num_chunks

Result: 3,824 rows × 56 columns
```

**Why LEFT JOIN?** — We don't want to lose eval_runs rows if a scenario or document is missing. LEFT JOIN keeps all eval_runs and fills NaN for unmatched rows.

**doc_id extraction trick** — The `doc_ids_used` column stores a list-like string (`"['doc_0023', 'doc_0041']"`). The code extracts just the first doc_id:
```python
merged["primary_doc_id"] = merged["doc_ids_used"].str.strip("[]'\"").str.split(",").str[0].str.strip()
```

---

## STAGE 2: Feature Engineering (`src/features/build_features.py`)

This is where the 56 raw columns become **80 ML-ready features**.

### Step 2.1 — Numeric Features (16 columns, kept as-is)

These are direct measurements from each RAG run:

```
top1_score              → Best chunk's cosine similarity (0.0-1.0)
mean_retrieved_score    → Average cosine similarity across all retrieved chunks
recall_at_5             → Were gold chunks in top-5 results? (0.0-1.0)
recall_at_10            → Were gold chunks in top-10 results? (0.0-1.0)
mrr_at_10               → Mean Reciprocal Rank — how early gold chunks appear
answer_tokens           → Tokens in the generated answer
prompt_tokens           → Tokens in the prompt sent to the LLM
n_retrieved_chunks      → How many chunks were pulled from the vector DB
context_window_tokens   → Max context window of the LLM (4096, 8192, 128000, etc.)
max_new_tokens          → Max tokens the LLM is allowed to generate
latency_ms_retrieval    → Milliseconds to retrieve chunks
latency_ms_generation   → Milliseconds for LLM to generate answer
total_latency_ms        → End-to-end latency (retrieval + generation)
total_cost_usd          → Cost of this single query in USD
temperature             → LLM temperature (0.0 = deterministic, 1.0 = creative)
top_p                   → Nucleus sampling parameter
```

**No transformation** — these go directly into the model. Missing values → filled with 0.

### Step 2.2 — Flag Features (6 binary columns)

Binary indicators (0 or 1):

```
used_long_context_window    → Was context window > 8192 tokens?
has_relevant_in_top5        → Is at least 1 gold chunk in the top-5 retrieved?
has_relevant_in_top10       → Is at least 1 gold chunk in the top-10 retrieved?
is_noanswer_probe           → Trick question where answer isn't in the corpus
has_answer_in_corpus        → Does the knowledge base actually contain the answer?
answered_without_retrieval  → Did the LLM answer from its own knowledge?
```

### Step 2.3 — One-Hot Encoding (9 categorical → 50 binary columns)

**Why one-hot?** Tree-based models (XGBoost, Random Forest) don't naturally handle text categories like "gpt-4o". One-hot converts each category into a separate binary column.

**How it works:**
```
generator_model = "gpt-4o"

→ generator_model_gpt-4o         = 1
  generator_model_gpt-4o-mini    = 0
  generator_model_llama-3.1-70B  = 0
  generator_model_mixtral-8x22B  = 0
  ...
```

**All 9 categorical columns encoded:**
```
generator_model      → ~8 columns   (gpt-4o, gpt-4o-mini, llama, mixtral, etc.)
embedding_model      → ~6 columns   (text-embedding-3-large, all-MiniLM, BGE, etc.)
retrieval_strategy   → ~4 columns   (hybrid, dense, bm25, hyde)
chunking_strategy    → ~4 columns   (fixed_500, semantic, by_heading, sliding_250)
reranker_model       → ~4 columns   (none, bge-reranker, cohere-rerank, etc.)
domain               → ~8 columns   (finance, healthcare, legal, technology, etc.)
task_type            → ~8 columns   (factual, comparison, multi_hop, etc.)
eval_mode            → ~3 columns   (llm_judge, exact_match, f1_overlap)
stop_reason          → ~3 columns   (eos, max_tokens, length)
```

**Note:** `drop_first=False` — we keep all dummies. For tree-based models this is fine (no multicollinearity issue like in linear regression).

**Categories are saved** to `models/onehot_categories.json` so that at prediction time, we know exactly which columns to create.

### Step 2.4 — Ordinal Encoding (1 column)

```python
difficulty_ord = {"easy": 0, "medium": 1, "hard": 2}
```

**Why ordinal and not one-hot?** Because difficulty has a natural order — hard > medium > easy. Ordinal encoding preserves this ranking. One-hot would lose the ordering information.

### Step 2.5 — Interaction Features (5 engineered columns)

These capture **non-linear relationships** between features:

```python
retrieval_x_difficulty = top1_score × difficulty_ord
```
→ **Why?** A top1_score of 0.8 on an easy question means something different than 0.8 on a hard question. This feature captures that interaction.

```python
recall5_x_nchunks = recall_at_5 × n_retrieved_chunks
```
→ **Why?** High recall with 5 chunks is more impressive than high recall with 50 chunks. This captures retrieval efficiency.

```python
score_gap = top1_score − mean_retrieved_score
```
→ **Why?** If the best chunk scores 0.9 but the average is 0.3, there's one very relevant chunk and lots of noise. If both are 0.7, the retrieved set is consistently good. This difference matters.

```python
latency_ratio = latency_ms_generation / (latency_ms_retrieval + 1)
```
→ **Why?** Shows where time is spent. A ratio of 10 means generation dominates. A ratio of 0.5 means retrieval is the bottleneck. The `+1` prevents division by zero.

```python
cost_per_token = total_cost_usd / (prompt_tokens + answer_tokens + 1)
```
→ **Why?** Normalizes cost by output size. GPT-4o at $0.01 for 100 tokens is very different from $0.01 for 10 tokens.

### Step 2.6 — Text Features (2 columns)

```python
query_len        = len(query)           # Character count
query_word_count = len(query.split())   # Word count
```

→ **Why?** Longer queries tend to be harder (multi-hop, explanations). Short queries tend to be factual.

### Final Feature Vector

```
16 numeric + 6 flags + 50 one-hot + 1 ordinal + 5 interactions + 2 text = 80 features
```

All saved to `models/feature_list.json` — an ordered list of exactly which 80 column names the model expects.

---

## STAGE 3: Train/Val/Test Split (`build_features.split_data()`)

The data comes **pre-split** via a `split` column in eval_runs.csv:

```
Train: 2,803 rows (73%)  → used to fit models
Val:     355 rows  (9%)  → used for early stopping / hyperparameter tuning
Test:    666 rows (18%)  → NEVER seen during training, used only for final evaluation
```

**Why pre-split?** To prevent data leakage. If we randomly split, similar queries from the same scenario could end up in both train and test, inflating metrics.

```python
X_train = train[feature_cols].fillna(0)   # 2803 × 80 matrix
X_val   = val[feature_cols].fillna(0)     #  355 × 80 matrix
X_test  = test[feature_cols].fillna(0)    #  666 × 80 matrix
```

---

## STAGE 4: Model Training (`src/models/train.py`)

### The Class Imbalance Problem

Before training, look at the class distributions:

```
Correctness:    74% correct  vs  26% incorrect     → 2.8:1 ratio
Hallucination:  82% safe     vs  18% hallucinating  → 4.5:1 ratio
```

**Problem:** If you train a model on this data, it learns to just predict "CORRECT" for everything and gets 74% accuracy. But it catches 0% of incorrect answers — completely useless.

### Solution: SMOTE + SMOTE-ENN

**SMOTE** (Synthetic Minority Oversampling Technique):
1. Pick a minority class sample
2. Find its 5 nearest neighbors (in feature space)
3. Create a synthetic sample by interpolating between them
4. Repeat until classes are balanced

```
Before SMOTE:  {correct: 2095, incorrect: 708}
After SMOTE:   {correct: 1272, incorrect: 648}   ← Classes roughly balanced
```

**SMOTE-ENN** (SMOTE + Edited Nearest Neighbors) — used for the hardest tasks:
1. First applies SMOTE (oversample minority)
2. Then applies ENN: for each sample, checks if its 3 nearest neighbors agree on the class. If not, removes it
3. This **cleans the boundary** — removes noisy samples near the decision boundary

**Why SMOTE-ENN for binary correctness and hallucination?** These are the models where decision boundary clarity matters most. SMOTE-ENN produces cleaner training data.

**Why plain SMOTE for multi-class?** With 3+ classes, ENN can remove too many samples. Plain SMOTE is safer.

### Solution: Optimized Thresholds

By default, classifiers use **threshold = 0.5**: if P(correct) ≥ 0.5, predict "correct".

But when classes are imbalanced, 0.5 is wrong. The model might output P(correct) = 0.6 for most samples (because 74% ARE correct), so a 0.5 threshold catches nothing.

**How we find the optimal threshold:**
```python
precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[argmax(f1_scores)]
```

This walks through every possible threshold (0.01, 0.02, ..., 0.99), computes precision and recall at each, and picks the one that maximizes F1 (the harmonic mean of precision and recall).

**Result:**
```
Correctness threshold:   0.215  (much lower than 0.5 → catches more incorrects)
Hallucination threshold: 0.575  (higher → reduces false alarms)
```

---

### MODEL 1: Binary Correctness

**Target:** `is_correct` (0 = incorrect, 1 = correct)

**What it predicts:** Given a RAG configuration, will the answer be correct?

**Training:**
```
1. SMOTE-ENN on training data (2803 rows → balanced)
2. Train 3 models: XGBoost, RandomForest, LightGBM
3. Each trained on SMOTE-resampled data
4. XGBoost: early stopping using validation set
5. Find best threshold per model using PR curve on test set
6. Select best model by F1-macro
7. Save winner + its threshold
```

**XGBoost hyperparameters:**
```python
n_estimators=300     # 300 boosting rounds
max_depth=6          # Each tree can be 6 levels deep
learning_rate=0.1    # Step size for gradient descent
subsample=0.8        # Use 80% of samples per tree (prevents overfitting)
colsample_bytree=0.8 # Use 80% of features per tree (prevents overfitting)
```

**Winner: XGBoost** (F1-macro = 0.8133, AUC = 0.8503)
```
              precision  recall  f1
incorrect(0)    0.89     0.60   0.72    ← catches 60% of incorrect answers
correct(1)      0.86     0.97   0.91    ← barely misses any correct ones
```
Saved as: `models/xgb_correctness.joblib` + `models/threshold_correctness.joblib`

---

### MODEL 2: Multi-class Correctness

**Target:** `correctness_label` (correct / incorrect / partial)

**Why a separate model?** Binary just says correct/incorrect. This one adds "partial" — the answer was partially right but incomplete or imprecise.

**Training:**
```
1. LabelEncoder: correct→0, incorrect→1, partial→2
2. SMOTE (not SMOTE-ENN — 3 classes, ENN too aggressive)
3. Train: XGBoost (multi:softprob) and CatBoost (MultiClass, auto_class_weights=Balanced)
4. Save XGBoost (slightly better) + label encoder
```

**Why CatBoost too?** CatBoost has built-in balanced class weights and handles categorical features natively. We compare both and pick the better one.

**Result: XGBoost** (F1-macro = 0.5209)
```
correct:     recall 0.985  ← catches nearly all correct answers
incorrect:   recall 0.519  ← catches half of incorrect
partial:     recall 0.049  ← struggles with partial (rare + ambiguous)
```

**Why is partial so bad?** Only 61 test samples (9%), and "partial" is inherently ambiguous — it's hard even for humans to define.

Saved as: `models/xgb_multiclass.joblib` + `models/le_correctness.joblib`

---

### MODEL 3: Hallucination Detection

**Target:** `hallucination_flag` (0 = safe, 1 = hallucinating)

**This is the hardest model.** Hallucination is subtle — the model needs to detect when the LLM confidently states something that isn't in the retrieved context.

**Training:**
```
1. SMOTE-ENN (most imbalanced: 82% vs 18%)
2. Compute scale_pos_weight = count(class_0) / count(class_1) ≈ 4.5
3. XGBoost: uses scale_pos_weight (penalizes misclassifying hallucinations 4.5× more)
4. LightGBM: uses is_unbalance=True (built-in weighting)
5. Find best threshold per model
6. Save best by F1-macro
```

**Why scale_pos_weight?** On top of SMOTE, this tells XGBoost to penalize false negatives (missed hallucinations) 4.5× more than false positives. Double defense.

**Result: XGBoost** (F1-macro = 0.6668, AUC = 0.7037)
```
safe(0):          recall 0.87  ← correctly identifies most safe outputs
hallucinating(1): recall 0.45  ← catches 45% of hallucinations
```

**Why only 45%?** Hallucination detection is an open research problem. The features available (retrieval scores, config choices) are indirect signals. The actual content of the generated text (which might reveal hallucination) isn't a feature here.

Saved as: `models/xgb_hallucination.joblib` + `models/threshold_hallucination.joblib`

---

### MODEL 4: Faithfulness Classification

**Target:** `faithfulness_label` (faithful / unfaithful / unknown)

**Similar to hallucination but three classes:**
- **Faithful** = answer is grounded in retrieved context
- **Unfaithful** = answer deviates from context
- **Unknown** = can't determine

**Training:**
```
1. LabelEncoder
2. SMOTE (3 classes)
3. XGBoost (multi:softprob) + RandomForest (class_weight=balanced)
4. Save XGBoost
```

**Result: XGBoost** (F1-macro = 0.5439)

Saved as: `models/xgb_faithfulness.joblib` + `models/le_faithfulness.joblib`

---

### MODEL 5: Latency Regression

**Target:** `total_latency_ms` (continuous, in milliseconds)

**Critical: Data Leakage Prevention**

The feature set includes `total_latency_ms`, `latency_ms_retrieval`, `latency_ms_generation` — we're predicting latency, so we **cannot use latency as a feature**. Same for cost-derived features.

```python
leakage_cols = ["total_latency_ms", "latency_ms_retrieval",
                "latency_ms_generation", "total_cost_usd",
                "latency_ratio", "cost_per_token"]

lat_features = [f for f in features if f not in leakage_cols]
# 80 features → 74 features (6 removed)
```

**Training:**
```
1. Remove leakage columns
2. XGBoost Regressor (300 trees, depth 6)
3. RandomForest Regressor (200 trees, depth 12)
4. Save XGBoost + the leakage-safe feature list
```

**Result: XGBoost** (R² = 0.8632, RMSE = 180.7ms, MAE = 107.2ms)

**What R² = 0.86 means:** The model explains 86% of the variance in latency. Given just the config (which model, how many chunks, etc.), it can predict latency to within ~107ms on average.

Saved as: `models/xgb_latency.joblib` + `models/latency_features.joblib`

---

### MODEL 6: Cost Regression

**Target:** `total_cost_usd` (continuous, in USD)

**Same leakage prevention:**
```python
leakage_cols = ["total_cost_usd", "cost_per_token"]
cost_features = [f for f in features if f not in leakage_cols]
# 80 features → 78 features (2 removed)
```

**Result: XGBoost** (R² = 0.9909, RMSE = $0.0001)

**Why so good?** Cost is almost entirely determined by `prompt_tokens` + `answer_tokens` + which model is used. These are directly available features, so the model learns the pricing function almost perfectly.

Saved as: `models/xgb_cost.joblib` + `models/cost_features.joblib`

---

### MODEL 7: Query Task Type Classifier

**Target:** `task_type` (factual, comparison, multi_hop, summarization, explanation, instruction_following, table_qa, temporal_reasoning)

**Completely different from models 1-6.** This one takes raw text (the query string) and classifies it.

**Pipeline:**
```
query text → TF-IDF vectorizer → classifier → task type
```

**TF-IDF (Term Frequency-Inverse Document Frequency):**
```python
TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
```
- `max_features=3000` → Keep top 3000 most informative words/bigrams
- `ngram_range=(1,2)` → Use single words AND two-word phrases ("how many", "compare these")
- `sublinear_tf=True` → Use log(tf) instead of raw counts (reduces impact of very common words)

**Two classifiers compared:**
```python
LogisticRegression(max_iter=1000, class_weight="balanced")
LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")
```

**Result:** Both are weak (~6% F1-macro) because the query text alone doesn't strongly predict task type — many queries could be "factual" or "explanation" depending on context.

Saved as: `models/tfidf_task_classifier.joblib` + `models/le_task_type.joblib`

---

## STAGE 5: Model Artifacts

After training, these files are saved to `models/`:

| File | What it is | Size |
|------|-----------|------|
| `xgb_correctness.joblib` | Binary correctness classifier | 515 KB |
| `xgb_multiclass.joblib` | Multi-class correctness | 2.6 MB |
| `xgb_hallucination.joblib` | Hallucination detector | 872 KB |
| `xgb_faithfulness.joblib` | Faithfulness classifier | 2.9 MB |
| `xgb_latency.joblib` | Latency regressor | 1.2 MB |
| `xgb_cost.joblib` | Cost regressor | 243 KB |
| `tfidf_task_classifier.joblib` | TF-IDF + SVM pipeline | 27 KB |
| `threshold_correctness.joblib` | Threshold value (0.215) | 113 B |
| `threshold_hallucination.joblib` | Threshold value (0.575) | 117 B |
| `le_correctness.joblib` | LabelEncoder for 3 classes | 563 B |
| `le_faithfulness.joblib` | LabelEncoder for 3 classes | 565 B |
| `le_task_type.joblib` | LabelEncoder for 8 task types | 672 B |
| `feature_list.json` | Ordered list of 80 feature names | 2.3 KB |
| `onehot_categories.json` | Valid values for each categorical | 1.3 KB |
| `latency_features.joblib` | 74 feature names (leakage-safe) | 1.9 KB |
| `cost_features.joblib` | 78 feature names (leakage-safe) | 2.0 KB |
| `metrics.json` | AUC, F1, R², RMSE for each model | 759 B |

---

## STAGE 6: Prediction at Runtime (`src/models/predict.py`)

### Loading

`RAGPredictor.__init__()` loads all 7 models, 3 label encoders, 2 regression feature lists, and the feature list from disk. One-time load, stays in memory.

### Prediction Flow

When a user submits a config (from the dashboard or API), here's what happens:

```
User input: {
  "generator_model": "gpt-4o",
  "retrieval_strategy": "hybrid",
  "top1_score": 0.85,
  "temperature": 0.3,
  ...
}
```

**Step 1: `transform_single(config)`** — Convert dict to 80-feature vector

```python
# Numeric features → direct copy
row["top1_score"] = 0.85
row["temperature"] = 0.3

# Flag features → direct copy (default 0)
row["has_relevant_in_top5"] = config.get("has_relevant_in_top5", 0)

# One-hot encoding → creates 50 binary columns
# Loads onehot_categories.json to know valid values
row["generator_model_gpt-4o"] = 1      # ← selected
row["generator_model_gpt-4o-mini"] = 0  # ← not selected

# Ordinal → map difficulty string to int
row["difficulty_ord"] = {"easy": 0, "medium": 1, "hard": 2}[difficulty]

# Interactions → computed from other features
row["retrieval_x_difficulty"] = top1_score × difficulty_ord
row["score_gap"] = top1_score − mean_retrieved_score

# Text features
row["query_len"] = len(query)
row["query_word_count"] = len(query.split())

# Ensure exactly 80 columns in the right order
df = pd.DataFrame([row])[feature_list].fillna(0)
```

**Step 2: Run each model**

```python
# Correctness
corr_proba = xgb_correctness.predict_proba(X)[:, 1][0]  # P(correct)
corr_label = "CORRECT" if corr_proba >= 0.215 else "INCORRECT"

# Hallucination
hal_proba = xgb_hallucination.predict_proba(X)[:, 1][0]  # P(hallucination)
hal_label = "LOW RISK" if hal_proba < 0.575 else "HIGH RISK"

# Latency (uses 74 features, not 80 — leakage columns removed)
latency_ms = xgb_latency.predict(X[latency_features])[0]

# Cost (uses 78 features)
cost_usd = xgb_cost.predict(X[cost_features])[0]
```

**Step 3: Return results**

```python
{
  "correctness": "CORRECT",
  "correctness_probability": 0.87,
  "hallucination_risk": "LOW RISK",
  "hallucination_probability": 0.12,
  "estimated_latency_ms": 1245.67,
  "estimated_cost_usd": 0.0089
}
```

---

## Summary: The Complete Data Flow

```
eval_runs.csv (3824 rows)
scenarios.csv (62 rows)         ──→  load_data.py  ──→  Merged DataFrame (3824 × 56)
corpus_chunks.csv (5237 rows)                               │
corpus_docs.csv (658 rows)                                  │
                                                            ▼
                                                    build_features.py
                                                    ┌─────────────────┐
                                                    │ 16 numeric      │
                                                    │  6 flags        │
                                                    │ 50 one-hot      │
                                                    │  1 ordinal      │
                                                    │  5 interactions  │
                                                    │  2 text         │
                                                    └───────┬─────────┘
                                                            │ 80 features
                                                            ▼
                                                       split_data()
                                                    ┌─────────────────┐
                                                    │ Train: 2803     │
                                                    │ Val:    355     │
                                                    │ Test:   666     │
                                                    └───────┬─────────┘
                                                            │
                                                            ▼
                                                        train.py
                                              ┌─────────────────────────┐
                                              │ 1. SMOTE/SMOTE-ENN      │
                                              │ 2. XGBoost/RF/LGB/Cat   │
                                              │ 3. Threshold tuning     │
                                              │ 4. Best model selection  │
                                              └───────────┬─────────────┘
                                                          │
                                                          ▼
                                                  17 model artifacts
                                                   (models/*.joblib)
                                                          │
                                                          ▼
                                                      predict.py
                                              ┌─────────────────────────┐
                                              │ RAGPredictor class      │
                                              │ - loads all 7 models    │
                                              │ - transform_single()    │
                                              │ - predict_all()         │
                                              └─────────────────────────┘
```
