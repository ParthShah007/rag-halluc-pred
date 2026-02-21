"""
Feature engineering module.

Defines the full feature engineering pipeline:
- Numeric, flag, nominal, ordinal features
- One-Hot encoding
- Ordinal encoding (difficulty)
- Interaction features
- Text-derived features
"""

import json
import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


# ══════════════════════════════════════════════════════════
# Feature column definitions
# ══════════════════════════════════════════════════════════

NUMERIC_FEATURES = [
    "top1_score", "mean_retrieved_score", "recall_at_5", "recall_at_10",
    "mrr_at_10", "answer_tokens", "prompt_tokens", "n_retrieved_chunks",
    "context_window_tokens", "max_new_tokens", "latency_ms_retrieval",
    "latency_ms_generation", "total_latency_ms", "total_cost_usd",
    "temperature", "top_p",
]

FLAG_FEATURES = [
    "used_long_context_window", "has_relevant_in_top5", "has_relevant_in_top10",
    "is_noanswer_probe", "has_answer_in_corpus", "answered_without_retrieval",
]

NOMINAL_FEATURES = [
    "domain", "task_type", "retrieval_strategy", "chunking_strategy",
    "generator_model", "embedding_model", "reranker_model", "eval_mode",
    "stop_reason",
]

ORDINAL_FEATURES = ["difficulty"]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]

TARGETS = {
    "binary_correctness": "is_correct",
    "multiclass_correctness": "correctness_label",
    "hallucination": "hallucination_flag",
    "faithfulness": "faithfulness_label",
    "latency": "total_latency_ms",
    "cost": "total_cost_usd",
    "task_type": "task_type",
}


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply full feature engineering pipeline.

    Steps:
        1. One-Hot encode nominal features
        2. Ordinal encode difficulty
        3. Create interaction features
        4. Create text-derived features
        5. Assemble final feature list

    Args:
        df: Merged DataFrame from load_data.

    Returns:
        Tuple of (engineered DataFrame, list of ALL feature column names).
    """
    df = df.copy()

    # ── 1. One-Hot Encoding (nominal) ──
    onehot_cols = []
    for col in NOMINAL_FEATURES:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            dummies = dummies.astype(int)
            df = pd.concat([df, dummies], axis=1)
            onehot_cols.extend(dummies.columns.tolist())

    # ── 2. Ordinal Encoding (difficulty) ──
    if "difficulty" in df.columns:
        diff_map = {v: i for i, v in enumerate(DIFFICULTY_ORDER)}
        df["difficulty_ord"] = df["difficulty"].map(diff_map).fillna(1).astype(int)
    ordinal_cols = ["difficulty_ord"]

    # ── 3. Interaction Features ──
    df["retrieval_x_difficulty"] = df["top1_score"] * df.get("difficulty_ord", 1)
    df["recall5_x_nchunks"] = df["recall_at_5"] * df["n_retrieved_chunks"]
    df["score_gap"] = df["top1_score"] - df["mean_retrieved_score"]
    df["latency_ratio"] = df["latency_ms_generation"] / (df["latency_ms_retrieval"] + 1)
    df["cost_per_token"] = df["total_cost_usd"] / (df["prompt_tokens"] + df["answer_tokens"] + 1)
    interaction_cols = [
        "retrieval_x_difficulty", "recall5_x_nchunks",
        "score_gap", "latency_ratio", "cost_per_token",
    ]

    # ── 4. Text Features ──
    if "query" in df.columns:
        df["query_len"] = df["query"].str.len()
        df["query_word_count"] = df["query"].str.split().str.len()
    text_cols = ["query_len", "query_word_count"]

    # ── 5. Assemble Feature List ──
    # Only include columns that actually exist in df
    all_features = []
    for col in NUMERIC_FEATURES + FLAG_FEATURES + ordinal_cols + onehot_cols + interaction_cols + text_cols:
        if col in df.columns:
            all_features.append(col)

    print(f"  Total features: {len(all_features)}")
    print(f"    Numeric:      {len([c for c in NUMERIC_FEATURES if c in df.columns])}")
    print(f"    Flags:        {len([c for c in FLAG_FEATURES if c in df.columns])}")
    print(f"    One-Hot:      {len(onehot_cols)}")
    print(f"    Ordinal:      {len(ordinal_cols)}")
    print(f"    Interactions: {len(interaction_cols)}")
    print(f"    Text:         {len(text_cols)}")

    return df, all_features


def split_data(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Split data using the built-in 'split' column.

    Args:
        df: Feature-engineered DataFrame.
        feature_cols: List of feature column names.

    Returns:
        Dictionary with train/val/test DataFrames and feature matrices.
    """
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    X_train = train[feature_cols].fillna(0)
    X_val = val[feature_cols].fillna(0)
    X_test = test[feature_cols].fillna(0)

    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"  Feature matrix: {X_train.shape[1]} columns")

    return {
        "train": train, "val": val, "test": test,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
    }


def save_feature_list(feature_cols: list[str]) -> str:
    """Save feature list to models/feature_list.json."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "feature_list.json")
    with open(path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Saved feature list: {path}")
    return path


def load_feature_list() -> list[str]:
    """Load feature list from models/feature_list.json."""
    path = os.path.join(MODELS_DIR, "feature_list.json")
    with open(path) as f:
        return json.load(f)


def save_onehot_categories(df: pd.DataFrame) -> str:
    """Save unique categories for each nominal feature."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    cats = {}
    for col in NOMINAL_FEATURES:
        if col in df.columns:
            cats[col] = sorted(df[col].dropna().unique().tolist())
    path = os.path.join(MODELS_DIR, "onehot_categories.json")
    with open(path, "w") as f:
        json.dump(cats, f, indent=2)
    return path


def load_onehot_categories() -> dict:
    """Load saved one-hot categories."""
    path = os.path.join(MODELS_DIR, "onehot_categories.json")
    with open(path) as f:
        return json.load(f)


def transform_single(config: dict, feature_list: list[str] = None) -> pd.DataFrame:
    """Transform a single config dict into a feature vector for prediction.

    Args:
        config: Dictionary with all RAG pipeline config values.
        feature_list: Saved feature list from training. If None, loads from disk.

    Returns:
        Single-row DataFrame with all feature columns, ready for model.predict().
    """
    if feature_list is None:
        feature_list = load_feature_list()

    categories = load_onehot_categories()

    row = {}

    # Numeric features
    for col in NUMERIC_FEATURES:
        row[col] = config.get(col, 0.0)

    # Flag features
    for col in FLAG_FEATURES:
        row[col] = config.get(col, 0)

    # Ordinal (difficulty)
    diff_map = {v: i for i, v in enumerate(DIFFICULTY_ORDER)}
    row["difficulty_ord"] = diff_map.get(config.get("difficulty", "medium"), 1)

    # One-Hot encoding
    for col, values in categories.items():
        selected = config.get(col, "")
        for v in values:
            col_name = f"{col}_{v}"
            row[col_name] = 1 if str(selected) == str(v) else 0

    # Interaction features
    row["retrieval_x_difficulty"] = row.get("top1_score", 0) * row.get("difficulty_ord", 1)
    row["recall5_x_nchunks"] = row.get("recall_at_5", 0) * row.get("n_retrieved_chunks", 0)
    row["score_gap"] = row.get("top1_score", 0) - row.get("mean_retrieved_score", 0)
    row["latency_ratio"] = row.get("latency_ms_generation", 0) / (row.get("latency_ms_retrieval", 0) + 1)
    row["cost_per_token"] = row.get("total_cost_usd", 0) / (row.get("prompt_tokens", 0) + row.get("answer_tokens", 0) + 1)

    # Text features
    query = config.get("query", "")
    row["query_len"] = len(query) if query else 0
    row["query_word_count"] = len(query.split()) if query else 0

    # Build DataFrame with exactly the training columns
    df = pd.DataFrame([row])
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    return df[feature_list].fillna(0)
