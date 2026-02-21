"""
Unit tests for the feature engineering pipeline.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def _make_sample_df(n: int = 100) -> pd.DataFrame:
    """Create a small sample DataFrame for testing."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "run_id": range(n),
        "scenario_id": [f"s_{i % 10}" for i in range(n)],
        "query": [f"What is topic {i}?" for i in range(n)],
        "gold_answer": [f"Answer {i}" for i in range(n)],
        "generated_answer": [f"Generated {i}" for i in range(n)],
        "domain": rng.choice(["finance", "medical", "legal"], n),
        "task_type": rng.choice(["factual", "analytical", "comparison"], n),
        "difficulty": rng.choice(["easy", "medium", "hard"], n),
        "retrieval_strategy": rng.choice(["bm25", "dense", "hybrid"], n),
        "chunking_strategy": rng.choice(["fixed_500", "semantic"], n),
        "generator_model": rng.choice(["gpt-4o", "claude-sonnet"], n),
        "embedding_model": rng.choice(["text-embedding-3-large", "e5-large"], n),
        "reranker_model": rng.choice(["none", "cohere-rerank"], n),
        "eval_mode": rng.choice(["llm_judge", "exact_match"], n),
        "stop_reason": rng.choice(["eos", "max_tokens"], n),
        "top1_score": rng.uniform(0, 1, n),
        "mean_retrieved_score": rng.uniform(0, 1, n),
        "recall_at_5": rng.choice([0, 1], n).astype(float),
        "recall_at_10": rng.choice([0, 1], n).astype(float),
        "mrr_at_10": rng.uniform(0, 1, n),
        "answer_tokens": rng.randint(10, 500, n),
        "prompt_tokens": rng.randint(100, 2000, n),
        "n_retrieved_chunks": rng.randint(1, 20, n),
        "context_window_tokens": rng.choice([4096, 8192, 16384], n),
        "max_new_tokens": rng.choice([256, 512, 1024], n),
        "latency_ms_retrieval": rng.uniform(50, 500, n),
        "latency_ms_generation": rng.uniform(200, 3000, n),
        "total_latency_ms": rng.uniform(300, 3500, n),
        "total_cost_usd": rng.uniform(0.001, 0.05, n),
        "temperature": rng.choice([0.0, 0.3, 0.7, 1.0], n),
        "top_p": rng.uniform(0.8, 1.0, n),
        "used_long_context_window": rng.choice([0, 1], n),
        "has_relevant_in_top5": rng.choice([0, 1], n),
        "has_relevant_in_top10": rng.choice([0, 1], n),
        "is_noanswer_probe": rng.choice([0, 1], n, p=[0.9, 0.1]),
        "has_answer_in_corpus": rng.choice([0, 1], n),
        "answered_without_retrieval": rng.choice([0, 1], n, p=[0.95, 0.05]),
        "is_correct": rng.choice([0, 1], n),
        "correctness_label": rng.choice(["correct", "partial", "incorrect"], n),
        "hallucination_flag": rng.choice([0, 1], n, p=[0.7, 0.3]),
        "faithfulness_label": rng.choice(["faithful", "unfaithful", "unknown"], n),
        "split": np.array(["train"] * 60 + ["val"] * 20 + ["test"] * 20),
    })


class TestBuildFeatures:
    """Tests for src.features.build_features."""

    def test_engineer_features_returns_tuple(self):
        from src.features.build_features import engineer_features
        df = _make_sample_df()
        result = engineer_features(df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_feature_count(self):
        from src.features.build_features import engineer_features
        df = _make_sample_df()
        df_feat, features = engineer_features(df)
        # Should have numeric + flags + one-hot + ordinal + interaction + text
        assert len(features) > 30, f"Expected >30 features, got {len(features)}"

    def test_no_label_encoder(self):
        """Verify One-Hot encoding is used, not LabelEncoder."""
        from src.features.build_features import engineer_features
        df = _make_sample_df()
        df_feat, features = engineer_features(df)
        # Should have one-hot columns like domain_finance, not domain_enc
        assert any("domain_" in f for f in features), "Missing one-hot columns for domain"
        assert not any(f.endswith("_enc") for f in features), "Found _enc columns (LabelEncoder)"

    def test_difficulty_ordinal(self):
        """Verify difficulty is ordinally encoded."""
        from src.features.build_features import engineer_features
        df = _make_sample_df()
        df_feat, features = engineer_features(df)
        assert "difficulty_ord" in features
        assert df_feat["difficulty_ord"].min() >= 0
        assert df_feat["difficulty_ord"].max() <= 2

    def test_interaction_features(self):
        from src.features.build_features import engineer_features
        df = _make_sample_df()
        df_feat, features = engineer_features(df)
        for f in ["retrieval_x_difficulty", "recall5_x_nchunks", "score_gap",
                   "latency_ratio", "cost_per_token"]:
            assert f in features, f"Missing interaction feature: {f}"

    def test_text_features(self):
        from src.features.build_features import engineer_features
        df = _make_sample_df()
        df_feat, features = engineer_features(df)
        assert "query_len" in features
        assert "query_word_count" in features
        assert (df_feat["query_len"] > 0).all()

    def test_split_data(self):
        from src.features.build_features import engineer_features, split_data
        df = _make_sample_df()
        df_feat, features = engineer_features(df)
        splits = split_data(df_feat, features)
        assert len(splits["train"]) == 60
        assert len(splits["val"]) == 20
        assert len(splits["test"]) == 20
        assert splits["X_train"].shape[1] == len(features)

    def test_no_nan_in_features(self):
        from src.features.build_features import engineer_features, split_data
        df = _make_sample_df()
        df_feat, features = engineer_features(df)
        splits = split_data(df_feat, features)
        assert not splits["X_train"].isnull().any().any(), "NaN found in X_train"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
