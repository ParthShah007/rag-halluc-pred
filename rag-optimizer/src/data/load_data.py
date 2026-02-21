"""
Data loading and merging module.

Loads all 4 RAG evaluation CSVs and merges them into a single DataFrame.
"""

import os
import pandas as pd


# Project root is two levels up from this file (src/data/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def load_raw_data() -> dict[str, pd.DataFrame]:
    """Load all 4 raw CSV files.

    Returns:
        Dictionary mapping table name to DataFrame.
    """
    tables = {}
    for name in ["eval_runs", "scenarios", "rag_corpus_chunks", "rag_corpus_documents"]:
        path = os.path.join(RAW_DATA_DIR, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing data file: {path}")
        tables[name] = pd.read_csv(path)
        print(f"  Loaded {name}: {tables[name].shape}")
    return tables


def merge_datasets(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge eval_runs with scenarios and document metadata.

    Joins:
        - eval_runs + scenarios (on scenario_id) -> adds gold_chunk_ids, num_gold_chunks
        - eval_runs + documents (on doc_id) -> adds source_type, word_count, num_chunks

    Args:
        tables: Dictionary from load_raw_data().

    Returns:
        Merged DataFrame with columns from all relevant tables.
    """
    eval_runs = tables["eval_runs"].copy()
    scenarios = tables["scenarios"].copy()
    documents = tables["rag_corpus_documents"].copy()

    # ── Merge with scenarios ──
    # Only bring in columns not already in eval_runs
    eval_cols = set(eval_runs.columns)
    scenario_cols_to_add = [c for c in scenarios.columns
                            if c not in eval_cols or c == "scenario_id"]
    if "scenario_id" in eval_cols and "scenario_id" in scenarios.columns:
        merged = eval_runs.merge(
            scenarios[scenario_cols_to_add],
            on="scenario_id",
            how="left"
        )
    else:
        merged = eval_runs

    # ── Merge with documents ──
    # Extract first doc_id if doc_ids_used is a list-like string
    if "doc_ids_used" in merged.columns:
        merged["primary_doc_id"] = (
            merged["doc_ids_used"]
            .astype(str)
            .str.strip("[]'\"")
            .str.split(",")
            .str[0]
            .str.strip()
        )
        # Rename doc_id in documents to match
        docs_renamed = documents.rename(columns={"doc_id": "primary_doc_id"})
        doc_cols = ["primary_doc_id", "source_type", "word_count", "num_chunks"]
        doc_cols = [c for c in doc_cols if c in docs_renamed.columns]

        if len(doc_cols) > 1:
            merged = merged.merge(
                docs_renamed[doc_cols],
                on="primary_doc_id",
                how="left"
            )

    print(f"  Merged dataset: {merged.shape}")
    return merged


def load_and_merge() -> pd.DataFrame:
    """Convenience function: load all data and merge in one call.

    Returns:
        Fully merged DataFrame ready for feature engineering.
    """
    print("Loading raw data...")
    tables = load_raw_data()
    print("Merging datasets...")
    df = merge_datasets(tables)
    return df


def save_processed(df: pd.DataFrame, filename: str = "merged.csv") -> str:
    """Save processed DataFrame to data/processed/.

    Args:
        df: DataFrame to save.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved processed data: {path}")
    return path


if __name__ == "__main__":
    df = load_and_merge()
    save_processed(df)
    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
