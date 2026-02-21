"""
Reusable plotting functions for the RAG-Optimize project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1)


def plot_target_distributions(df: pd.DataFrame):
    """Plot distributions of all 4 target variables."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df["is_correct"].value_counts().plot.pie(
        ax=axes[0, 0], autopct="%1.1f%%",
        colors=["#e74c3c", "#2ecc71"], startangle=90)
    axes[0, 0].set_title("is_correct", fontweight="bold")
    axes[0, 0].set_ylabel("")

    df["correctness_label"].value_counts().plot.bar(
        ax=axes[0, 1], color=["#2ecc71", "#f39c12", "#e74c3c"], edgecolor="k")
    axes[0, 1].set_title("correctness_label", fontweight="bold")

    df["hallucination_flag"].value_counts().plot.pie(
        ax=axes[1, 0], autopct="%1.1f%%",
        colors=["#3498db", "#e74c3c"], startangle=90)
    axes[1, 0].set_title("hallucination_flag", fontweight="bold")
    axes[1, 0].set_ylabel("")

    df["faithfulness_label"].value_counts().plot.bar(
        ax=axes[1, 1], color=["#2ecc71", "#95a5a6", "#e74c3c"], edgecolor="k")
    axes[1, 1].set_title("faithfulness_label", fontweight="bold")

    plt.suptitle("Target Distributions", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str]):
    """Plot correlation heatmap for numeric columns."""
    fig, ax = plt.subplots(figsize=(16, 12))
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_model_comparison(df: pd.DataFrame, group_col: str, target: str = "is_correct"):
    """Bar chart comparing performance across a categorical column."""
    perf = df.groupby(group_col).agg(
        accuracy=(target, "mean"),
        count=("run_id", "count"),
    ).round(4).sort_values("accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    perf["accuracy"].plot.barh(ax=ax, color="#2ecc71", edgecolor="k")
    ax.set_title(f"Accuracy by {group_col}", fontweight="bold")
    ax.set_xlim(0.5, 1.0)
    plt.tight_layout()
    return fig


def plot_pareto(df: pd.DataFrame):
    """Pareto analysis: accuracy vs cost vs latency."""
    pareto = df.groupby(["generator_model", "retrieval_strategy"]).agg(
        accuracy=("is_correct", "mean"),
        avg_cost=("total_cost_usd", "mean"),
        avg_latency=("total_latency_ms", "mean"),
        halluc_rate=("hallucination_flag", "mean"),
        count=("run_id", "count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    s1 = axes[0].scatter(pareto["avg_cost"], pareto["accuracy"],
                          c=pareto["halluc_rate"], cmap="RdYlGn_r",
                          s=pareto["count"] * 2, alpha=0.7, edgecolor="k")
    axes[0].set_xlabel("Cost (USD)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs Cost", fontweight="bold")
    plt.colorbar(s1, ax=axes[0], label="Hallucination Rate")

    s2 = axes[1].scatter(pareto["avg_latency"], pareto["accuracy"],
                          c=pareto["avg_cost"], cmap="viridis",
                          s=pareto["count"] * 2, alpha=0.7, edgecolor="k")
    axes[1].set_xlabel("Latency (ms)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs Latency", fontweight="bold")
    plt.colorbar(s2, ax=axes[1], label="Cost (USD)")

    plt.tight_layout()
    return fig
