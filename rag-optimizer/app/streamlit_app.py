"""
RAG-Optimize: Interactive Streamlit Dashboard v2

Features:
  - Tab 1: EDA Dashboard
  - Tab 2: Model Explorer
  - Tab 3: Config Recommender
  - Tab 4: Pareto Optimizer
  - Tab 5: 🔮 Live Predictor (interactive form → real-time predictions)

Run with: streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.load_data import load_and_merge
from src.features.build_features import (
    engineer_features, NUMERIC_FEATURES, NOMINAL_FEATURES,
    transform_single, load_feature_list, load_onehot_categories,
)
from src.models.predict import RAGPredictor

# ── Page Config ──
st.set_page_config(
    page_title="RAG-Optimize",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    .block-container { padding: 1rem 2rem; }
    h1, h2, h3 { color: #f1f1f1; }
    .stMetric { background: rgba(255,255,255,0.05); border-radius: 12px;
                padding: 16px; border: 1px solid rgba(255,255,255,0.1); }
    .prediction-card {
        background: rgba(255,255,255,0.08); border-radius: 16px;
        padding: 20px; border: 1px solid rgba(255,255,255,0.15);
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached data loading ──
@st.cache_data(show_spinner="Loading data...")
def get_data():
    df = load_and_merge()
    df_feat, features = engineer_features(df)
    return df, df_feat, features


@st.cache_resource(show_spinner="Loading models...")
def get_predictor():
    return RAGPredictor()


@st.cache_data(show_spinner=False)
def get_categories():
    try:
        return load_onehot_categories()
    except Exception:
        return {}


# ── Sidebar ──
st.sidebar.title("🚀 RAG-Optimize")
st.sidebar.markdown("Intelligent RAG Configuration Recommender")
st.sidebar.markdown("---")

tab_choice = st.sidebar.radio(
    "Navigate",
    ["📊 EDA Dashboard", "🤖 Model Explorer", "🎯 Config Recommender",
     "⚖️ Pareto Optimizer", "🔮 Live Predictor"],
    index=4,
)

# Load data
try:
    raw_df, feat_df, features = get_data()
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.info("Make sure data files are in `data/raw/`.")
    st.stop()


# ══════════════════════════════════════════════════════════
# Tab 1: EDA Dashboard
# ══════════════════════════════════════════════════════════
if tab_choice == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Runs", f"{len(raw_df):,}")
    col2.metric("Accuracy", f"{raw_df['is_correct'].mean():.1%}")
    col3.metric("Hallucination Rate", f"{raw_df['hallucination_flag'].mean():.1%}")
    col4.metric("Avg Latency", f"{raw_df['total_latency_ms'].mean():.0f} ms")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(raw_df, names="is_correct", title="Correctness Distribution",
                     color_discrete_sequence=["#e74c3c", "#2ecc71"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(raw_df, x="correctness_label", color="correctness_label",
                           title="Correctness Labels",
                           color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance Breakdown")
    group_col = st.selectbox("Group by:", ["generator_model", "retrieval_strategy",
                                            "embedding_model", "domain", "difficulty",
                                            "chunking_strategy"])
    perf = raw_df.groupby(group_col).agg(
        accuracy=("is_correct", "mean"),
        halluc_rate=("hallucination_flag", "mean"),
        avg_latency=("total_latency_ms", "mean"),
        avg_cost=("total_cost_usd", "mean"),
        count=("run_id", "count"),
    ).round(4).sort_values("accuracy", ascending=False)

    st.dataframe(perf.style.background_gradient(cmap="RdYlGn", subset=["accuracy"])
                 .background_gradient(cmap="RdYlGn_r", subset=["halluc_rate"]),
                 use_container_width=True)

    fig = px.bar(perf.reset_index(), x=group_col, y=["accuracy", "halluc_rate"],
                 barmode="group", title=f"Accuracy & Hallucination by {group_col}",
                 color_discrete_sequence=["#2ecc71", "#e74c3c"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Correlations")
    num_cols = [c for c in NUMERIC_FEATURES if c in raw_df.columns]
    corr = raw_df[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title="Numeric Feature Correlations", aspect="auto")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# Tab 2: Model Explorer
# ══════════════════════════════════════════════════════════
elif tab_choice == "🤖 Model Explorer":
    st.title("🤖 Model Performance Explorer")
    try:
        predictor = get_predictor()
        metrics = predictor.get_metrics()
    except Exception:
        st.warning("Models not trained yet. Run `python -m src.models.train` first.")
        st.stop()

    if metrics:
        st.subheader("Classification Models")
        if "classification" in metrics:
            clf_df = pd.DataFrame(metrics["classification"])
            st.dataframe(clf_df.style.background_gradient(cmap="RdYlGn",
                         subset=[c for c in ["accuracy", "auc", "f1_macro", "recall_minority"]
                                 if c in clf_df.columns]),
                         use_container_width=True)

        st.subheader("Regression Models")
        if "regression" in metrics:
            reg_df = pd.DataFrame(metrics["regression"])
            st.dataframe(reg_df.style.background_gradient(cmap="RdYlGn", subset=["r2"]),
                         use_container_width=True)

    st.subheader("Feature Importance (Top 20)")
    try:
        model = predictor.models.get("correctness")
        if model and features:
            importances = pd.Series(
                model.feature_importances_, index=features
            ).sort_values(ascending=True).tail(20)
            fig = px.bar(x=importances.values, y=importances.index,
                         orientation="h", title="Top 20 Feature Importances (Correctness Model)",
                         labels={"x": "Importance", "y": "Feature"},
                         color=importances.values, color_continuous_scale="viridis")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importance not available: {e}")


# ══════════════════════════════════════════════════════════
# Tab 3: Config Recommender
# ══════════════════════════════════════════════════════════
elif tab_choice == "🎯 Config Recommender":
    st.title("🎯 RAG Configuration Recommender")
    try:
        predictor = get_predictor()
    except Exception:
        st.warning("Models not trained. Run `python -m src.models.train`.")
        st.stop()

    st.subheader("Top Configurations by Accuracy")
    best = raw_df.groupby(["generator_model", "retrieval_strategy", "embedding_model"]).agg(
        accuracy=("is_correct", "mean"),
        halluc_rate=("hallucination_flag", "mean"),
        avg_latency=("total_latency_ms", "mean"),
        avg_cost=("total_cost_usd", "mean"),
        count=("run_id", "count"),
    ).round(4)
    best = best[best["count"] >= 10].sort_values("accuracy", ascending=False).head(15)
    st.dataframe(best.style.background_gradient(cmap="RdYlGn", subset=["accuracy"])
                 .background_gradient(cmap="RdYlGn_r", subset=["halluc_rate", "avg_cost"]),
                 use_container_width=True)

    st.subheader("Safest Configurations (Lowest Hallucination)")
    safe = raw_df.groupby(["generator_model", "retrieval_strategy"]).agg(
        halluc_rate=("hallucination_flag", "mean"),
        accuracy=("is_correct", "mean"),
        count=("run_id", "count"),
    ).round(4)
    safe = safe[safe["count"] >= 10].sort_values("halluc_rate").head(10)
    st.dataframe(safe.style.background_gradient(cmap="RdYlGn_r", subset=["halluc_rate"]),
                 use_container_width=True)


# ══════════════════════════════════════════════════════════
# Tab 4: Pareto Optimizer
# ══════════════════════════════════════════════════════════
elif tab_choice == "⚖️ Pareto Optimizer":
    st.title("⚖️ Pareto Frontier: Cost vs Accuracy vs Latency")

    pareto = raw_df.groupby(["generator_model", "retrieval_strategy"]).agg(
        accuracy=("is_correct", "mean"),
        avg_cost=("total_cost_usd", "mean"),
        avg_latency=("total_latency_ms", "mean"),
        halluc_rate=("hallucination_flag", "mean"),
        count=("run_id", "count"),
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        max_cost = st.slider("Max Cost (USD)", 0.0, float(pareto["avg_cost"].max()),
                              float(pareto["avg_cost"].max()), 0.001)
    with col2:
        max_latency = st.slider("Max Latency (ms)", 0, int(pareto["avg_latency"].max()),
                                 int(pareto["avg_latency"].max()), 100)

    filtered = pareto[(pareto["avg_cost"] <= max_cost) & (pareto["avg_latency"] <= max_latency)]

    fig1 = px.scatter(filtered, x="avg_cost", y="accuracy",
                      color="halluc_rate", size="count",
                      hover_data=["generator_model", "retrieval_strategy"],
                      title="Accuracy vs Cost",
                      color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(filtered, x="avg_latency", y="accuracy",
                      color="avg_cost", size="count",
                      hover_data=["generator_model", "retrieval_strategy"],
                      title="Accuracy vs Latency",
                      color_continuous_scale="viridis")
    st.plotly_chart(fig2, use_container_width=True)

    if len(filtered) > 0:
        st.subheader("📋 Recommendations")
        best_acc = filtered.loc[filtered["accuracy"].idxmax()]
        cheapest = filtered.loc[filtered["avg_cost"].idxmin()]
        fastest = filtered.loc[filtered["avg_latency"].idxmin()]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**🏆 Highest Accuracy**")
            st.write(f"Model: `{best_acc['generator_model']}`")
            st.write(f"Strategy: `{best_acc['retrieval_strategy']}`")
            st.metric("Accuracy", f"{best_acc['accuracy']:.4f}")
        with c2:
            st.markdown("**💰 Cheapest**")
            st.write(f"Model: `{cheapest['generator_model']}`")
            st.metric("Cost", f"${cheapest['avg_cost']:.4f}")
        with c3:
            st.markdown("**⚡ Fastest**")
            st.write(f"Model: `{fastest['generator_model']}`")
            st.metric("Latency", f"{fastest['avg_latency']:.0f} ms")


# ══════════════════════════════════════════════════════════
# Tab 5: 🔮 Live Predictor
# ══════════════════════════════════════════════════════════
elif tab_choice == "🔮 Live Predictor":
    st.title("🔮 Live RAG Configuration Predictor")
    st.markdown("Configure a RAG pipeline below and get **real-time predictions** for "
                "correctness, hallucination risk, latency, and cost.")

    try:
        predictor = get_predictor()
        categories = get_categories()
    except Exception as e:
        st.error(f"Models not loaded: {e}")
        st.info("Run `python -m src.models.train` first.")
        st.stop()

    if not categories:
        st.warning("Category data not found. Please retrain models.")
        st.stop()

    st.markdown("---")

    # ── Input Form ──
    st.subheader("⚙️ Pipeline Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🤖 Model Selection**")
        generator_model = st.selectbox("Generator Model",
            categories.get("generator_model", ["gpt-4o"]))
        embedding_model = st.selectbox("Embedding Model",
            categories.get("embedding_model", ["text-embedding-3-large"]))
        reranker_model = st.selectbox("Reranker",
            categories.get("reranker_model", ["none"]))

    with col2:
        st.markdown("**🔍 Retrieval Config**")
        retrieval_strategy = st.selectbox("Retrieval Strategy",
            categories.get("retrieval_strategy", ["hybrid"]))
        chunking_strategy = st.selectbox("Chunking Strategy",
            categories.get("chunking_strategy", ["fixed_500"]))
        n_retrieved_chunks = st.slider("Top-K Chunks", 1, 50, 5)

    with col3:
        st.markdown("**📋 Query Context**")
        domain = st.selectbox("Domain",
            categories.get("domain", ["finance"]))
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
        task_type = st.selectbox("Task Type",
            categories.get("task_type", ["factual"]))

    st.markdown("---")

    st.subheader("📊 Retrieval Quality Scores")
    col4, col5, col6 = st.columns(3)

    with col4:
        top1_score = st.slider("Top-1 Similarity Score", 0.0, 1.0, 0.75, 0.01)
        mean_retrieved_score = st.slider("Mean Chunk Score", 0.0, 1.0, 0.55, 0.01)
        mrr_at_10 = st.slider("MRR@10", 0.0, 1.0, 0.6, 0.01)

    with col5:
        recall_at_5 = st.slider("Recall@5", 0.0, 1.0, 1.0, 0.1)
        recall_at_10 = st.slider("Recall@10", 0.0, 1.0, 1.0, 0.1)
        has_relevant_in_top5 = st.selectbox("Has Relevant in Top-5?", [1, 0])

    with col6:
        has_relevant_in_top10 = st.selectbox("Has Relevant in Top-10?", [1, 0])
        has_answer_in_corpus = st.selectbox("Has Answer in Corpus?", [1, 0])
        is_noanswer_probe = st.selectbox("Is No-Answer Probe?", [0, 1])

    st.markdown("---")

    st.subheader("🎛️ Generation Parameters")
    col7, col8, col9 = st.columns(3)

    with col7:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.3, 0.1)
        top_p = st.slider("Top-P", 0.0, 1.0, 0.9, 0.05)

    with col8:
        prompt_tokens = st.number_input("Prompt Tokens", 50, 10000, 800)
        answer_tokens = st.number_input("Answer Tokens", 10, 5000, 150)
        context_window_tokens = st.selectbox("Context Window", [4096, 8192, 16384, 32768], index=1)
        max_new_tokens = st.number_input("Max New Tokens", 64, 4096, 1024)

    with col9:
        used_long_context_window = st.selectbox("Used Long Context?", [0, 1])
        answered_without_retrieval = st.selectbox("Answered Without Retrieval?", [0, 1])
        eval_mode = st.selectbox("Eval Mode",
            categories.get("eval_mode", ["llm_judge"]))
        stop_reason = st.selectbox("Stop Reason",
            categories.get("stop_reason", ["eos"]))

    st.markdown("---")

    st.subheader("⏱️ Latency & Cost")
    col10, col11 = st.columns(2)

    with col10:
        latency_ms_retrieval = st.number_input("Retrieval Latency (ms)", 0.0, 10000.0, 150.0)
        latency_ms_generation = st.number_input("Generation Latency (ms)", 0.0, 30000.0, 800.0)
        total_latency_ms = st.number_input("Total Latency (ms)", 0.0, 60000.0, 950.0)

    with col11:
        total_cost_usd = st.number_input("Total Cost (USD)", 0.0, 10.0, 0.01, format="%.6f")

    st.markdown("---")
    st.markdown("**📝 Query**")
    query = st.text_area("Enter your query:", value="What are the main causes of inflation?",
                         height=80)

    # ── Predict Button ──
    st.markdown("---")

    if st.button("🚀 **Predict Performance**", use_container_width=True, type="primary"):

        config = {
            # Pipeline config
            "domain": domain,
            "task_type": task_type,
            "difficulty": difficulty,
            "generator_model": generator_model,
            "embedding_model": embedding_model,
            "reranker_model": reranker_model,
            "retrieval_strategy": retrieval_strategy,
            "chunking_strategy": chunking_strategy,
            "eval_mode": eval_mode,
            "stop_reason": stop_reason,
            # Retrieval quality
            "top1_score": top1_score,
            "mean_retrieved_score": mean_retrieved_score,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_10,
            "mrr_at_10": mrr_at_10,
            "n_retrieved_chunks": n_retrieved_chunks,
            "has_relevant_in_top5": has_relevant_in_top5,
            "has_relevant_in_top10": has_relevant_in_top10,
            "has_answer_in_corpus": has_answer_in_corpus,
            "is_noanswer_probe": is_noanswer_probe,
            # Generation params
            "temperature": temperature,
            "top_p": top_p,
            "prompt_tokens": prompt_tokens,
            "answer_tokens": answer_tokens,
            "context_window_tokens": context_window_tokens,
            "max_new_tokens": max_new_tokens,
            "used_long_context_window": used_long_context_window,
            "answered_without_retrieval": answered_without_retrieval,
            # Latency & cost
            "latency_ms_retrieval": latency_ms_retrieval,
            "latency_ms_generation": latency_ms_generation,
            "total_latency_ms": total_latency_ms,
            "total_cost_usd": total_cost_usd,
            # Query
            "query": query,
        }

        with st.spinner("Running predictions..."):
            try:
                X_input = transform_single(config)

                # Correctness
                corr_model = predictor.models.get("correctness")
                thresh_path = os.path.join(PROJECT_ROOT, "models", "threshold_correctness.joblib")
                threshold = joblib.load(thresh_path) if os.path.exists(thresh_path) else 0.5
                corr_proba = corr_model.predict_proba(X_input)[:, 1][0]
                corr_pred = int(corr_proba >= threshold)

                # Hallucination
                hal_model = predictor.models.get("hallucination")
                thresh_path_h = os.path.join(PROJECT_ROOT, "models", "threshold_hallucination.joblib")
                threshold_h = joblib.load(thresh_path_h) if os.path.exists(thresh_path_h) else 0.5
                hal_proba = hal_model.predict_proba(X_input)[:, 1][0]
                hal_pred = int(hal_proba >= threshold_h)

                # Latency estimate
                lat_feats = predictor.models.get("latency_features_list", features)
                lat_cols = [c for c in lat_feats if c in X_input.columns]
                lat_pred = predictor.models["latency"].predict(X_input[lat_cols])[0] if lat_cols else 0

                # Cost estimate
                cost_feats = predictor.models.get("cost_features_list", features)
                cost_cols = [c for c in cost_feats if c in X_input.columns]
                cost_pred = predictor.models["cost"].predict(X_input[cost_cols])[0] if cost_cols else 0

                # Task type
                task_pred = predictor.classify_query(query)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()

        # ── Display Results ──
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            color = "#2ecc71" if corr_pred == 1 else "#e74c3c"
            st.markdown(f"""
            <div class="prediction-card" style="border-left: 4px solid {color}">
                <h3>✅ Correctness</h3>
                <h1 style="color: {color}">{'CORRECT' if corr_pred else 'INCORRECT'}</h1>
                <p>Confidence: {corr_proba:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            color = "#e74c3c" if hal_pred == 1 else "#2ecc71"
            st.markdown(f"""
            <div class="prediction-card" style="border-left: 4px solid {color}">
                <h3>⚠️ Hallucination Risk</h3>
                <h1 style="color: {color}">{'HIGH RISK' if hal_pred else 'LOW RISK'}</h1>
                <p>Probability: {hal_proba:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class="prediction-card" style="border-left: 4px solid #3498db">
                <h3>⏱️ Est. Latency</h3>
                <h1 style="color: #3498db">{lat_pred:.0f} ms</h1>
                <p>{lat_pred/1000:.2f} seconds</p>
            </div>
            """, unsafe_allow_html=True)

        with r4:
            st.markdown(f"""
            <div class="prediction-card" style="border-left: 4px solid #f39c12">
                <h3>💰 Est. Cost</h3>
                <h1 style="color: #f39c12">${cost_pred:.4f}</h1>
                <p>Per query</p>
            </div>
            """, unsafe_allow_html=True)

        # Task type
        st.info(f"📝 Detected query type: **{task_pred}**")

        # Confidence bars
        st.subheader("Confidence Breakdown")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[corr_proba], y=["Correctness"], orientation="h",
            marker_color="#2ecc71" if corr_pred else "#e74c3c",
            text=[f"{corr_proba:.1%}"], textposition="auto",
        ))
        fig.add_trace(go.Bar(
            x=[1 - hal_proba], y=["Safety (1-Halluc)"], orientation="h",
            marker_color="#2ecc71" if not hal_pred else "#e74c3c",
            text=[f"{1-hal_proba:.1%}"], textposition="auto",
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 1], title="Probability"),
            showlegend=False,
            height=200,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # SHAP explanation
        st.subheader("🔍 SHAP Feature Explanations")
        try:
            import shap
            explainer = shap.TreeExplainer(corr_model)
            shap_values = explainer.shap_values(X_input)

            if isinstance(shap_values, list):
                sv = shap_values[1]
            else:
                sv = shap_values

            top_n = 15
            abs_shap = np.abs(sv[0])
            top_idx = np.argsort(abs_shap)[::-1][:top_n]
            top_features = [features[i] for i in top_idx if i < len(features)]
            top_vals = [float(sv[0][i]) for i in top_idx if i < len(features)]

            colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top_vals]

            fig = go.Figure(go.Bar(
                x=top_vals, y=top_features, orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in top_vals],
                textposition="auto",
            ))
            fig.update_layout(
                title="Top Feature Contributions (Green=↑Correct, Red=↑Incorrect)",
                xaxis_title="SHAP Value",
                height=400,
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"SHAP not available: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built for DataHack 2026")

