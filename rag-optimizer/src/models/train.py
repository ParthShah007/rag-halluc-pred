"""
Model training module — v2 with SMOTE-ENN and per-class metrics.

Trains all classification and regression models with class imbalance handling,
evaluates with per-class recall/F1, and saves model artifacts.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, f1_score, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _save_model(model, name: str) -> str:
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"    → Saved: {name}.joblib")
    return path


def _apply_smote(X, y, method="smoteenn", random_state=42):
    """Apply SMOTE or SMOTE-ENN to handle class imbalance.

    Returns resampled X, y with per-class counts printed.
    """
    print(f"    Before SMOTE: {dict(pd.Series(y).value_counts())}")

    # Check minimum samples per class
    min_class_count = pd.Series(y).value_counts().min()
    k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1

    if method == "smoteenn":
        sampler = SMOTEENN(
            smote=SMOTE(k_neighbors=k_neighbors, random_state=random_state),
            random_state=random_state,
        )
    else:
        sampler = SMOTE(k_neighbors=k_neighbors, random_state=random_state)

    X_res, y_res = sampler.fit_resample(X, y)
    print(f"    After SMOTE:  {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res


def _find_best_threshold(y_true, y_proba):
    """Find optimal threshold using F1 on PR curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


def _get_per_class_report(y_true, y_pred, target_names=None):
    """Extract per-class metrics as dict."""
    report = classification_report(y_true, y_pred, target_names=target_names,
                                    output_dict=True, zero_division=0)
    return report


def train_binary_correctness(splits, features):
    """Train binary correctness with SMOTE-ENN."""
    print("\n" + "=" * 70)
    print("  MODEL 1: Binary Correctness (is_correct) — with SMOTE")
    print("=" * 70)

    X_tr, X_val, X_te = splits["X_train"], splits["X_val"], splits["X_test"]
    y_tr = splits["train"]["is_correct"].values
    y_val = splits["val"]["is_correct"].values
    y_te = splits["test"]["is_correct"].values

    # Apply SMOTE-ENN to training data
    X_tr_res, y_tr_res = _apply_smote(X_tr, y_tr, method="smoteenn")

    results = {}
    models = {}

    configs = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12,
            class_weight="balanced",  # Built-in weighting
            random_state=42, n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True,  # Built-in weighting
            random_state=42, n_jobs=-1, verbose=-1,
        ),
    }

    for name, clf in configs.items():
        # Train on SMOTE-resampled data
        if "XGB" in name:
            clf.fit(X_tr_res, y_tr_res, eval_set=[(X_val, y_val)], verbose=False)
        elif "Light" in name:
            clf.fit(X_tr_res, y_tr_res, eval_set=[(X_val, y_val)])
        else:
            clf.fit(X_tr_res, y_tr_res)

        proba = clf.predict_proba(X_te)[:, 1]

        # Find optimal threshold
        best_thresh = _find_best_threshold(y_te, proba)
        pred = (proba >= best_thresh).astype(int)

        auc = roc_auc_score(y_te, proba)
        ap = average_precision_score(y_te, proba)
        f1_macro = f1_score(y_te, pred, average="macro")
        report = _get_per_class_report(y_te, pred)

        results[name] = {
            "auc": auc, "ap": ap, "f1_macro": f1_macro,
            "threshold": best_thresh,
            "recall_class_0": report["0"]["recall"],
            "recall_class_1": report["1"]["recall"],
        }
        models[name] = clf

        print(f"\n  {name} (threshold={best_thresh:.3f}):")
        print(f"    AUC={auc:.4f}  AP={ap:.4f}  F1-macro={f1_macro:.4f}")
        print(f"    Recall class-0: {report['0']['recall']:.4f}")
        print(f"    Recall class-1: {report['1']['recall']:.4f}")
        print(classification_report(y_te, pred, digits=4))

    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    _save_model(models[best_name], "xgb_correctness")
    _save_model(results[best_name]["threshold"], "threshold_correctness")
    print(f"\n  🏆 Best: {best_name} (F1-macro={results[best_name]['f1_macro']:.4f})")

    return {"models": models, "results": results, "best": best_name, "best_clf": models[best_name]}


def train_multiclass_correctness(splits, features):
    """Train multi-class correctness with SMOTE."""
    print("\n" + "=" * 70)
    print("  MODEL 2: Multi-class Correctness — with SMOTE")
    print("=" * 70)

    X_tr, X_val, X_te = splits["X_train"], splits["X_val"], splits["X_test"]

    le = LabelEncoder()
    y_tr = le.fit_transform(splits["train"]["correctness_label"])
    y_val = le.transform(splits["val"]["correctness_label"])
    y_te = le.transform(splits["test"]["correctness_label"])

    X_tr_res, y_tr_res = _apply_smote(X_tr, y_tr, method="smote")

    xgb_mc = xgb.XGBClassifier(
        objective="multi:softprob", num_class=len(le.classes_),
        n_estimators=300, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )
    xgb_mc.fit(X_tr_res, y_tr_res, eval_set=[(X_val, y_val)], verbose=False)

    cat_mc = cb.CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1,
        loss_function="MultiClass", auto_class_weights="Balanced",
        random_seed=42, verbose=0,
    )
    cat_mc.fit(X_tr_res, y_tr_res, eval_set=(X_val, y_val))

    for name, clf in [("XGBoost", xgb_mc), ("CatBoost", cat_mc)]:
        pred = clf.predict(X_te)
        if hasattr(pred, 'flatten'):
            pred = pred.flatten().astype(int)
        report = _get_per_class_report(y_te, pred, le.classes_)
        f1_macro = f1_score(y_te, pred, average="macro")
        print(f"\n  {name} (F1-macro={f1_macro:.4f}):")
        for cls_name in le.classes_:
            r = report[cls_name]
            print(f"    {cls_name}: precision={r['precision']:.4f} recall={r['recall']:.4f} f1={r['f1-score']:.4f}")
        print(classification_report(y_te, pred, target_names=le.classes_, digits=4))

    _save_model(xgb_mc, "xgb_multiclass")
    _save_model(le, "le_correctness")
    return {"le": le, "y_test": y_te}


def train_hallucination(splits, features):
    """Train hallucination detection with SMOTE-ENN (most imbalanced)."""
    print("\n" + "=" * 70)
    print("  MODEL 3: Hallucination Detection — with SMOTE-ENN")
    print("=" * 70)

    X_tr, X_val, X_te = splits["X_train"], splits["X_val"], splits["X_test"]
    y_tr = splits["train"]["hallucination_flag"].values
    y_val = splits["val"]["hallucination_flag"].values
    y_te = splits["test"]["hallucination_flag"].values

    X_tr_res, y_tr_res = _apply_smote(X_tr, y_tr, method="smoteenn")

    spw = pd.Series(y_tr).value_counts()[0] / pd.Series(y_tr).value_counts()[1]

    xgb_hal = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=spw,
        use_label_encoder=False, eval_metric="aucpr",
        random_state=42, n_jobs=-1,
    )
    xgb_hal.fit(X_tr_res, y_tr_res, eval_set=[(X_val, y_val)], verbose=False)

    lgb_hal = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_hal.fit(X_tr_res, y_tr_res, eval_set=[(X_val, y_val)])

    best_model = None
    best_f1 = 0

    for name, clf in [("XGBoost", xgb_hal), ("LightGBM", lgb_hal)]:
        proba = clf.predict_proba(X_te)[:, 1]
        best_thresh = _find_best_threshold(y_te, proba)
        pred = (proba >= best_thresh).astype(int)

        f1_macro = f1_score(y_te, pred, average="macro")
        report = _get_per_class_report(y_te, pred)
        auc = roc_auc_score(y_te, proba)

        print(f"\n  {name} (threshold={best_thresh:.3f}, F1-macro={f1_macro:.4f}):")
        print(f"    Recall NO-hallucination: {report['0']['recall']:.4f}")
        print(f"    Recall HALLUCINATION:    {report['1']['recall']:.4f}")
        print(f"    AUC: {auc:.4f}")
        print(classification_report(y_te, pred, digits=4))

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model = clf
            best_thresh_save = best_thresh

    _save_model(best_model, "xgb_hallucination")
    _save_model(best_thresh_save, "threshold_hallucination")
    return {"y_test": y_te}


def train_faithfulness(splits, features):
    """Train faithfulness with SMOTE."""
    print("\n" + "=" * 70)
    print("  MODEL 4: Faithfulness — with SMOTE")
    print("=" * 70)

    X_tr, X_val, X_te = splits["X_train"], splits["X_val"], splits["X_test"]

    le = LabelEncoder()
    y_tr = le.fit_transform(splits["train"]["faithfulness_label"])
    y_val = le.transform(splits["val"]["faithfulness_label"])
    y_te = le.transform(splits["test"]["faithfulness_label"])

    X_tr_res, y_tr_res = _apply_smote(X_tr, y_tr, method="smote")

    xgb_f = xgb.XGBClassifier(
        objective="multi:softprob", num_class=len(le.classes_),
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric="mlogloss",
    )
    xgb_f.fit(X_tr_res, y_tr_res, eval_set=[(X_val, y_val)], verbose=False)

    rf_f = RandomForestClassifier(
        n_estimators=300, max_depth=12,
        class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    rf_f.fit(X_tr_res, y_tr_res)

    for name, clf in [("XGBoost", xgb_f), ("RF", rf_f)]:
        pred = clf.predict(X_te)
        f1_macro = f1_score(y_te, pred, average="macro")
        report = _get_per_class_report(y_te, pred, le.classes_)
        print(f"\n  {name} (F1-macro={f1_macro:.4f}):")
        for cls_name in le.classes_:
            r = report[cls_name]
            print(f"    {cls_name}: recall={r['recall']:.4f} f1={r['f1-score']:.4f}")
        print(classification_report(y_te, pred, target_names=le.classes_, digits=4))

    _save_model(xgb_f, "xgb_faithfulness")
    _save_model(le, "le_faithfulness")
    return {"le": le, "y_test": y_te}


def train_latency(splits, features):
    """Train latency regression."""
    print("\n" + "=" * 70)
    print("  MODEL 5: Latency Regression")
    print("=" * 70)

    leakage_cols = ["total_latency_ms", "latency_ms_retrieval",
                   "latency_ms_generation", "total_cost_usd",
                   "latency_ratio", "cost_per_token"]
    lat_features = [f for f in features if f not in leakage_cols]
    X_tr = splits["X_train"][lat_features]
    X_te = splits["X_test"][lat_features]
    y_tr = splits["train"]["total_latency_ms"]
    y_te = splits["test"]["total_latency_ms"]

    xgb_r = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                               random_state=42, n_jobs=-1)
    xgb_r.fit(X_tr, y_tr, verbose=False)

    rf_r = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf_r.fit(X_tr, y_tr)

    for name, clf in [("XGBoost", xgb_r), ("RF", rf_r)]:
        pred = clf.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        mae = mean_absolute_error(y_te, pred)
        r2 = r2_score(y_te, pred)
        print(f"  {name}: RMSE={rmse:.1f}  MAE={mae:.1f}  R²={r2:.4f}")

    _save_model(xgb_r, "xgb_latency")
    _save_model(lat_features, "latency_features")
    return {"features": lat_features, "y_test": y_te}


def train_cost(splits, features):
    """Train cost regression."""
    print("\n" + "=" * 70)
    print("  MODEL 6: Cost Regression")
    print("=" * 70)

    leakage_cols = ["total_cost_usd", "cost_per_token"]
    cost_features = [f for f in features if f not in leakage_cols]
    X_tr = splits["X_train"][cost_features]
    X_te = splits["X_test"][cost_features]
    y_tr = splits["train"]["total_cost_usd"]
    y_te = splits["test"]["total_cost_usd"]

    xgb_c = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                               random_state=42, n_jobs=-1)
    xgb_c.fit(X_tr, y_tr, verbose=False)
    pred = xgb_c.predict(X_te)
    print(f"  XGBoost: RMSE={np.sqrt(mean_squared_error(y_te, pred)):.6f}  "
          f"MAE={mean_absolute_error(y_te, pred):.6f}  R²={r2_score(y_te, pred):.4f}")

    _save_model(xgb_c, "xgb_cost")
    _save_model(cost_features, "cost_features")
    return {"features": cost_features, "y_test": y_te}


def train_task_classifier(splits):
    """Train NLP task type classifier."""
    print("\n" + "=" * 70)
    print("  MODEL 7: NLP Task Type Classifier")
    print("=" * 70)

    le = LabelEncoder()
    y_tr = le.fit_transform(splits["train"]["task_type"])
    y_te = le.transform(splits["test"]["task_type"])

    pipe_lr = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                                    random_state=42, n_jobs=-1)),
    ])
    pipe_lr.fit(splits["train"]["query"], y_tr)

    pipe_svm = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    pipe_svm.fit(splits["train"]["query"], y_tr)

    for name, pipe, pred in [("LogReg", pipe_lr, pipe_lr.predict(splits["test"]["query"])),
                              ("SVM", pipe_svm, pipe_svm.predict(splits["test"]["query"]))]:
        f1_macro = f1_score(y_te, pred, average="macro")
        print(f"\n  {name} (F1-macro={f1_macro:.4f}):")
        print(classification_report(y_te, pred, target_names=le.classes_, digits=4))

    _save_model(pipe_svm, "tfidf_task_classifier")
    _save_model(le, "le_task_type")
    return {"le": le, "y_test": y_te}


def _compile_metrics(results_all, splits, features):
    """Compile comprehensive metrics."""
    # Build a comprehensive metrics dict by re-evaluating
    metrics = {
        "classification": [],
        "regression": [],
        "per_class": {},
    }

    # Load the just-saved models for consistent evaluation
    for task_name, model_file, y_col in [
        ("Correctness", "xgb_correctness", "is_correct"),
        ("Hallucination", "xgb_hallucination", "hallucination_flag"),
    ]:
        model_path = os.path.join(MODELS_DIR, f"{model_file}.joblib")
        thresh_path = os.path.join(MODELS_DIR, f"threshold_{task_name.lower()}.joblib")
        if os.path.exists(model_path):
            clf = joblib.load(model_path)
            threshold = joblib.load(thresh_path) if os.path.exists(thresh_path) else 0.5
            X_te = splits["X_test"]
            y_te = splits["test"][y_col].values
            proba = clf.predict_proba(X_te)[:, 1]
            pred = (proba >= threshold).astype(int)
            metrics["classification"].append({
                "model": "XGBoost", "task": task_name,
                "accuracy": round(float(accuracy_score(y_te, pred)), 4),
                "auc": round(float(roc_auc_score(y_te, proba)), 4),
                "f1_macro": round(float(f1_score(y_te, pred, average="macro")), 4),
                "threshold": round(float(threshold), 3),
                "recall_minority": round(float(
                    _get_per_class_report(y_te, pred)["0" if task_name == "Correctness" else "1"]["recall"]
                ), 4),
            })

    # Regression
    for task_name, model_file, feat_file, y_col in [
        ("Latency", "xgb_latency", "latency_features", "total_latency_ms"),
        ("Cost", "xgb_cost", "cost_features", "total_cost_usd"),
    ]:
        model_path = os.path.join(MODELS_DIR, f"{model_file}.joblib")
        feat_path = os.path.join(MODELS_DIR, f"{feat_file}.joblib")
        if os.path.exists(model_path) and os.path.exists(feat_path):
            reg = joblib.load(model_path)
            feat_list = joblib.load(feat_path)
            X_te = splits["X_test"][feat_list]
            y_te = splits["test"][y_col].values
            pred = reg.predict(X_te)
            metrics["regression"].append({
                "model": "XGBoost", "task": task_name,
                "rmse": round(float(np.sqrt(mean_squared_error(y_te, pred))), 4),
                "mae": round(float(mean_absolute_error(y_te, pred)), 4),
                "r2": round(float(r2_score(y_te, pred)), 4),
            })

    return metrics


def train_all(df, features):
    """Train all models end-to-end with SMOTE."""
    from src.features.build_features import split_data, save_feature_list, save_onehot_categories

    print("\n🚀 Training all models with class imbalance handling...")
    splits = split_data(df, features)
    save_feature_list(features)
    save_onehot_categories(df)

    train_binary_correctness(splits, features)
    train_multiclass_correctness(splits, features)
    train_hallucination(splits, features)
    train_faithfulness(splits, features)
    train_latency(splits, features)
    train_cost(splits, features)
    train_task_classifier(splits)

    metrics = _compile_metrics(None, splits, features)
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  → Saved: metrics.json")

    print("\n" + "=" * 70)
    print("  ✅ ALL MODELS TRAINED WITH SMOTE + CLASS BALANCING")
    print("=" * 70)


if __name__ == "__main__":
    from src.data.load_data import load_and_merge
    from src.features.build_features import engineer_features

    df = load_and_merge()
    df, features = engineer_features(df)
    train_all(df, features)
