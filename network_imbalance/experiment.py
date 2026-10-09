import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from .data import load_dataset, class_distribution

def _binary_encode(y):
    # For IDS datasets, BENIGN is treated as the normal class when present.
    normalized = y.astype(str).str.strip().str.upper()
    benign_names = {"BENIGN", "NORMAL", "NORMAL TRAFFIC", "BENIGN TRAFFIC"}
    if normalized.isin(benign_names).any():
        return (~normalized.isin(benign_names)).astype(int).to_numpy()
    labels = sorted(normalized.unique())
    if len(labels) != 2:
        preview = ", ".join(labels[:12])
        suffix = " ..." if len(labels) > 12 else ""
        raise ValueError(
            "This experiment needs a binary target. If your folder contains one CSV per attack "
            "type, add a benign/normal traffic CSV as well. "
            f"Found {len(labels)} distinct labels: {preview}{suffix}"
        )
    return normalized.map({labels[0]: 0, labels[1]: 1}).to_numpy()

def _build_estimators(random_state, top_k):
    base = LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state)
    return {
        "baseline": Pipeline([("model", clone(base))]),
        "standardized": Pipeline([("scaler", StandardScaler()), ("model", clone(base))]),
        "feature_selection": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(mutual_info_classif, k=top_k)),
            ("model", clone(base))
        ]),
        "class_weighted": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, solver="liblinear",
                                         class_weight="balanced", random_state=random_state))
        ]),
        "smote": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            ("model", clone(base))
        ]),
    }

def run_experiment(config):
    out = Path(config.output_dir)
    (out/"figures").mkdir(parents=True, exist_ok=True)
    (out/"tables").mkdir(parents=True, exist_ok=True)

    X, y_text = load_dataset(config.data_path, config.target_column, config.max_rows)
    y = _binary_encode(y_text)

    dist = class_distribution(y_text)
    dist.to_csv(out/"tables/class_distribution.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.random_state
    )

    # Mutual-information selector cannot request more features than available.
    top_k = min(config.top_k_features, X_train.shape[1])
    estimators = _build_estimators(config.random_state, top_k)
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    scoring = {"accuracy":"accuracy", "balanced_accuracy":"balanced_accuracy",
               "precision":"precision", "recall":"recall", "f1":"f1",
               "pr_auc":"average_precision", "roc_auc":"roc_auc"}

    rows=[]
    for name in config.strategies:
        if name not in estimators: continue
        pipe=estimators[name]
        # SMOTE can be expensive on huge datasets; cap only its training sample.
        Xcv, ycv = X_train, y_train
        if name=="smote" and len(Xcv)>config.smote_max_rows:
            rng=np.random.default_rng(config.random_state)
            idx=rng.choice(len(Xcv), config.smote_max_rows, replace=False)
            Xcv, ycv=Xcv.iloc[idx], ycv[idx]
        start=time.perf_counter()
        cvres=cross_validate(pipe, Xcv, ycv, cv=cv, scoring=scoring,
                             n_jobs=config.n_jobs, error_score="raise")
        fit_seconds=time.perf_counter()-start
        pipe.fit(Xcv,ycv)
        pred=pipe.predict(X_test)
        proba=pipe.predict_proba(X_test)[:,1]
        tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
        rows.append({
            "strategy":name,
            "accuracy":accuracy_score(y_test,pred),
            "balanced_accuracy":balanced_accuracy_score(y_test,pred),
            "precision":precision_score(y_test,pred,zero_division=0),
            "recall":recall_score(y_test,pred,zero_division=0),
            "f1":f1_score(y_test,pred,zero_division=0),
            "pr_auc":average_precision_score(y_test,proba),
            "roc_auc":roc_auc_score(y_test,proba),
            "minority_recall":recall_score(y_test,pred,pos_label=1,zero_division=0),
            "false_positive_rate":fp/(fp+tn) if fp+tn else 0,
            "false_negative_rate":fn/(fn+tp) if fn+tp else 0,
            "fit_seconds":fit_seconds,
            "cv_f1_mean":np.mean(cvres["test_f1"]),
            "cv_f1_std":np.std(cvres["test_f1"]),
            "cv_balanced_accuracy_mean":np.mean(cvres["test_balanced_accuracy"]),
        })
    results=pd.DataFrame(rows).sort_values("f1",ascending=False)
    results.to_csv(out/"tables/experiment_results.csv",index=False)

    # Main comparison plot.
    ax=results.set_index("strategy")[["f1","minority_recall","balanced_accuracy"]].plot(kind="bar",figsize=(11,6))
    ax.set_ylim(0,1.05); ax.set_ylabel("Score"); ax.set_title("Preprocessing and Imbalance Strategy Comparison")
    ax.grid(axis="y",alpha=.25); plt.tight_layout(); plt.savefig(out/"figures/strategy_comparison.png",dpi=180); plt.close()

    # Class distribution plot.
    dist["count"].plot(kind="bar",figsize=(8,5),title="Class Distribution")
    plt.ylabel("Samples"); plt.tight_layout(); plt.savefig(out/"figures/class_distribution.png",dpi=180); plt.close()

    return results
