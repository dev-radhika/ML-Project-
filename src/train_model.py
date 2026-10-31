
from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "poi_3_0.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
TARGET = "Defect"
DROP_COLUMNS = ["ClassName"]
RANDOM_STATE = 42

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH)

    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")
    y = data[TARGET].astype(int)
    X = data.drop(columns=[TARGET] + [c for c in DROP_COLUMNS if c in data.columns])
    X = X.apply(pd.to_numeric, errors="coerce")
    if y.nunique() != 2:
        raise ValueError("This project expects a binary target with labels 0 and 1.")

    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    
    numeric_prep = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    scaled_prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    candidates = {
        "Logistic Regression": (
            Pipeline([("prep", scaled_prep),
                      ("model", LogisticRegression(max_iter=3000, class_weight="balanced",
                                                   random_state=RANDOM_STATE))]),
            {"model__C": [0.1, 1.0, 10.0]}
        ),
        "Decision Tree": (
            Pipeline([("prep", numeric_prep),
                      ("model", DecisionTreeClassifier(class_weight="balanced",
                                                       random_state=RANDOM_STATE))]),
            {"model__max_depth": [3, 5, None],
             "model__min_samples_leaf": [2, 5, 10]}
        ),
        "Random Forest": (
            Pipeline([("prep", numeric_prep),
                      ("model", RandomForestClassifier(class_weight="balanced",
                                                       random_state=RANDOM_STATE,
                                                       n_jobs=-1))]),
            {"model__n_estimators": [200, 400],
             "model__max_depth": [None, 8, 14],
             "model__min_samples_leaf": [1, 3, 5]}
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = []
    best_name, best_search, best_cv_f1 = None, None, -1
    for name, (pipeline, grid) in candidates.items():
        search = GridSearchCV(
            pipeline, grid, scoring="f1", cv=cv, n_jobs=-1,
            refit=True, return_train_score=False
        )
        search.fit(X_train, y_train)
        comparison.append({
            "model": name,
            "cv_f1_mean": float(search.best_score_),
            "cv_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "best_parameters": json.dumps(search.best_params_),
        })
        if search.best_score_ > best_cv_f1:
            best_name, best_search, best_cv_f1 = name, search, search.best_score_

   
    model = best_search.best_estimator_
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "selected_model": best_name,
        "selection_metric": "mean 5-fold training CV F1",
        "cv_f1_mean": float(best_cv_f1),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_rows": int(len(y_test)),
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "features": feature_names,
        "best_parameters": best_search.best_params_,
    }

    joblib.dump(model, MODEL_DIR / "defect_model.joblib")
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(comparison).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    report = classification_report(y_test, predictions, target_names=["Non-defective", "Defective"],
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / "classification_report.csv")

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-defective", "Defective"],
                yticklabels=["Non-defective", "Defective"])
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

   
    perm = permutation_importance(
        model, X_test, y_test, scoring="f1", n_repeats=20,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    print("\nTraining complete.")
    print(f"Selected model: {best_name}")
    print(f"Training CV F1: {best_cv_f1:.3f}")
    print("\nUntouched test-set metrics:")
    for key in ["test_accuracy", "test_balanced_accuracy", "test_precision",
                "test_recall", "test_f1", "test_roc_auc"]:
        print(f"{key}: {metrics[key]:.3f}")
    print(f"\nArtifacts saved in: {MODEL_DIR} and {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
