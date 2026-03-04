import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix, f1_score,
)


def evaluate(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_prob)
    pr  = average_precision_score(y_test, y_prob)

    print(f"\n[{name}]")
    print(f"  ROC-AUC : {roc:.4f}  |  PR-AUC : {pr:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"  Classification Report:\n{classification_report(y_test, y_pred)}")

    return {"name": name, "model": model, "roc_auc": roc, "pr_auc": pr}


def find_best_threshold(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    y_prob = model.predict_proba(X_test)[:, 1]
    best_t, best_f1 = 0.5, 0.0

    for t in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_test, (y_prob >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\n[Threshold] Optimal = {best_t:.3f}  |  F1 = {best_f1:.4f}")
    return float(best_t)


def assign_risk_band(score: float, threshold: float) -> str:
    if score >= 0.75:        return "Very High"
    elif score >= 0.50:      return "High"
    elif score >= threshold: return "Medium"
    else:                    return "Low"


def risk_band_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("risk_band")
        .agg(
            count          = ("overdue_flag", "count"),
            actual_overdue = ("overdue_flag", "sum"),
            overdue_rate   = ("overdue_flag", "mean"),
            avg_score      = ("risk_score", "mean"),
        )
        .round(4)
    )
