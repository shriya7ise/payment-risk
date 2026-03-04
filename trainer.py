import joblib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from config import (
    FEATURES, TEST_SIZE, CV_FOLDS, RANDOM_STATE,
    MODEL_PATH, ENCODERS_PATH, FEATURES_PATH, THRESHOLD_PATH, SCORES_PATH,
    ARTIFACTS_DIR,
)
from data_loader import load_financial, load_sales
from data_engineer import run_pipeline
from model_builder import get_all_models
from evaluator import evaluate, find_best_threshold, assign_risk_band, risk_band_summary


def train():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PAYMENT RISK MODEL — TRAINING")
    print("=" * 60)

    ft_raw = load_financial()
    st_raw = load_sales()

    df, encoders = run_pipeline(ft_raw, st_raw)

    X = df[FEATURES].fillna(0)
    y = df["overdue_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        stratify     = y,
        random_state = RANDOM_STATE,
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = get_all_models(scale_pos_weight)

    results = []
    for name, model in models.items():
        print(f"\n[Training] {name}...")
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring = "roc_auc",
            n_jobs  = -1,
        )
        print(f"  CV-AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        results.append(evaluate(name, model, X_test, y_test))

    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\n[Best] → {best['name']}  (ROC-AUC: {best['roc_auc']:.4f})")

    threshold = find_best_threshold(best["model"], X_test, y_test)

    calibrated = CalibratedClassifierCV(best["model"], method="sigmoid", cv="prefit")
    calibrated.fit(X_test, y_test)
    print(f"[Calibrated] ROC-AUC: {roc_auc_score(y_test, calibrated.predict_proba(X_test)[:, 1]):.4f}")

    df["risk_score"] = calibrated.predict_proba(X.fillna(0))[:, 1] * 100
    df["risk_band"]  = df["risk_score"].apply(lambda s: assign_risk_band(s / 100, threshold))

    print("\n[Risk Band Summary]")
    print(risk_band_summary(df))

    joblib.dump(calibrated, MODEL_PATH)
    joblib.dump(encoders,   ENCODERS_PATH)
    joblib.dump(FEATURES,   FEATURES_PATH)
    joblib.dump(threshold,  THRESHOLD_PATH)
    df.to_csv(SCORES_PATH, index=False)

    print("\n[Saved artifacts to ./artifacts/]")


if __name__ == "__main__":
    train()
