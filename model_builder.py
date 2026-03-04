from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from config import RANDOM_STATE, CV_FOLDS


def get_xgboost(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators     = 500,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale_pos_weight,
        eval_metric      = "auc",
        random_state     = RANDOM_STATE,
        verbosity        = 0,
    )


def get_catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations         = 500,
        depth              = 6,
        learning_rate      = 0.05,
        auto_class_weights = "Balanced",
        eval_metric        = "AUC",
        random_seed        = RANDOM_STATE,
        verbose            = 0,
    )


def get_stacking(scale_pos_weight: float) -> StackingClassifier:
    return StackingClassifier(
        estimators      = [
            ("xgb", get_xgboost(scale_pos_weight)),
            ("cat", get_catboost()),
        ],
        final_estimator = LogisticRegression(max_iter=1000, C=0.1),
        cv              = CV_FOLDS,
        passthrough     = False,
    )


def get_all_models(scale_pos_weight: float) -> dict:
    return {
        "XGBoost"  : get_xgboost(scale_pos_weight),
        "CatBoost" : get_catboost(),
        "Stacking" : get_stacking(scale_pos_weight),
    }
