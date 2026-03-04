from pathlib import Path

BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

FT_PATH  = DATA_DIR / "financial_tracking (1).csv"
ST_PATH  = DATA_DIR / "sales_transactions (1).csv"

MODEL_PATH     = ARTIFACTS_DIR / "payment_risk_model.pkl"
ENCODERS_PATH  = ARTIFACTS_DIR / "label_encoders.pkl"
FEATURES_PATH  = ARTIFACTS_DIR / "feature_list.pkl"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.pkl"
SCORES_PATH    = ARTIFACTS_DIR / "scored_transactions.csv"

OVERDUE_DAYS   = 60
TEST_SIZE      = 0.2
CV_FOLDS       = 5
RANDOM_STATE   = 42

CAT_COLS = ["channel", "region", "category", "transaction_type"]

FEATURES = [
    "discount_pct", "total_value", "gross_margin", "quantity",
    "price_to_cost_ratio", "is_negative_margin", "discount_x_margin",
    "margin_pct", "channel_enc", "region_enc", "category_enc",
    "transaction_type_enc", "vendor_class",
    "rolling_avg_delay", "rolling_overdue_rate", "payment_volatility",
    "dist_txn_count", "dist_avg_value",
    "fulfillment_rate_pct", "is_partial_delivery", "qty_gap",
    "transaction_value",
]
