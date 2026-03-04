import joblib
import pandas as pd
from config import MODEL_PATH, ENCODERS_PATH, FEATURES_PATH, THRESHOLD_PATH, CAT_COLS
from evaluator import assign_risk_band


class PaymentRiskScorer:

    def __init__(self):
        self.model     = joblib.load(MODEL_PATH)
        self.encoders  = joblib.load(ENCODERS_PATH)
        self.features  = joblib.load(FEATURES_PATH)
        self.threshold = joblib.load(THRESHOLD_PATH)

    def score(self, txn: dict) -> dict:
        df = pd.DataFrame([txn])

        for col in CAT_COLS:
            df[col + "_enc"] = self.encoders[col].transform(df[col].astype(str))

        prob  = self.model.predict_proba(df[self.features].fillna(0))[0][1]
        band  = assign_risk_band(prob, self.threshold)

        return {
            "risk_score"          : round(prob * 100, 2),
            "risk_band"           : band,
            "overdue_probability" : round(prob, 4),
        }

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in CAT_COLS:
            df[col + "_enc"] = self.encoders[col].transform(df[col].astype(str))

        probs = self.model.predict_proba(df[self.features].fillna(0))[:, 1]

        df["risk_score"] = (probs * 100).round(2)
        df["risk_band"]  = [assign_risk_band(p, self.threshold) for p in probs]

        return df


if __name__ == "__main__":
    scorer = PaymentRiskScorer()

    result = scorer.score({
        "discount_pct"        : 10,
        "total_value"         : 85000,
        "gross_margin"        : -5000,
        "quantity"            : 200,
        "price_to_cost_ratio" : 0.95,
        "is_negative_margin"  : 1,
        "discount_x_margin"   : 0.06,
        "margin_pct"          : -0.059,
        "channel"             : "General Trade",
        "region"              : "Delhi",
        "category"            : "Beverages",
        "transaction_type"    : "SALE",
        "vendor_class"        : 2,
        "rolling_avg_delay"   : 35,
        "rolling_overdue_rate": 0.4,
        "payment_volatility"  : 12,
        "dist_txn_count"      : 20,
        "dist_avg_value"      : 70000,
        "fulfillment_rate_pct": 85.0,
        "is_partial_delivery" : 1,
        "qty_gap"             : 15,
        "transaction_value"   : 85000,
    })

    print(f"Risk Score : {result['risk_score']}%")
    print(f"Risk Band  : {result['risk_band']}")
