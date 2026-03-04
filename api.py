import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

API_KEY        = os.getenv("API_KEY", "change-this-before-deploy")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(title="Payment Risk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model     = joblib.load("artifacts/payment_risk_model.pkl")
encoders  = joblib.load("artifacts/label_encoders.pkl")
features  = joblib.load("artifacts/feature_list.pkl")
threshold = joblib.load("artifacts/threshold.pkl")


def verify_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return key


def assign_risk_band(score: float) -> str:
    if score >= 0.75:        return "Very High"
    elif score >= 0.50:      return "High"
    elif score >= threshold: return "Medium"
    else:                    return "Low"


class Transaction(BaseModel):
    discount_pct:          float
    total_value:           float
    gross_margin:          float
    quantity:              int
    price_to_cost_ratio:   float
    is_negative_margin:    int
    discount_x_margin:     float
    margin_pct:            float
    channel:               str
    region:                str
    category:              str
    transaction_type:      str
    vendor_class:          int
    rolling_avg_delay:     float
    rolling_overdue_rate:  float
    payment_volatility:    float
    dist_txn_count:        int
    dist_avg_value:        float
    fulfillment_rate_pct:  float
    is_partial_delivery:   int
    qty_gap:               float
    transaction_value:     float


@app.get("/health")
def health():
    return {"status": "ok", "model": "CatBoost", "version": "1.0.0"}


@app.post("/score")
def score(txn: Transaction, key: str = Security(verify_key)):
    df = pd.DataFrame([txn.model_dump()])

    for col in ["channel", "region", "category", "transaction_type"]:
        df[col + "_enc"] = encoders[col].transform(df[col].astype(str))

    prob  = model.predict_proba(df[features].fillna(0))[0][1]
    band  = assign_risk_band(prob)

    return {
        "risk_score":          round(prob * 100, 2),
        "risk_band":           band,
        "overdue_probability": round(prob, 4),
        "threshold_used":      round(threshold, 3),
        "disclaimer":          "Model catches ~54% of overdue cases. Always manually review Low band edge cases.",
    }


@app.post("/score/batch")
def score_batch(transactions: list[Transaction], key: str = Security(verify_key)):
    if len(transactions) > 500:
        raise HTTPException(status_code=400, detail="Max 500 transactions per batch")

    df = pd.DataFrame([t.model_dump() for t in transactions])

    for col in ["channel", "region", "category", "transaction_type"]:
        df[col + "_enc"] = encoders[col].transform(df[col].astype(str))

    probs = model.predict_proba(df[features].fillna(0))[:, 1]

    return [
        {
            "index":               i,
            "risk_score":          round(float(p) * 100, 2),
            "risk_band":           assign_risk_band(float(p)),
            "overdue_probability": round(float(p), 4),
        }
        for i, p in enumerate(probs)
    ]