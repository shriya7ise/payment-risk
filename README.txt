payment_risk/
│
├── config.py          # All constants, paths, feature list
├── data_loader.py     # Reads CSVs, validates columns
├── data_engineer.py   # Cleaning, merging, feature engineering
├── model_builder.py   # XGBoost, CatBoost, Stacking definitions
├── evaluator.py       # Metrics, threshold tuning, risk bands
├── trainer.py         # Orchestrates full training pipeline
├── scorer.py          # Inference on new transactions
│
├── data/
│   ├── financial_tracking.csv
│   └── sales_transactions.csv
│
└── artifacts/         # Auto-created on train
    ├── payment_risk_model.pkl
    ├── label_encoders.pkl
    ├── feature_list.pkl
    ├── threshold.pkl
    └── scored_transactions.csv


SETUP
-----
pip install catboost xgboost scikit-learn pandas numpy joblib

Put your CSVs in the data/ folder.


TRAIN
-----
python trainer.py


SCORE A NEW TRANSACTION
-----------------------
python scorer.py


SCORE IN YOUR OWN CODE
-----------------------
from scorer import PaymentRiskScorer

scorer = PaymentRiskScorer()
result = scorer.score({ ...transaction dict... })
print(result["risk_score"], result["risk_band"])


SCORE A BATCH CSV
-----------------
from scorer import PaymentRiskScorer
import pandas as pd

scorer = PaymentRiskScorer()
df     = pd.read_csv("new_transactions.csv")
output = scorer.score_batch(df)
output.to_csv("scored_output.csv", index=False)
