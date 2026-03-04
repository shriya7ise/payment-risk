import pandas as pd
from pathlib import Path
from config import FT_PATH, ST_PATH


def load_financial(path: Path = FT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_financial(df)
    return df


def load_sales(path: Path = ST_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_sales(df)
    return df


def _validate_financial(df: pd.DataFrame) -> None:
    required = {
        "transaction_id", "date", "sku_id", "transaction_type",
        "channel", "region", "quantity", "unit_cost", "selling_price",
        "total_value", "gross_margin", "discount_pct",
        "payment_status", "days_to_payment", "reference_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"financial_tracking missing columns: {missing}")


def _validate_sales(df: pd.DataFrame) -> None:
    required = {
        "order_id", "sku_id", "channel", "region", "distributor_id",
        "qty_ordered", "qty_delivered", "fulfillment_rate_pct",
        "is_partial_delivery", "unit_cost", "selling_price_per_unit",
        "discount_pct", "total_revenue", "gross_margin",
        "payment_status", "days_to_payment", "transaction_type",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sales_transactions missing columns: {missing}")
