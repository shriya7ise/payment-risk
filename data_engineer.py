import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from config import OVERDUE_DAYS, CAT_COLS, FEATURES


def build_target(ft: pd.DataFrame) -> pd.DataFrame:
    ft = ft[ft["transaction_type"] == "SALE"].copy()
    ft = ft[ft["quantity"] > 0].copy()

    ft["overdue_flag"] = (ft["days_to_payment"] > OVERDUE_DAYS).astype(int)
    ft.loc[ft["payment_status"] == "WRITTEN_OFF", "overdue_flag"] = 1
    ft.loc[
        (ft["payment_status"] == "PENDING") & (ft["days_to_payment"] > OVERDUE_DAYS),
        "overdue_flag"
    ] = 1

    ft["channel"] = ft["channel"].fillna("Unknown")
    ft["region"]  = ft["region"].fillna("Unknown")

    return ft


def build_distributor_features(st: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    st = st[st["transaction_type"] == "SALE"].copy()
    st = st.sort_values("order_date")

    st["is_overdue"]        = (st["days_to_payment"] > OVERDUE_DAYS).astype(int)
    st["transaction_value"] = st["selling_price_per_unit"] * st["qty_delivered"]

    dist = (
        st.groupby("distributor_id")
        .agg(
            rolling_avg_delay    = ("days_to_payment", "mean"),
            rolling_overdue_rate = ("is_overdue", "mean"),
            payment_volatility   = ("days_to_payment", "std"),
            dist_txn_count       = ("order_id", "count"),
            dist_avg_value       = ("transaction_value", "mean"),
        )
        .reset_index()
    )
    dist["payment_volatility"] = dist["payment_volatility"].fillna(0)

    return dist, st


def build_vendor_class(ft: pd.DataFrame) -> pd.DataFrame:
    vendor_stats = (
        ft.groupby("sku_id")["overdue_flag"]
        .mean()
        .reset_index()
        .rename(columns={"overdue_flag": "sku_overdue_rate"})
    )
    vendor_stats["vendor_class"] = pd.cut(
        vendor_stats["sku_overdue_rate"],
        bins=[-0.001, 0.1, 0.25, 0.5, 1.0],
        labels=[0, 1, 2, 3],
    ).astype(int)

    ft = ft.merge(vendor_stats[["sku_id", "vendor_class"]], on="sku_id", how="left")
    ft["vendor_class"] = ft["vendor_class"].fillna(0).astype(int)
    return ft


def merge_datasets(ft: pd.DataFrame, st_raw: pd.DataFrame) -> pd.DataFrame:
    dist_features, st = build_distributor_features(st_raw)

    merged = ft.merge(
        st[[
            "order_id", "distributor_id", "qty_ordered", "qty_delivered",
            "fulfillment_rate_pct", "is_partial_delivery", "transaction_value",
        ]],
        left_on  = "reference_id",
        right_on = "order_id",
        how      = "left",
    )

    merged = merged.merge(dist_features, on="distributor_id", how="left")
    merged = build_vendor_class(merged)

    return merged


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    df["margin_pct"]          = df["gross_margin"] / df["total_value"].replace(0, np.nan)
    df["price_to_cost_ratio"] = df["selling_price"] / df["unit_cost"].replace(0, np.nan)
    df["is_negative_margin"]  = (df["gross_margin"] < 0).astype(int)
    df["discount_x_margin"]   = df["discount_pct"] * df["margin_pct"].fillna(0)
    df["qty_gap"]             = df["qty_ordered"].fillna(0) - df["qty_delivered"].fillna(0)

    df["margin_pct"]          = df["margin_pct"].fillna(0)
    df["price_to_cost_ratio"] = df["price_to_cost_ratio"].fillna(1)
    df["fulfillment_rate_pct"]= df["fulfillment_rate_pct"].fillna(100)
    df["is_partial_delivery"] = df["is_partial_delivery"].fillna(False).astype(int)
    df["transaction_value"]   = df["transaction_value"].fillna(df["total_value"])

    for col in ["rolling_avg_delay", "rolling_overdue_rate", "payment_volatility",
                "dist_txn_count", "dist_avg_value"]:
        df[col] = df[col].fillna(df[col].median())

    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def run_pipeline(ft_raw: pd.DataFrame, st_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ft     = build_target(ft_raw)
    merged = merge_datasets(ft, st_raw)
    df, encoders = engineer_features(merged)

    print(f"[Pipeline] Shape    : {df.shape}")
    print(f"[Pipeline] Overdue  : {df['overdue_flag'].sum()} / {len(df)} ({df['overdue_flag'].mean()*100:.1f}%)")

    return df, encoders
