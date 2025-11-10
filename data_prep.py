# modules/data_prep.py
import pandas as pd

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and compute missing essentials."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Compute TotalRevenue if not present
    if "TotalRevenue" not in df.columns:
        if {"Price", "Quantity"}.issubset(df.columns):
            df["TotalRevenue"] = df["Price"] * df["Quantity"]

    # Parse InvoiceDate if possible
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    return df


def prepare_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly revenue for trend & forecast."""
    df = df.copy()
    if "InvoiceDate" not in df.columns:
        raise ValueError("InvoiceDate column required for monthly prep")

    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = (
        df.groupby("InvoiceMonth")["TotalRevenue"]
        .sum()
        .reset_index()
        .rename(columns={"InvoiceMonth": "ds", "TotalRevenue": "y"})
    )
    monthly = monthly.sort_values("ds").reset_index(drop=True)
    return monthly
