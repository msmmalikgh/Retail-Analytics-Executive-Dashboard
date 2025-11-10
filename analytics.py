# modules/analytics.py
import pandas as pd
import numpy as np

def compute_kpis(data: pd.DataFrame) -> dict:
    """Compute executive KPIs."""
    total_sales = data["TotalRevenue"].sum()
    total_orders = data["Invoice"].nunique() if "Invoice" in data.columns else np.nan
    total_customers = data["Customer ID"].nunique() if "Customer ID" in data.columns else np.nan
    aov = total_sales / total_orders if total_orders > 0 else 0

    return {
        "total_sales": total_sales,
        "total_orders": total_orders,
        "total_customers": total_customers,
        "aov": aov,
    }


def compute_rfm(data: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, and Monetary (RFM) metrics."""
    data = data.copy()
    if "InvoiceDate" not in data.columns:
        raise ValueError("InvoiceDate column missing for RFM calculation.")

    latest_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        data.groupby("Customer ID")
        .agg(
            Recency=("InvoiceDate", lambda x: (latest_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalRevenue", "sum"),
        )
        .reset_index()
    )

    # Quartile segmentation
    r_labels = [4, 3, 2, 1]
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 4, labels=r_labels).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["RFM"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    return rfm


def compute_cohort(data: pd.DataFrame) -> pd.DataFrame:
    """Create cohort retention matrix."""
    data = data.copy()
    data["CohortMonth"] = data.groupby("Customer ID")["InvoiceDate"].transform("min").dt.to_period("M")
    data["InvoiceMonth"] = data["InvoiceDate"].dt.to_period("M")

    cohort_counts = (
        data.groupby(["CohortMonth", "InvoiceMonth"])["Customer ID"].nunique().reset_index()
    )
    cohort_pivot = cohort_counts.pivot(index="CohortMonth", columns="InvoiceMonth", values="Customer ID").fillna(0)

    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0).fillna(0) * 100
    return retention
