# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Retail Analytics Dashboard")

# -------------------------
# Helper / loader
# -------------------------
@st.cache_data
def ensure_columns(df):
    # Basic normalization of column names
    df = df.copy()
    df.columns = df.columns.str.strip()
    # required columns mapping tolerance
    if "TotalRevenue" not in df.columns:
        if "Price" in df.columns and "Quantity" in df.columns:
            df["TotalRevenue"] = df["Price"] * df["Quantity"]
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df

def prepare_monthly(data):
    df = data.copy()
    if "InvoiceMonth" in df.columns:
        ms = df.groupby("InvoiceMonth")["TotalRevenue"].sum().reset_index().rename(columns={"InvoiceMonth":"ds","TotalRevenue":"y"})
    else:
        ms = df.groupby(pd.Grouper(key="InvoiceDate", freq="M"))["TotalRevenue"].sum().reset_index().rename(columns={"InvoiceDate":"ds","TotalRevenue":"y"})
    ms = ms.sort_values("ds")
    return ms

# -------------------------
# Load / accept data
# -------------------------
st.sidebar.header("Data input")
st.sidebar.info("If you already have `data` in memory, the app will attempt to use it. Otherwise upload a CSV/Excel export.")

uploaded = st.sidebar.file_uploader("Upload transactions (csv/xlsx)", type=["csv","xlsx"])
if uploaded is not None:
    if uploaded.name.endswith(".csv"):
        data = pd.read_csv(uploaded)
    else:
        data = pd.read_excel(uploaded)
    st.sidebar.success("File loaded.")
else:
    # try to get 'data' from environment (for advanced users who run within the same session)
    try:
        # when running in same environment, user can set `data` variable in Python
        data = globals().get("data", None)
        if data is None:
            raise Exception()
        st.sidebar.success("Using data variable from environment.")
    except Exception:
        st.sidebar.warning("No data loaded. Upload a file to proceed.")
        st.stop()

# normalize
data = ensure_columns(data)

# Prepare derived columns if not present
if "InvoiceMonth" not in data.columns:
    # ensure InvoiceDate is datetime first
    if "InvoiceDate" in data.columns:
        data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
        data["InvoiceMonth"] = data["InvoiceDate"].dt.to_period("M").astype(str)
    else:
        data["InvoiceMonth"] = np.nan

if "TotalRevenue" not in data.columns:
    # try Price * Quantity (fallback)
    if "Quantity" in data.columns and "Price" in data.columns:
        data["TotalRevenue"] = data["Quantity"] * data["Price"]
    else:
        data["TotalRevenue"] = 0.0

# Prepare monthly_sales
monthly_sales = prepare_monthly(data)

# product_metrics fallback
product_metrics = globals().get("product_metrics", None)
if product_metrics is None:
    pm = (data.groupby(["StockCode","Description"], dropna=False)
            .agg(Total_Revenue=("TotalRevenue","sum"),
                 Quantity_Sold=("Quantity","sum"),
                 Num_Orders=("Invoice","nunique"))
            .reset_index())
    pm["Return_Rate"] = 0.0
    pm["Margin"] = pm["Total_Revenue"] * 0.30
    product_metrics = pm

# rfm fallback
rfm = globals().get("rfm", None)
if rfm is None:
    latest_date = data["InvoiceDate"].max() + pd.Timedelta(days=1) if "InvoiceDate" in data.columns else pd.Timestamp.now()
    rfm = (data.groupby("Customer ID")
           .agg(Recency=("InvoiceDate", lambda x: (latest_date - x.max()).days if not x.isna().all() else np.nan),
                Frequency=("Invoice", "nunique"),
                Monetary=("TotalRevenue","sum"))
           .reset_index())
    # simple segmentation via quartiles
    r_labels = [4,3,2,1]
    try:
        rfm["R_Score"] = pd.qcut(rfm["Recency"].fillna(rfm["Recency"].max()), 4, labels=r_labels).astype(int)
    except Exception:
        rfm["R_Score"] = 1
    try:
        rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first").fillna(0), 4, labels=[1,2,3,4]).astype(int)
    except Exception:
        rfm["F_Score"] = 1
    try:
        rfm["M_Score"] = pd.qcut(rfm["Monetary"].fillna(0), 4, labels=[1,2,3,4]).astype(int)
    except Exception:
        rfm["M_Score"] = 1
    rfm["RFM"] = rfm["R_Score"].astype(str)+rfm["F_Score"].astype(str)+rfm["M_Score"].astype(str)

# forecast fallback (if user doesn't have)
forecast = globals().get("forecast", None)
if forecast is None:
    # naive forecast: last value repeated
    test_len = max(3, int(len(monthly_sales)*0.15))
    forecast = monthly_sales.copy()
    if "y" in forecast.columns:
        forecast["yhat"] = forecast["y"].shift(1).fillna(method="bfill")
    else:
        forecast["yhat"] = 0.0
    # extend forecast horizon by test_len with last known value
    if len(monthly_sales) > 0:
        last_ds = monthly_sales["ds"].max()
        try:
            future_dates = pd.date_range(start=pd.to_datetime(last_ds) + pd.offsets.MonthBegin(), periods=test_len, freq='M')
        except Exception:
            future_dates = pd.date_range(start=pd.Timestamp.today(), periods=test_len, freq='M')
        future_df = pd.DataFrame({"ds": future_dates, "y": np.nan, "yhat": monthly_sales["y"].iloc[-1] if "y" in monthly_sales.columns else 0})
        forecast = pd.concat([forecast, future_df], ignore_index=True)

# -------------------------
# PAGE NAVIGATION
# -------------------------
pages = ["Executive Summary", "Revenue Analysis", "Customer Insights", "Forecasting"]
page = st.sidebar.selectbox("Select page", pages)

# -------------------------
# EXECUTIVE SUMMARY PAGE
# -------------------------
if page == "Executive Summary":
    st.title("Retail Analytics — Executive Dashboard")
    st.markdown("**Executive summary & interactive exploration**")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_sales = data["TotalRevenue"].sum()
    total_orders = data["Invoice"].nunique()
    total_customers = data["Customer ID"].nunique()
    aov = total_sales / (total_orders if total_orders>0 else 1)
    avg_margin_pct = (product_metrics["Margin"].sum() / total_sales * 100) if total_sales>0 else 0

    kpi1.metric("Total Sales (£)", f"{total_sales:,.0f}")
    kpi2.metric("Total Orders", f"{total_orders:,}")
    kpi3.metric("Total Customers", f"{total_customers:,}")
    kpi4.metric("AOV (£)", f"{aov:,.2f}")

    st.markdown(f"**Avg Margin (est.):** {avg_margin_pct:.2f}% (product-level margin estimate)")

    # -------------------------
    # Revenue & Forecast Visualization
    # -------------------------
    st.subheader("Revenue & Forecast")
    col1, col2 = st.columns([3,1])

    # ensure monthly_sales ds is datetime for plotting
    ms_for_plot = monthly_sales.copy()
    if "ds" in ms_for_plot.columns and ms_for_plot["ds"].dtype == object:
        try:
            ms_for_plot["ds"] = pd.to_datetime(ms_for_plot["ds"])
        except Exception:
            pass

    with col1:
        fig = go.Figure()
        if "ds" in ms_for_plot.columns:
            fig.add_trace(go.Scatter(x=ms_for_plot["ds"], y=ms_for_plot["y"], mode="lines+markers", name="Actual"))
        else:
            fig.add_trace(go.Scatter(x=list(range(len(ms_for_plot))), y=ms_for_plot["y"], mode="lines+markers", name="Actual"))
        # overlay forecast yhat if available
        fc = forecast.copy()
        if "ds" in fc.columns and "yhat" in fc.columns:
            if fc["ds"].dtype == object:
                try:
                    fc["ds"] = pd.to_datetime(fc["ds"])
                except Exception:
                    pass
            fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines+markers", name="Forecast"))
        fig.update_layout(title="Monthly Revenue (Actual vs Forecast)", xaxis_title="Date", yaxis_title="Revenue (£)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("Quick stats")
        st.write({"Months": len(monthly_sales), "Latest Month Revenue": float(monthly_sales['y'].iloc[-1]) if len(monthly_sales)>0 else 0})

    # -------------------------
    # Product & Region Analysis
    # -------------------------
    st.subheader("Top Products & Region Contribution")
    pcol1, pcol2 = st.columns(2)

    with pcol1:
        top_products = product_metrics.sort_values("Total_Revenue", ascending=False).head(15)
        y_col = "Description" if "Description" in top_products.columns else "StockCode"
        figp = px.bar(top_products, x="Total_Revenue", y=y_col,
                      orientation='h', title="Top products by revenue")
        figp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(figp, use_container_width=True)

    with pcol2:
        if "Country" in data.columns:
            country_rev = data.groupby("Country")["TotalRevenue"].sum().reset_index().sort_values("TotalRevenue", ascending=False)
            figc = px.bar(country_rev.head(15), x="TotalRevenue", y="Country", orientation='h', title="Top countries by revenue")
            figc.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(figc, use_container_width=True)
        else:
            st.info("Country column not found in data.")

    # -------------------------
    # Customer & Marketing Insights (RFM)
    # -------------------------
    st.subheader("Customer RFM & Segments")
    rcol1, rcol2 = st.columns([2,1])

    with rcol1:
        rfm_sample = rfm.copy()
        # show RFM table (top 50 by Monetary)
        if "Monetary" in rfm_sample.columns:
            st.dataframe(rfm_sample.sort_values("Monetary", ascending=False).head(50))
        else:
            st.dataframe(rfm_sample.head(50))

    with rcol2:
        seg_counts = rfm["RFM"].value_counts().sort_index()
        fig_seg = px.bar(x=seg_counts.index, y=seg_counts.values, labels={'x':'RFM', 'y':'Count'}, title="RFM segment counts")
        st.plotly_chart(fig_seg, use_container_width=True)

    # ROAS placeholder (needs ad-spend data)
    st.markdown("**ROAS / Marketing effectiveness** — upload campaign spend data to compute ROAS.")

    # -------------------------
    # Cohort retention heatmap (diagnostic)
    # -------------------------
    st.subheader("Cohort Retention")
    if "InvoiceDate" in data.columns and "Customer ID" in data.columns:
        data["CohortMonth"] = data.groupby("Customer ID")["InvoiceDate"].transform("min").dt.to_period("M")
        data["InvoiceMonthPeriod"] = data["InvoiceDate"].dt.to_period("M")
        cohort_counts = (data.groupby(["CohortMonth","InvoiceMonthPeriod"])["Customer ID"].nunique().reset_index())
        if not cohort_counts.empty:
            cohort_pivot = cohort_counts.pivot(index="CohortMonth", columns="InvoiceMonthPeriod", values="Customer ID").fillna(0)
            if not cohort_pivot.empty:
                cohort_size = cohort_pivot.iloc[:,0]
                retention = cohort_pivot.divide(cohort_size, axis=0).fillna(0) * 100
                fig_heat = px.imshow(retention.values, labels=dict(x="Months since cohort start", y="Cohort month", color="Retention %"),
                                      x=[str(x) for x in retention.columns], y=[str(x) for x in retention.index], color_continuous_scale="Blues")
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Not enough data for cohort analysis.")
        else:
            st.info("Not enough data for cohort analysis.")
    else:
        st.info("InvoiceDate or Customer ID column missing; cohort analysis not available.")

    # -------------------------
    # Download prepared data
    # -------------------------
    st.subheader("Download data")
    st.download_button("Download monthly sales (CSV)", monthly_sales.to_csv(index=False).encode('utf-8'), file_name="monthly_sales.csv")
    st.download_button("Download product metrics (CSV)", product_metrics.to_csv(index=False).encode('utf-8'), file_name="product_metrics.csv")
    st.download_button("Download RFM table (CSV)", rfm.to_csv(index=False).encode('utf-8'), file_name="rfm.csv")

    st.markdown("### Notes")
    st.markdown("""
    - This app expects a transactions table with `InvoiceDate`, `Quantity`, `Price`, and `Customer ID`.
    - ROAS requires campaign-level spend & impressions (not included).
    - For forecast, provide a `forecast` DataFrame with columns `ds`, `y`, `yhat` in the environment, or upload one.
    """)

# -------------------------
# REVENUE ANALYSIS PAGE (placeholder)
# -------------------------
elif page == "Revenue Analysis":
    st.title("Revenue Analysis")
    st.markdown("This page will contain filters and deeper revenue charts (time, category, region).")
    st.info("Next step: add date range, country and product filters + multi-chart layout. Tell me to continue and I'll add it.")

# -------------------------
# CUSTOMER INSIGHTS PAGE (placeholder)
# -------------------------
elif page == "Customer Insights":
    st.title("Customer Insights")
    st.markdown("This page will contain RFM visualizations, churn predictions and customer lists.")
    st.info("Next step: add interactive RFM heatmap, segment filters and churn model outputs. Tell me to continue and I'll add it.")

# -------------------------
# FORECASTING PAGE (placeholder)
# -------------------------
elif page == "Forecasting":
    st.title("Forecasting")
    st.markdown("This page will show forecast plots (ARIMA / Prophet) and evaluation metrics.")
    st.info("Next step: add model selection controls and evaluation table. Tell me to continue and I'll add it.")
