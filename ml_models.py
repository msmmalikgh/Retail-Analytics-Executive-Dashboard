# modules/ml_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neighbors import NearestNeighbors


# -------------------------
# 1️⃣ Regression — Predict sales or profit
# -------------------------
def train_regression(data, target_col="TotalRevenue"):
    """Train a regression model to predict sales or profit."""
    numeric = data.select_dtypes(include=np.number).dropna()
    if target_col not in numeric.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    
    X = numeric.drop(columns=[target_col])
    y = numeric[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return model, {"RMSE": rmse, "R2": model.score(X_test_scaled, y_test)}


# -------------------------
# 2️⃣ Classification — Predict churn / re-order
# -------------------------
def train_classification(rfm):
    """Predict customer churn based on RFM features."""
    df = rfm.copy()
    df["Churn"] = (df["Recency"] > df["Recency"].median()).astype(int)  # high recency → churn
    
    X = df[["Recency", "Frequency", "Monetary"]]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    return model, {"Accuracy": acc}


# -------------------------
# 3️⃣ Clustering — Customer Segmentation (K-Means)
# -------------------------
def cluster_customers(rfm, n_clusters=4):
    """Segment customers into behavioral clusters using KMeans."""
    features = rfm[["Recency", "Frequency", "Monetary"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(scaled)
    return rfm, kmeans


# -------------------------
# 4️⃣ Recommendation System — Collaborative Filtering
# -------------------------
def build_recommendation(data):
    """Simple item-based recommendation using nearest neighbors."""
    pivot = data.pivot_table(index="Customer ID", columns="Description", values="Quantity", fill_value=0)
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(pivot.values)
    return pivot, model


def recommend_items(customer_id, pivot, model, top_n=5):
    """Recommend top N similar customers/products."""
    if customer_id not in pivot.index:
        return []
    idx = list(pivot.index).index(customer_id)
    distances, indices = model.kneighbors([pivot.iloc[idx]], n_neighbors=top_n + 1)
    similar_customers = [pivot.index[i] for i in indices.flatten()[1:]]
    return similar_customers
