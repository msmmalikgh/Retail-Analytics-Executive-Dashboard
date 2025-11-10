# 🛍️ Retail Analytics — Executive Dashboard

A complete data analysis and visualization project built for **Online Retail II dataset**.  
It transforms raw transactional data into powerful business insights through interactive dashboards and automated reports.

---

## 🚀 Features

### 🧹 Data Cleaning & Preparation
- Combines multiple yearly sheets (2009–2011)
- Removes duplicates, missing values, and invalid entries
- Computes **Total Revenue = Quantity × Price**
- Extracts Year, Month, and Invoice Period

### 📊 Sales & Revenue Analytics
- **Monthly Sales Trend** — visualize seasonality and growth patterns  
- **Top Products** — identify highest revenue-generating items  
- **Revenue by Country** — discover strong and weak markets  
- **Pareto (80/20) Analysis** — reveal how few customers drive most revenue  

### 👥 Customer Segmentation (RFM Model)
- Recency, Frequency, Monetary scoring for each customer  
- Auto-segmentation into: **VIP, Loyal, Regular, At Risk, Lost**  
- Visual segment breakdown (bar + pie chart)

### 📈 Executive KPI Overview
- Key performance indicators (KPIs):  
  - 💰 Total Revenue  
  - 👥 Unique Customers  
  - 🧾 Total Invoices  

### 🧾 Automated Report Generation
- One-click **PowerPoint (PPTX)** executive summary generation  
- Includes KPIs, Pareto stats, and segment insights  

### 🧠 Optional Add-on (Advanced)
- RFM segmentation export to Excel or CSV  
- Customer-level profile insights (Recency, Frequency, Revenue)

---

## 🛠️ Tech Stack

| Tool / Library | Purpose |
|----------------|----------|
| **Python (3.10+)** | Core analysis |
| **Pandas, NumPy** | Data manipulation |
| **Matplotlib, Seaborn** | Visualization |
| **Scikit-learn** | Clustering & scoring |
| **Streamlit** | Interactive dashboard |
| **Plotly** | Interactive charts |
| **OpenPyXL, python-pptx** | Excel & PowerPoint export |

---

## 🧩 Project Workflow

1. **Load Data** → Read and merge yearly retail sales data  
2. **Clean Data** → Handle duplicates, nulls, invalid prices  
3. **Feature Engineering** → Add `TotalRevenue`, `InvoiceMonth`  
4. **EDA** → Visualize trends, top products, and geography  
5. **Customer Segmentation** → Apply RFM analysis  
6. **Reporting** → Generate visual dashboard + PPT report  

---

## 📂 File Structure

