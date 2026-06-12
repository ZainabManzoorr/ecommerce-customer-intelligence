RetailPulse-AI: Ecommerce Customer Intelligence System
Overview

RetailPulse-AI is an end-to-end customer analytics and segmentation system designed to analyze ecommerce transaction data and extract actionable business insights. The project applies machine learning (K-Means clustering) on RFM-based features to segment customers and visualize behavioral patterns through an interactive dashboard.

Objectives
Understand customer purchasing behavior using RFM analysis
Segment customers into meaningful groups using unsupervised learning
Identify high-value, regular, and at-risk customers
Provide interactive visual insights for business decision-making
 Key Features
RFM feature engineering (Recency, Frequency, Monetary)
K-Means clustering for customer segmentation
Model evaluation using Silhouette Score and Inertia
Cluster profiling for business interpretation
Interactive Streamlit dashboard for exploration
Visual analytics (distribution, scatter plots, KPIs)
Tech Stack
Python 
Pandas & NumPy
Scikit-learn (KMeans, preprocessing, metrics)
Streamlit (dashboard UI)
Plotly (interactive visualizations)
Business Insights

The model identifies:

 High-value (VIP) customers contributing maximum revenue
 Regular customers with stable engagement
 At-risk / low-engagement customers requiring retention strategies
 How to Run Locally
# Clone repository
git clone https://github.com/your-username/RetailPulse-AI.git

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

OR (if using Poetry)

poetry install
poetry run streamlit run app.py
Dashboard Preview

The Streamlit dashboard includes:

Customer segment explorer
KPI metrics (Recency, Frequency, Monetary)
Cluster distribution charts
Behavioral scatter plots
Downloadable segmented dataset