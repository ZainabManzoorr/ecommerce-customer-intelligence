Project Objective

The goal of this project is to identify distinct customer groups using clustering algorithms so businesses can:

Improve marketing ROI

Increase customer retention

Identify VIP customers

Reactivate churned customers

Optimize promotional campaigns

This project uses KMeans clustering on multiple customer value metrics (RFM + CLV) to find actionable customer segments.

a) Dataset Description:

This dataset is a cleaned and feature-engineered e-commerce dataset suitable for machine learning and analytics. It contains transaction-level, customer-level, and product-level features.

b) Columns & Features:

Transaction-level: Revenue, IsReturn, InvoiceDate features (Year, Month, Hour, Weekday, etc.)

Customer-level: Recency, Frequency, Monetary, CLV, AverageBasketSize

Product-level: ProductRevenue, ProductQuantity, ReturnRate

c) Potential Use-Cases:

Customer segmentation & marketing analysis

CLV prediction & churn prediction

Product recommendation systems

Sales forecasting & trend analysis

d) Instructions to Use:

Load CSV files into pandas

Merge using CustomerID or StockCode

Use ML_dataset.csv directly for ML

Customer Segmentation Using RFM & CLV

A machine learning project that segments customers based on Recency, Frequency, Monetary (RFM) metrics and Customer Lifetime Value (CLV) to help businesses understand user behavior and define targeted marketing strategy.

From this raw data, we derived:

Feature	Description
Recency	Days since the last purchase
Frequency	Number of purchases
Monetary	Total spend
CLV	Predicted long-term revenue from a customer

Feature Engineering
1️.Recency

Calculated as:

Recency = (Latest Date in Dataset - Customer's Last Purchase Date)

2️. Frequency
Frequency = Number of unique purchase invoices

3️. Monetary
Monetary = Total spend = Σ(Quantity × UnitPrice)

4️. Customer Lifetime Value (CLV)

Formula used:

CLV = (Average Order Value × Purchase Frequency × Gross Margin)


Data Preprocessing

✔ Removed missing IDs
✔ Removed negative quantities (returns)
✔ Handled duplicates
✔ Converted dates
✔ Normalized features using StandardScaler (KMeans requirement)

Clustering Method
Why KMeans?

Works well with normalized numeric features

Fast and scalable

Perfect for behavioral segmentation

Choosing Number of Clusters

We used:

Elbow Method

Silhouette Score

Optimal K = 4 clusters.

Final Customer Segments
1. Segment 0 — Mid-Value Regular Customers

Recency: ~41 days

Frequency: moderate

Monetary: moderate

CLV: medium
They buy routinely. Good for loyalty campaigns.

2. Segment 1 — VIP Big Spenders

Rare purchases

Extremely high spending

Massive CLV
High-value premium customers. Maintain with VIP incentives.

3. Segment 2 — Low-Value / Near-Churn Customers

Very old recency

Very few purchases

Low monetary value
These are almost churned. Need reactivation offers.

4. Segment 3 — Loyal Super Buyers

Very recent recency (~6 days)

Extremely high frequency

High total spend

High CLV
Core customers. Keep engaged and upsell.

Visualizations
![Scatterplot for Customer Segmentation](image-1.png)
![Predicted CLV VS Actual CLV](image-2.png)
![Count Plot of Customer Segments](image-3.png)

Business Insights
1. Improve Retention

Target Segment 2 (near-churn) with:

discount coupons

retargeted ads

reminder emails

2. Protect High-Value Customers

Segment 1 (VIP) and Segment 3 (Loyal Super Buyers) should receive:

priority support

loyalty program

premium offers

3. Increase ROI

Allocate marketing budget based on CLV scores instead of random ads.

Tech Stack

Python

Pandas

NumPy

Matplotlib / Seaborn

Scikit-learn

How to Run
# 1. Install dependencies
poetry install

# 2. (Optional) Activate shell
poetry shell

# 3. Run the project
poetry run python customer-segmentation.py


