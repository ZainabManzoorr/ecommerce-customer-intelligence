import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Converts cleaned ecommerce transactions into customer intelligence features.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # -----------------------------
    # 1. RFM FEATURES (CORE)
    # -----------------------------
    def create_rfm(self):

        snapshot_date = self.df["InvoiceDate"].max()

        rfm = self.df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
            "InvoiceNo": "nunique",
            "Revenue": "sum"
        }).reset_index()

        rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

        logger.info(f"RFM table created: {rfm.shape}")

        self.rfm = rfm
        return self

    # -----------------------------
    # 2. BASKET FEATURES
    # -----------------------------
    def create_basket_features(self):

        basket = self.df.groupby("CustomerID").agg({
            "Quantity": "sum",
            "Revenue": "mean",
            "StockCode": "nunique"
        }).reset_index()

        basket.columns = [
            "CustomerID",
            "TotalItems",
            "AvgRevenuePerTransaction",
            "UniqueProducts"
        ]

        logger.info(f"Basket features created: {basket.shape}")

        self.basket = basket
        return self

    # -----------------------------
    # 3. TIME-BASED FEATURES
    # -----------------------------
    def create_time_features(self):

        customer_time = self.df.groupby("CustomerID").agg({
            "InvoiceDate": [
                lambda x: (x.max() - x.min()).days,  # lifespan
                "count"  # total transactions
            ]
        })

        customer_time.columns = ["LifespanDays", "TotalTransactions"]
        customer_time = customer_time.reset_index()

        logger.info(f"Time features created: {customer_time.shape}")

        self.time_features = customer_time
        return self

    # -----------------------------
    # 4. FINAL MERGE
    # -----------------------------
    def build_feature_table(self):

        df = self.rfm.merge(self.basket, on="CustomerID", how="left")
        df = df.merge(self.time_features, on="CustomerID", how="left")

        self.features = df

        logger.info(f"Final feature table shape: {df.shape}")

        return df

    # -----------------------------
    # OUTPUT
    # -----------------------------
    def get_features(self):
        return self.features