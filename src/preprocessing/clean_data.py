import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Cleans ecommerce transaction dataset for analytics + ML.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # -----------------------------
    # 1. Basic Structural Cleaning
    # -----------------------------
    def drop_missing_customers(self):
        before = self.df.shape[0]
        self.df = self.df.dropna(subset=["CustomerID"])
        logger.info(f"Dropped missing CustomerID rows: {before - self.df.shape[0]}")
        return self

    def remove_duplicates(self):
        before = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        logger.info(f"Removed duplicates: {before - self.df.shape[0]}")
        return self

    # -----------------------------
    # 2. Transaction Cleaning
    # -----------------------------
    def remove_invalid_transactions(self):
        """
        Removes:
        - Quantity <= 0
        - UnitPrice <= 0
        """
        before = self.df.shape[0]

        self.df = self.df[
            (self.df["Quantity"] > 0) &
            (self.df["UnitPrice"] > 0)
        ]

        logger.info(f"Removed invalid transactions: {before - self.df.shape[0]}")
        return self

    # -----------------------------
    # 3. Revenue Validation
    # -----------------------------
    def fix_revenue(self):
        """
        Recalculate Revenue to ensure correctness.
        """
        self.df["CalculatedRevenue"] = self.df["Quantity"] * self.df["UnitPrice"]

        mismatch = (self.df["Revenue"] - self.df["CalculatedRevenue"]).abs().sum()

        self.df["Revenue"] = self.df["CalculatedRevenue"]
        self.df.drop(columns=["CalculatedRevenue"], inplace=True)

        logger.info(f"Revenue corrected. Total mismatch sum was: {mismatch:.2f}")

        return self

    # -----------------------------
    # 4. Datetime Cleaning
    # -----------------------------
    def clean_datetime(self):
        self.df["InvoiceDate"] = pd.to_datetime(
            self.df["InvoiceDate"],
            errors="coerce"
        )

        before = self.df.shape[0]
        self.df = self.df.dropna(subset=["InvoiceDate"])

        logger.info(f"Removed invalid dates: {before - self.df.shape[0]}")
        return self

    # -----------------------------
    # 5. Outlier Handling
    # -----------------------------
    def remove_outliers(self, column="Revenue", quantile=0.99):
        """
        Removes extreme high-value transactions.
        """
        threshold = self.df[column].quantile(quantile)

        before = self.df.shape[0]
        self.df = self.df[self.df[column] <= threshold]

        logger.info(
            f"Removed outliers above {quantile} quantile: {before - self.df.shape[0]}"
        )

        return self

    # -----------------------------
    # 6. DATA AUDIT LAYER (IMPORTANT)
    # -----------------------------
    def data_audit(self):
        df = self.df

        report = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_customer_id": int(df["CustomerID"].isna().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "zero_price": int((df["UnitPrice"] <= 0).sum()),
            "zero_quantity": int((df["Quantity"] <= 0).sum()),
            "negative_revenue": int((df["Revenue"] < 0).sum()),
            "date_range": {
                "min": str(df["InvoiceDate"].min()),
                "max": str(df["InvoiceDate"].max())
            }
        }

        logger.info("Data audit completed successfully")
        return report

    # -----------------------------
    # FINAL OUTPUT
    # -----------------------------
    def get_clean_data(self):
        return self.df