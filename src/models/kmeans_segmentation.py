import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class CustomerSegmentationKMeans:
    """
    ML-based customer segmentation using KMeans clustering.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self.scaler = StandardScaler()

    # -----------------------------
    # 1. FEATURE SELECTION
    # -----------------------------
    def prepare_features(self):

        features = self.df.drop(columns=["CustomerID"])

        self.feature_columns = features.columns

        logger.info(f"Features used for clustering: {list(self.feature_columns)}")

        self.X = features
        return self

    # -----------------------------
    # 2. SCALING
    # -----------------------------
    def scale_features(self):

        self.X_scaled = self.scaler.fit_transform(self.X)

        logger.info("Feature scaling completed")

        return self

    # -----------------------------
    # 3. TRAIN KMEANS
    # -----------------------------
    def train_kmeans(self, n_clusters=4, random_state=42):

        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

        self.df["Cluster"] = self.model.fit_predict(self.X_scaled)

        logger.info(f"KMeans trained with {n_clusters} clusters")

        return self

    # -----------------------------
    # 4. CLUSTER ANALYSIS
    # -----------------------------
    def analyze_clusters(self):

        summary = self.df.groupby("Cluster").agg({
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean",
            "CustomerID": "count"
        }).rename(columns={"CustomerID": "NumCustomers"})

        self.cluster_summary = summary

        logger.info("Cluster profiling completed")

        return summary

    # -----------------------------
    # OUTPUT
    # -----------------------------
    def get_segmented_data(self):

        return self.df