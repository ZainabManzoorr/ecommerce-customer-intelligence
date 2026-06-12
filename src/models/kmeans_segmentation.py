import logging
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CustomerSegmentationKMeans:
    """
    ML-based customer segmentation using KMeans clustering.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.model = None

    # -----------------------------
    # 1. FEATURE PREPARATION
    # -----------------------------
    def prepare_features(self):

        # Remove ID column
        self.X = self.df.drop(columns=["CustomerID"]).copy()

        # Log transform skewed features
        skewed_cols = [
            "Monetary",
            "Frequency",
            "TotalItems",
            "TotalTransactions",
        ]

        for col in skewed_cols:
            if col in self.X.columns:
                self.X[col] = np.log1p(self.X[col])

        self.feature_columns = self.X.columns.tolist()

        logger.info(f"Using features: {self.feature_columns}")

        return self

    # -----------------------------
    # 2. SCALE FEATURES
    # -----------------------------
    def scale_features(self):

        self.X_scaled = self.scaler.fit_transform(self.X)

        logger.info("Feature scaling completed.")

        return self

    # -----------------------------
    # 3. TRAIN MODEL
    # -----------------------------
    def train_kmeans(self, n_clusters=4, random_state=42):

        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )

        self.df["Cluster"] = self.model.fit_predict(self.X_scaled)

        self.inertia = self.model.inertia_

        self.silhouette = silhouette_score(
            self.X_scaled,
            self.df["Cluster"]
        )

        logger.info(
            f"KMeans trained with {n_clusters} clusters "
            f"(Silhouette={self.silhouette:.3f})"
        )

        return self

    # -----------------------------
    # 4. CLUSTER PROFILE
    # -----------------------------
    def analyze_clusters(self):

        summary = (
            self.df
            .groupby("Cluster")
            .agg(
                Customers=("CustomerID", "count"),
                AvgRecency=("Recency", "mean"),
                AvgFrequency=("Frequency", "mean"),
                AvgMonetary=("Monetary", "mean"),
                AvgItems=("TotalItems", "mean"),
                AvgProducts=("UniqueProducts", "mean"),
                AvgLifespan=("LifespanDays", "mean"),
            )
            .round(2)
        )

        self.cluster_summary = summary

        logger.info("Cluster profiling completed.")

        return summary

    # -----------------------------
    # 5. MODEL METRICS
    # -----------------------------
    def get_metrics(self):

        return {
            "inertia": self.inertia,
            "silhouette_score": self.silhouette,
        }

    # -----------------------------
    # 6. OUTPUT
    # -----------------------------
    def get_segmented_data(self):

        return self.df