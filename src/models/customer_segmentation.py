import pandas as pd


class CustomerSegmentLabeler:
    """
    Maps KMeans cluster IDs to business-friendly customer segments
    and generates segment-level analytics.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def assign_labels(self, mapping: dict):
        """
        Map cluster IDs to business segments.

        Example:
        {
            0: "Low Engagement",
            1: "Loyal Customers",
            2: "VIP Customers",
            3: "Regular Customers"
        }
        """

        # Validate clusters exist in mapping
        missing_clusters = set(self.df["Cluster"].unique()) - set(mapping.keys())
        if missing_clusters:
            raise ValueError(f"Missing mappings for clusters: {missing_clusters}")

        self.df["Segment"] = self.df["Cluster"].map(mapping)

        return self.df

    def segment_summary(self):
        """
        Returns aggregated business insights per segment.
        """

        required_cols = ["CustomerID", "Recency", "Frequency", "Monetary"]

        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        summary = (
            self.df.groupby("Segment")
            .agg(
                Customers=("CustomerID", "count"),
                AvgRecency=("Recency", "mean"),
                AvgFrequency=("Frequency", "mean"),
                AvgMonetary=("Monetary", "mean"),
            )
            .round(2)
            .sort_values("AvgMonetary", ascending=False)
        )

        return summary