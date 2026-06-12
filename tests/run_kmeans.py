from src.data_ingestion.load_data import load_dataset
from src.preprocessing.clean_data import DataCleaner
from src.features.features import FeatureEngineer
from src.models.kmeans_segmentation import CustomerSegmentationKMeans

# Load
df = load_dataset("data/raw/Transactions.csv")

# Clean
clean_df = (
    DataCleaner(df)
    .drop_missing_customers()
    .remove_duplicates()
    .remove_invalid_transactions()
    .fix_revenue()
    .clean_datetime()
    .remove_outliers()
    .get_clean_data()
)

# Feature Engineering
feature_df = (
    FeatureEngineer(clean_df)
    .create_rfm()
    .create_basket_features()
    .create_time_features()
    .build_feature_table()
)

# KMeans
segmenter = CustomerSegmentationKMeans(feature_df)

segmented_df = (
    segmenter
    .prepare_features()
    .scale_features()
    .train_kmeans(n_clusters=4)
    .get_segmented_data()
)

print(segmented_df.head())

print("\nCluster Counts:")
print(segmented_df["Cluster"].value_counts())

print("\nCluster Profile:")
print(segmenter.analyze_clusters())

print("\nModel Metrics:")
print(segmenter.get_metrics())