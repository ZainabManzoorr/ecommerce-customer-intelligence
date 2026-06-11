from src.models.kmeans_segmentation import CustomerSegmentationKMeans
from tests.run_features import feature_df
segmenter = CustomerSegmentationKMeans(feature_df)

segmented_df = (
    segmenter
    .prepare_features()
    .scale_features()
    .train_kmeans(n_clusters=4)
    .get_segmented_data()
)

print(segmented_df.head())
print(segmented_df["Cluster"].value_counts())