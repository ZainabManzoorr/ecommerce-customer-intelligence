from src.models.kmeans_segmentation import CustomerSegmentationKMeans
from src.models.customer_segmentation import CustomerSegmentLabeler
from tests.run_features import feature_df

# -------------------------
# STEP 1: Train Segmentation Model
# -------------------------
segmenter = CustomerSegmentationKMeans(feature_df)

segmented_df = (
    segmenter
    .prepare_features()
    .scale_features()
    .train_kmeans(n_clusters=4)
    .get_segmented_data()
)

# -------------------------
# STEP 2: Assign Business Labels
# -------------------------
label_mapping = {
    0: "Low Engagement",
    1: "Regular Customers",
    2: "Loyal Customers",
    3: "VIP Customers"
}

labeler = CustomerSegmentLabeler(segmented_df)
segmented_df = labeler.assign_labels(label_mapping)

# -------------------------
# STEP 3: Generate Segment Summary
# -------------------------
summary = labeler.segment_summary()

# -------------------------
# STEP 4: Output Results
# -------------------------
print(segmented_df.head())
print(segmented_df["Cluster"].value_counts())
print(summary)

segmented_df.to_csv("outputs/customer_segments.csv", index=False)
summary.to_csv("outputs/segment_summary.csv", index=False)