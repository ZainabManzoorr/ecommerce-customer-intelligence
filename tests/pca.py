import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tests.run_segmentation import segmented_df

# -----------------------
# 1. Prepare features
# -----------------------
features = segmented_df.drop(columns=["CustomerID", "Cluster"])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# -----------------------
# 2. Apply PCA (2D)
# -----------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

segmented_df["PCA1"] = pca_result[:, 0]
segmented_df["PCA2"] = pca_result[:, 1]

# -----------------------
# 3. Plot clusters
# -----------------------
plt.figure(figsize=(10, 6))

for cluster in sorted(segmented_df["Cluster"].unique()):
    cluster_data = segmented_df[segmented_df["Cluster"] == cluster]
    plt.scatter(
        cluster_data["PCA1"],
        cluster_data["PCA2"],
        label=f"Cluster {cluster}",
        alpha=0.6
    )

plt.title("Customer Segments Visualization (PCA + KMeans)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()