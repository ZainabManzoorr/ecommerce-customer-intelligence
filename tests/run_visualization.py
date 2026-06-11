import pandas as pd
import matplotlib.pyplot as plt
from tests.run_segmentation import segmented_df

cluster_summary = segmented_df.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()

cluster_summary.plot(kind="bar", figsize=(10,6))
plt.title("Cluster Behavior Profile (RFM Averages)")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.show()