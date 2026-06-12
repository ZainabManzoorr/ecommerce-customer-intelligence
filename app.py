import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("outputs/customer_segments.csv")
    return df

df = load_data()

# -------------------------
# SIDEBAR FILTERS
# -------------------------
st.sidebar.title("🔍 Segment Explorer")

segment_filter = st.sidebar.multiselect(
    "Select Segment",
    options=df["Segment"].unique(),
    default=df["Segment"].unique()
)

filtered_df = df[df["Segment"].isin(segment_filter)]

# -------------------------
# TITLE
# -------------------------
st.title("📊 Customer Segmentation Dashboard")
st.write("Analyze customer behavior across different segments using K-Means clustering.")

# -------------------------
# KPI METRICS
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(filtered_df))
col2.metric("Avg Recency", round(filtered_df["Recency"].mean(), 2))
col3.metric("Avg Monetary", round(filtered_df["Monetary"].mean(), 2))

# -------------------------
# SEGMENT DISTRIBUTION
# -------------------------
st.subheader("📌 Segment Distribution")

segment_counts = (
    filtered_df["Segment"]
    .value_counts()
    .reset_index()
)

fig1 = px.bar(
    segment_counts,
    x="Segment",
    y="count",
    color="Segment",
    title="Customer Count by Segment"
)

st.plotly_chart(fig1, use_container_width=True)
# -------------------------
# MONETARY DISTRIBUTION
# -------------------------
st.subheader("💰 Monetary Value by Segment")

fig2 = px.box(
    filtered_df,
    x="Segment",
    y="Monetary",
    color="Segment",
    title="Spending Distribution per Segment"
)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# RECENCY VS FREQUENCY
# -------------------------
st.subheader("📈 Behavior Analysis")

fig3 = px.scatter(
    filtered_df,
    x="Recency",
    y="Frequency",
    color="Segment",
    title="Recency vs Frequency (Customer Behavior)"
)
st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# DATA TABLE
# -------------------------
st.subheader("📄 Customer Data")
st.dataframe(filtered_df)

# -------------------------
# DOWNLOAD OPTION
# -------------------------
st.download_button(
    label="📥 Download Segmented Data",
    data=filtered_df.to_csv(index=False),
    file_name="customer_segments_filtered.csv",
    mime="text/csv"
)