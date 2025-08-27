import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv("creditcard_with_fraud.csv")  # should contain 'reconstruction_error' and 'is_fraud'

st.title("💸 Fraud Detection Dashboard")

st.metric("Total Transactions", len(df))
st.metric("Frauds Detected", int(df['is_fraud_pca'].sum()))

# Line Chart
st.subheader("Reconstruction Error Trend")
fig1 = px.line(df, y='reconstruction_error', title="Anomaly Score Over Time")
st.plotly_chart(fig1)

# Heatmap
st.subheader("Fraud Heatmap (Location vs Amount)")
if "location" in df.columns:
    fig2 = px.density_heatmap(df, x="location", y="amount", z="reconstruction_error", nbinsx=20)
    st.plotly_chart(fig2)

# Top frauds
st.subheader("Top 5 Suspicious Transactions")
st.write(df[df['is_fraud_pca'] == True].sort_values(by="reconstruction_error", ascending=False).head(10))
