
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")

st.sidebar.title("🔘 Navigation")
view = st.sidebar.radio("######################", [
    "Data Preprocessing",
    "Supervised Models",
    "Unsupervised Models",
    "Dashboard"
])

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

df = load_data()
X = df.drop(columns=["Class"])
y = df["Class"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# --- Data Preprocessing --- #
if view == "Data Preprocessing":
    st.subheader("🔍 Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Class distribution:")
    st.bar_chart(df["Class"].value_counts())

    st.subheader("📊 Amount Distribution by Class")
    fraud = df[df['Class'] == 1]
    valid = df[df['Class'] == 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    fig.suptitle('Amount per Transaction by Class')
    ax1.hist(fraud['Amount'], bins=50, color='red')
    ax1.set_title('Fraud')
    ax2.hist(valid['Amount'], bins=50, color='green')
    ax2.set_title('Valid')
    ax2.set_xlabel('Amount')
    ax2.set_ylabel('Number of Transactions')
    plt.xlim(0, 5000)
    plt.yscale('log')
    st.pyplot(fig)

    st.subheader("📊 Feature Scaling Preview")
    st.write(pd.DataFrame(X_scaled, columns=X.columns).head(10))

# --- Supervised Models --- #
elif view == "Supervised Models":
    st.subheader("✅ Supervised Learning")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced"),
       
    }

    for name, model in models.items():
        st.markdown(f"### {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"Confusion Matrix - {name}")
        st.pyplot(fig)

        st.text(classification_report(y_test, y_pred))
        st.metric(f"{name} F1 Score", round(f1_score(y_test, y_pred), 3))
    
        best_model = ""
    best_f1 = 0.0

    for name, model in models.items():
        st.markdown(f"### {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"Confusion Matrix - {name}")
        st.pyplot(fig)

        st.text(classification_report(y_test, y_pred))
        f1 = f1_score(y_test, y_pred)
        st.metric(f"{name} F1 Score", round(f1, 4))

        # Track best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = name

    # Display best model summary
    st.success(f"🏆 Best Model: {best_model} with F1 Score: {best_f1:.4f}")


# --- Unsupervised Models --- #
elif view == "Unsupervised Models":
    st.subheader("🧠 Unsupervised Learning")

    # Isolation Forest
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso.fit(X_scaled)
    iso_scores = -iso.decision_function(X_scaled)
    iso_threshold = np.percentile(iso_scores, 95)
    df["iso_score"] = iso_scores
    df["iso_pred"] = iso_scores > iso_threshold

    st.markdown("### Isolation Forest")
    fig = px.line(df, y="iso_score", title="Isolation Forest Scores")
    fig.add_hline(y=iso_threshold, line_color='red')
    st.plotly_chart(fig)
    st.metric("Anomalies (IsoForest)", int(df["iso_pred"].sum()))

    # One-Class SVM
    ocsvm = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1)
    ocsvm.fit(X_scaled)
    df["svm_pred"] = ocsvm.predict(X_scaled) == -1

    st.markdown("### One-Class SVM")
    svm_count = df["svm_pred"].sum()
    st.metric("Anomalies (SVM)", int(svm_count))
    st.write(df[df["svm_pred"] == True][["Amount"]].head(10))

# --- Dashboard --- #
elif view == "Dashboard":
    st.subheader("📊 Interactive Fraud Detection Dashboard")
    df["is_fraud"] = df["Class"] == 1

    st.metric("Total Transactions", len(df))
    st.metric("Confirmed Frauds", int(df["is_fraud"].sum()))

    st.markdown("### 💰 Amount Distribution")
    fig = px.histogram(df, x="Amount", color="is_fraud", nbins=100, title="Amount Distribution by Fraud")
    st.plotly_chart(fig)

    st.markdown("### 🔥 Heatmap - Correlation Matrix")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
    st.pyplot(fig)

    st.markdown("### 🚨 Top Suspicious Transactions")
    st.dataframe(df[df["is_fraud"] == True].sort_values(by="Amount", ascending=False).head(10))
    


