# app.py (enhanced, complete with detection summary)
import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from fraud_utils import (
    train_model,
    predict_supervised,
    anomaly_detect,
    hybrid_predict,
    suggest_threshold,
    get_pr_curve,
)

# ===============================
# Streamlit Setup
# ===============================
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Fraud Detection & Anomaly Dashboard — Enhanced")

# Session defaults
if "sup_threshold" not in st.session_state:
    st.session_state.sup_threshold = 0.5
if "contamination" not in st.session_state:
    st.session_state.contamination = 0.02

# ===============================
# File Upload
# ===============================
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded Dataset Preview:")
    st.dataframe(df.head())

    # ===============================
    # Controls
    # ===============================
    is_labeled = "Class" in df.columns
    st.sidebar.header("⚙️ Controls")

    if is_labeled:
        mode = st.sidebar.selectbox("Mode", ["Supervised", "Hybrid", "Unsupervised"], index=0)
        use_smote = st.sidebar.checkbox("Use SMOTE (if available)", value=False)
        use_balanced_rf = st.sidebar.checkbox("Use BalancedRandomForest (if available)", value=True)
        model, scaler = train_model(
            df, label_col="Class", use_smote=use_smote, use_balanced_rf=use_balanced_rf
        )
    else:
        mode = st.sidebar.selectbox("Mode", ["Unsupervised"], index=0)

    sup_threshold = st.sidebar.slider(
        "Fraud Probability Threshold", 0.0, 1.0, st.session_state.sup_threshold, 0.01, key="sup_threshold"
    )
    contamination = st.sidebar.slider(
        "IsolationForest Contamination (anomaly share)", 0.005, 0.20, st.session_state.contamination, 0.005, key="contamination"
    )

    # ===============================
    # Threshold Suggestion
    # ===============================
    if is_labeled:
        with st.expander("🔧 Auto-suggest threshold"):
            pred_tmp = predict_supervised(df, threshold=0.5)
            y_true = df["Class"].astype(int).values
            y_proba = pred_tmp["Fraud_Probability"].values

            strat = st.selectbox("Strategy", ["max_f1", "target_precision"], index=0)
            min_prec = st.slider("Min precision (for target_precision)", 0.5, 0.99, 0.8, 0.01)

            if st.button("Suggest threshold"):
                info = suggest_threshold(y_true, y_proba, strategy=strat, min_precision=min_prec)
                st.success(
                    f"Suggested threshold: {info['threshold']:.3f} | AP: {info['avg_precision']:.3f} | ROC-AUC: {info['roc_auc']:.3f}"
                )
                st.session_state.sup_threshold = float(info["threshold"])
                st.rerun()

            pr = get_pr_curve(y_true, y_proba)
            fig, ax = plt.subplots()
            ax.plot(pr["recall"], pr["precision"], lw=2)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision–Recall Curve")
            st.pyplot(fig)

    # ===============================
    # Run predictions
    # ===============================
    if mode == "Hybrid" and is_labeled:
        fusion = st.sidebar.selectbox("Fusion Strategy", ["or", "and", "weighted"], index=0)
        w_sup = st.sidebar.slider("Weight: Supervised", 0.0, 1.0, 0.7, 0.05)
        w_an = 1.0 - w_sup
        result_df = hybrid_predict(
            df,
            sup_threshold=st.session_state.sup_threshold,
            contamination=st.session_state.contamination,
            strategy=fusion,
            weight_supervised=w_sup,
            weight_anomaly=w_an,
        )
        pred_col = "Hybrid_Prediction"
    elif mode == "Supervised" and is_labeled:
        result_df = predict_supervised(df, threshold=st.session_state.sup_threshold)
        pred_col = "Fraud_Prediction"
    else:
        result_df = anomaly_detect(df, contamination=st.session_state.contamination)
        pred_col = "Anomaly"

    # ===============================
    # Save results
    # ===============================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fraud_results_{timestamp}.csv"
    result_df.to_csv(filename, index=False)
    st.info(f"💾 Results saved locally as {filename}")

    # ===============================
    # 📌 Detection Summary
    # ===============================
    st.subheader("📌 Detection Summary")

    anomalies = result_df[result_df[pred_col] == 1]
    normals = result_df[result_df[pred_col] == 0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔴 Anomalies / Frauds", len(anomalies))
    with col2:
        st.metric("🟢 Normal Transactions", len(normals))

    total = len(result_df)
    if total > 0:
        st.write(f"Frauds/Anomalies: **{len(anomalies)} ({len(anomalies)/total:.2%})**")
        st.write(f"Normal: **{len(normals)} ({len(normals)/total:.2%})**")

    if len(anomalies) > 0:
        st.write("📌 Top Suspicious Transactions:")
        st.dataframe(anomalies.head(20))

    # ===============================
    # 📊 Prediction Results
    # ===============================
    st.subheader("📊 Prediction Results")

    filter_choice = st.sidebar.radio(
        "🔍 Filter transactions:",
        ["Show All", "Show Fraud/Anomalies Only", "Show Normal Only"],
    )

    filtered_df = result_df.copy()
    if filter_choice == "Show Fraud/Anomalies Only":
        filtered_df = result_df[result_df[pred_col] == 1]
    elif filter_choice == "Show Normal Only":
        filtered_df = result_df[result_df[pred_col] == 0]

    st.dataframe(filtered_df.head(50))

    # Count chart
    st.subheader("📈 Prediction Distribution")
    st.bar_chart(result_df[pred_col].value_counts())

    # Pie chart
    st.subheader("🥧 Fraud vs Non-Fraud Pie Chart")
    fig, ax = plt.subplots()
    labels = result_df[pred_col].value_counts().index.astype(str)
    result_df[pred_col].value_counts().plot.pie(
        autopct='%1.1f%%', labels=labels, ax=ax, colors=["skyblue", "salmon"]
    )
    ax.set_ylabel("")
    st.pyplot(fig)

    # ===============================
    # Confusion Matrix & Report
    # ===============================
    if is_labeled:
        st.subheader("📌 Confusion Matrix")
        y_true = result_df["Class"].astype(int)
        y_pred = result_df[pred_col].astype(int)

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("📑 Classification Report")
        report = classification_report(
            y_true, y_pred, target_names=["Normal", "Fraud"], output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose())

        st.info(
            f"⚙️ Mode: {mode} | Threshold: {st.session_state.sup_threshold:.3f} | Contamination: {st.session_state.contamination:.3f}"
        )

    # ===============================
    # Download CSV
    # ===============================
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered Results CSV",
        data=csv,
        file_name="fraud_results_filtered.csv",
        mime="text/csv",
    )
