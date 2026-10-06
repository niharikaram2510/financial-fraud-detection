import sys
from pathlib import Path

import streamlit as st

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_detector import (
    prepare_transaction,
    analyze_transaction,
    explain_transaction
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🔐",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🔐 Financial Transaction Fraud Detection")

st.markdown(
    "### AI-powered fraud and anomaly detection system"
)

st.divider()


# --------------------------------------------------
# Transaction inputs
# --------------------------------------------------

st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100000.0
    )

    oldbalance_org = st.number_input(
        "Sender's Previous Balance",
        min_value=0.0,
        value=150000.0
    )

    newbalance_orig = st.number_input(
        "Sender's New Balance",
        min_value=0.0,
        value=50000.0
    )


with col2:

    oldbalance_dest = st.number_input(
        "Recipient's Previous Balance",
        min_value=0.0,
        value=200000.0
    )

    newbalance_dest = st.number_input(
        "Recipient's New Balance",
        min_value=0.0,
        value=300000.0
    )

    transaction_type = st.selectbox(
        "Transaction Type",
        [
            "CASH_IN",
            "CASH_OUT",
            "DEBIT",
            "PAYMENT",
            "TRANSFER"
        ]
    )


step = st.number_input(
    "Transaction Step",
    min_value=1,
    max_value=743,
    value=600
)


col3, col4 = st.columns(2)

with col3:

    recipient_history = st.number_input(
        "Recipient Transactions Before",
        min_value=0,
        value=0
    )


with col4:

    sender_recipient_history = st.number_input(
        "Previous Transactions Between Sender & Recipient",
        min_value=0,
        value=0
    )


st.divider()


# --------------------------------------------------
# Analyze button
# --------------------------------------------------

if st.button(
    "🔍 Analyze Transaction",
    use_container_width=True
):

    transaction = prepare_transaction(
        step=step,
        amount=amount,
        oldbalanceOrg=oldbalance_org,
        newbalanceOrig=newbalance_orig,
        oldbalanceDest=oldbalance_dest,
        newbalanceDest=newbalance_dest,
        transaction_type=transaction_type,
        recipient_txn_count_before=recipient_history,
        sender_recipient_count_before=sender_recipient_history
    )

    result = analyze_transaction(transaction)

    fraud_probability = result["fraud_probability"]
    anomaly_score = result["anomaly_score"]
    risk_level = result["risk_level"]

    reasons = explain_transaction(
        transaction,
        risk_level
    )


    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    st.divider()

    st.subheader("Analysis Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.4f}%"
        )

    with result_col2:

        st.metric(
            "Anomaly Score",
            f"{anomaly_score:.4f}"
        )

    with result_col3:

        st.metric(
            "Risk Level",
            risk_level
        )


    # --------------------------------------------------
    # Risk message
    # --------------------------------------------------

    if risk_level == "HIGH":

        st.error(
            "🚨 HIGH RISK — This transaction shows strong "
            "indicators of potential fraud."
        )

    elif risk_level == "MEDIUM":

        st.warning(
            "⚠️ MEDIUM RISK — This transaction shows "
            "suspicious characteristics and may require review."
        )

    else:

        st.success(
            "✅ LOW RISK — No significant fraud indicators detected."
        )


    # --------------------------------------------------
    # Reasons
    # --------------------------------------------------

    if risk_level == "LOW":
        st.subheader("Fraud Assessment")
    else:
        st.subheader("Why was this transaction flagged?")

    for reason in reasons:

        st.write(f"• {reason}")