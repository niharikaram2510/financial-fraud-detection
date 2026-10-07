import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path


# --------------------------------------------------
# Load trained models
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

xgb_model = joblib.load(
    MODELS_DIR / "xgboost_fraud_model.pkl"
)

isolation_model = joblib.load(
    MODELS_DIR / "isolation_forest_model.pkl"
)


# --------------------------------------------------
# Model feature order
# --------------------------------------------------

MODEL_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_change",
    "amount_to_balance_ratio",
    "recipient_txn_count_before",
    "sender_recipient_count_before",
    "is_new_recipient",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]


# --------------------------------------------------
# Prepare transaction
# --------------------------------------------------

def prepare_transaction(
    step,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
    transaction_type,
    recipient_txn_count_before=0,
    sender_recipient_count_before=0
):

    balance_change = oldbalanceOrg - newbalanceOrig

    amount_to_balance_ratio = (
        amount / (oldbalanceOrg + 1)
    )

    is_new_recipient = int(
        recipient_txn_count_before == 0
    )

    transaction = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "balance_change": balance_change,
        "amount_to_balance_ratio": amount_to_balance_ratio,
        "recipient_txn_count_before": recipient_txn_count_before,
        "sender_recipient_count_before": sender_recipient_count_before,
        "is_new_recipient": is_new_recipient,
        "type_CASH_IN": 0,
        "type_CASH_OUT": 0,
        "type_DEBIT": 0,
        "type_PAYMENT": 0,
        "type_TRANSFER": 0
    }

    type_column = "type_" + transaction_type
    transaction[type_column] = 1

    return pd.DataFrame(
        [transaction],
        columns=MODEL_FEATURES
    )


# --------------------------------------------------
# Risk engine
# --------------------------------------------------

def calculate_risk(fraud_probability, anomaly_score):

    # High confidence fraud
    if fraud_probability >= 0.50:
        return "HIGH"

    # Suspicious transaction
    elif fraud_probability >= 0.01 or anomaly_score > 0:
        return "MEDIUM"

    # Low risk
    else:
        return "LOW"


# --------------------------------------------------
# Analyze transaction
# --------------------------------------------------

def analyze_transaction(transaction):

    fraud_probability = (
        xgb_model.predict_proba(transaction)[0, 1]
    )

    anomaly_score = (
        -isolation_model.decision_function(transaction)[0]
    )

    risk_level = calculate_risk(
        fraud_probability,
        anomaly_score
    )

    return {
        "fraud_probability": fraud_probability,
        "anomaly_score": anomaly_score,
        "risk_level": risk_level
    }


# --------------------------------------------------
# Human-readable SHAP explanations
# --------------------------------------------------

FEATURE_REASONS = {

    "amount_to_balance_ratio":
        "Transaction amount is very large relative to the sender's available balance.",

    "balance_change":
        "The sender's balance shows a significant change during the transaction.",

    "newbalanceOrig":
        "The sender's resulting balance contributes strongly to the fraud prediction.",

    "oldbalanceOrg":
        "The sender's original balance contributes to the fraud risk.",

    "amount":
        "The transaction amount contributes significantly to the fraud prediction.",

    "type_CASH_OUT":
        "The transaction is a CASH_OUT, which contributes to higher fraud risk in this dataset.",

    "type_TRANSFER":
        "The transaction is a TRANSFER, which contributes to higher fraud risk in this dataset.",

    "oldbalanceDest":
        "The recipient's original balance contributes to the fraud prediction.",

    "newbalanceDest":
        "The recipient's resulting balance contributes to the fraud prediction.",

    "recipient_txn_count_before":
        "The recipient's previous transaction history contributes to the prediction.",

    "sender_recipient_count_before":
        "The previous transaction relationship contributes to the prediction."
}


def explain_transaction(transaction, risk_level, top_n=4):

    if risk_level == "LOW":
        return [
            "No significant fraud indicators detected."
        ]

    explainer = shap.TreeExplainer(xgb_model)

    shap_values = explainer.shap_values(transaction)

    explanation = pd.DataFrame({
        "feature": transaction.columns,
        "shap_value": shap_values[0],
        "feature_value": transaction.iloc[0].values
    })

    # Keep features pushing toward fraud
    explanation = explanation[
        explanation["shap_value"] > 0
    ]

    # Ignore inactive transaction types
    type_features = [
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER"
    ]

    explanation = explanation[
        ~(
            explanation["feature"].isin(type_features)
            &
            (explanation["feature_value"] == 0)
        )
    ]

    explanation = explanation.sort_values(
        "shap_value",
        ascending=False
    ).head(top_n)

    reasons = []

    for _, row in explanation.iterrows():

        feature = row["feature"]

        reason = FEATURE_REASONS.get(
            feature,
            f"{feature} contributes to the fraud prediction."
        )

        reasons.append(reason)

    if not reasons:
        reasons.append(
            "No significant fraud indicators detected."
        )

    return reasons
# --------------------------------------------------
# Analyze raw transaction
# --------------------------------------------------

def analyze_raw_transaction(row):
    """
    Convert a raw PaySim transaction into model features
    and run the fraud detection pipeline.
    """

    transaction = prepare_transaction(
        step=int(row["step"]),
        amount=float(row["amount"]),
        oldbalanceOrg=float(row["oldbalanceOrg"]),
        newbalanceOrig=float(row["newbalanceOrig"]),
        oldbalanceDest=float(row["oldbalanceDest"]),
        newbalanceDest=float(row["newbalanceDest"]),
        transaction_type=row["type"],
        recipient_txn_count_before=int(
            row["recipient_txn_count_before"]
        ),
        sender_recipient_count_before=int(
            row["sender_recipient_count_before"]
        )
    )

    result = analyze_transaction(transaction)

    return {
        "fraud_probability": result["fraud_probability"],
        "anomaly_score": result["anomaly_score"],
        "risk_level": result["risk_level"],
        "transaction": transaction
    }