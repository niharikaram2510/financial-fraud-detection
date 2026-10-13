# Financial Transaction Fraud & Anomaly Detection using ML

An end-to-end machine learning project that detects potentially fraudulent financial transactions using supervised learning, unsupervised anomaly detection, and Explainable AI (XAI).

The system combines **XGBoost, Random Forest, Isolation Forest, and SHAP** with an interactive Streamlit dashboard to classify transaction risk and explain suspicious predictions.

## 1. Project Overview

Financial fraud detection is challenging because fraudulent transactions represent only a small proportion of total transactions. Traditional rule-based approaches may struggle to identify complex transaction patterns.

This project uses machine learning to analyse transaction amounts, account balance changes, and transaction-related features to identify suspicious activity.

The system provides:
- Fraud probability estimation.
- Anomaly detection.
- LOW, MEDIUM, and HIGH risk classification.
- Explainable predictions using SHAP.
- Interactive transaction monitoring and investigation through a Streamlit dashboard.

**Dataset:** PaySim synthetic financial transaction dataset.

**Application:** Simulated transaction monitoring, not a live banking system.

## 2. Objectives

- Develop a machine learning pipeline for financial fraud detection.
- Perform data preprocessing and feature engineering.
- Handle highly imbalanced transaction data.
- Compare supervised fraud classification models.
- Apply unsupervised anomaly detection.
- Explain model predictions using SHAP.
- Classify transactions according to risk.
- Build an interactive dashboard for transaction monitoring.

## 3. System Architecture

```text
PaySim Transaction Dataset
          |
          v
Data Preprocessing & EDA
          |
          v
Feature Engineering
          |
          v
Time-Based Train/Test Split
          |
     +----+------------------+
     |                       |
     v                       v
Random Forest             XGBoost
(Baseline)             (Primary Classifier)
                             |
                             v
                       Fraud Probability

Legitimate Training Data
          |
          v
    Isolation Forest
          |
          v
      Anomaly Score

Fraud Probability + Anomaly Score
                 |
                 v
          Risk Classification
                 |
                 v
           SHAP Explanations
                 |
                 v
        Streamlit Dashboard
```

## 4. Dataset

The project uses the PaySim synthetic financial transaction dataset.

| Dataset property | Value |
|---|---:|
| Total transactions | 6,362,620 |
| Original features | 11 |
| Fraudulent transactions | 8,213 |
| Legitimate transactions | 6,354,407 |
| Fraud rate | Approximately 0.129% |
| Transaction types | 5 |
| Time steps | 743 |

The dataset includes the following transaction types:

- CASH_IN
- CASH_OUT
- DEBIT
- PAYMENT
- TRANSFER

The dataset contains no missing values.

### Data splitting

A time-based split was used to evaluate the models on later transactions rather than randomly mixing transactions across the training and testing sets.

| Dataset | Transactions | Fraud cases |
|---|---:|---:|
| Training set | 6,239,040 | 6,559 |
| Test set | 123,580 | 1,654 |

The training period ends at step 594, with subsequent transactions reserved for testing.

## 5. Feature Engineering

Transaction and account balance information was used to derive additional features.

Important features include:

- **Balance change:** Difference between the sender's original and new balance.
- **Amount-to-balance ratio:** Transaction amount relative to the sender's original balance.
- **Recipient transaction count:** Number of previous transactions associated with a recipient.
- **Sender-recipient transaction count:** Previous transactions between the same sender and recipient.
- **New recipient indicator:** Identifies recipients not previously observed in the transaction history.
- **Transaction type:** Encoded using one-hot encoding.
- **Account balance features:** Original and new sender and recipient balances.

These features help the models learn transaction patterns associated with fraudulent and legitimate activity.

## 6. Machine Learning Models

### Random Forest

Random Forest was used as a supervised classification baseline. It combines multiple decision trees to classify transactions.

### XGBoost

XGBoost is the primary supervised fraud classification model. It estimates the probability that a transaction is fraudulent.

Class imbalance was addressed using `scale_pos_weight`, with a value of approximately 76.23 during training.

### Isolation Forest

Isolation Forest is an unsupervised anomaly detection algorithm trained on legitimate transactions from the training set.

It generates an anomaly score to identify transactions that differ from typical transaction patterns. A higher anomaly score indicates more anomalous behaviour.

### SHAP

SHAP (SHapley Additive exPlanations) provides explanations for XGBoost predictions by identifying features that contribute to a prediction and the direction of their influence.

This helps make transaction-level model outputs more interpretable.

## 7. Model Evaluation

The models were evaluated using ROC-AUC, PR-AUC, precision, recall, F1-score, and confusion matrices.

### Random Forest results

| Metric | Score |
|---|---:|
| ROC-AUC | 1.0000 |
| PR-AUC | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1-score | 1.0000 |

### XGBoost results

| Metric | Score |
|---|---:|
| ROC-AUC | 0.999998 |
| PR-AUC | 0.999849 |
| Precision | 0.998792 |
| Recall | 0.999395 |
| F1-score | 0.999093 |

XGBoost confusion matrix:

```text
                 Predicted
                 Legit   Fraud

Actual Legit     121924      2
Actual Fraud          1   1653
```

### Isolation Forest results

| Metric | Score |
|---|---:|
| ROC-AUC | 0.8837 |
| PR-AUC | 0.1631 |

Isolation Forest provides a complementary anomaly signal rather than directly replacing the supervised classifier.

### Feature importance

The models identified transaction and balance-related features as important predictive signals, particularly:

- Amount-to-balance ratio
- Balance change
- New sender balance
- Original sender balance
- Transaction amount

## 8. Risk Classification

The risk engine combines the fraud probability estimated by XGBoost with the anomaly score from Isolation Forest.

| Risk level | Rule |
|---|---|
| HIGH | Fraud probability ≥ 0.50 |
| MEDIUM | Fraud probability ≥ 0.01 and below 0.50, or the anomaly score triggers additional attention |
| LOW | Fraud probability below 0.01 and no anomaly flag |

These categories help prioritize transactions for further investigation. A risk label is a model-generated assessment, not proof that a transaction is fraudulent.

## 9. Streamlit Dashboard

An interactive dashboard was developed using Streamlit to demonstrate the transaction monitoring pipeline.

### Key features

- Batch transaction processing.
- Transaction count and risk KPI cards.
- Transaction activity visualizations.
- Risk distribution charts.
- Transaction-type analysis.
- Fraud probability trends.
- HIGH- and MEDIUM-risk attention queue.
- Individual transaction investigation.
- Fraud probability and anomaly score display.
- Transaction details and model-based explanations.
- Processed transaction table.
- Monitoring session reset.

The dashboard uses the saved models and a prepared demo dataset to simulate transaction monitoring.

## 10. Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost |
| Anomaly detection | Isolation Forest |
| Explainable AI | SHAP |
| Visualization | Matplotlib |
| Dashboard | Streamlit |
| Model persistence | Joblib |
| Development | Jupyter Notebook, VS Code |

## 11. Project Structure

```text
financial-fraud-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│       └── demo_transactions.csv
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   └── fraud_detector.py
│
├── models/
│   ├── isolation_forest_model.pkl
│   └── xgboost_fraud_model.pkl
│
├── dashboard/
│   └── app.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

The raw dataset is excluded from Git because of its large size. The prepared demo dataset and saved model files are included in the repository.

## 12. Installation and Usage

### Prerequisites

- Python 3.13
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/niharikaram2510/financial-fraud-detection.git
cd financial-fraud-detection
```

### Step 2: Create a virtual environment

**Windows (PowerShell):**

```powershell
py -3.13 -m venv venv
venv\Scripts\Activate.ps1
```

**macOS:**

```bash
python3.13 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the dashboard

```bash
streamlit run dashboard/app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

Use **Process next batch** in the sidebar to begin exploring the dashboard.

## 13. Limitations

- PaySim is a synthetic dataset and may not represent the complexity of real-world banking transactions.
- The very high supervised model scores reflect patterns in this particular dataset and should not be assumed to generalize to real financial systems.
- Some behavioural history features are sparse because many senders have little or no previous transaction history.
- Isolation Forest is less effective at identifying labelled fraud cases than the supervised classifier in this evaluation.
- The dashboard is a simulation and is not connected to live banking systems.
- Real-world deployment would require validation on representative financial data, monitoring for changing fraud patterns, and appropriate security and operational controls.

## 14. Future Scope

- Integrate real-time transaction streams.
- Evaluate models on additional real-world or independently validated datasets.
- Improve behavioural features using richer transaction histories.
- Introduce automated alerts and investigation workflows.
- Monitor model drift and periodically evaluate model performance.
- Explore secure cloud deployment and scalable processing.

## 15. Conclusion

This project demonstrates an end-to-end machine learning approach to financial transaction monitoring. By combining supervised fraud classification, unsupervised anomaly detection, explainable AI, and an interactive dashboard, the system supports risk prioritization and transaction-level investigation.

The project provides a foundation for further experimentation in financial fraud detection while acknowledging the limitations of synthetic data and the need for real-world validation.
