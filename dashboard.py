
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="FraudLens | Fraud Analytics",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

MODEL_PATHS = {
    "Logistic Regression": BASE_DIR / "logistic_regression_model.pkl",
    "Random Forest": BASE_DIR / "fraud_model.pkl",
    "XGBoost": BASE_DIR / "xgboost_fraud_model.pkl",
    "Isolation Forest": BASE_DIR / "isolation_forest_model.pkl"
}

# -------------------------------
# PREMIUM FRAUDLENS THEME
# -------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0D0F10;
    --surface: #17191B;
    --surface-hover: #202326;
    --accent: #F97316;
    --accent-hover: #FB923C;
    --fraud: #FF755E;
    --text: #F5F5F5;
    --muted: #A5ABB3;
    --border: #2B2F33;
}

/* Main application */
.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Main content spacing */
.block-container {
    max-width: 1500px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

/* Typography */
h1, h2, h3, h4 {
    color: var(--text) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    letter-spacing: -0.035em;
}

h1 {
    font-weight: 800 !important;
    letter-spacing: -0.045em;
}

h2 {
    font-weight: 700 !important;
}

h3, h4 {
    font-weight: 600 !important;
}

p, label, li {
    color: var(--text);
    line-height: 1.65;
}

[data-testid="stCaptionContainer"] {
    color: var(--muted);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 22px;
    min-height: 115px;
    transition: border-color 0.2s ease,
                background 0.2s ease;
}

[data-testid="stMetric"]:hover {
    background: var(--surface-hover);
    border-color: #555B62;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.04em;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111315;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--muted);
}

/* Inputs and dropdowns */
.stSelectbox [data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input {
    background: var(--surface);
    color: var(--text);
    border-color: var(--border);
    border-radius: 10px;
}

[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1px dashed #555B62;
    border-radius: 14px;
    padding: 16px;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent);
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: var(--accent);
    color: #111315;
    border: 1px solid var(--accent);
    border-radius: 10px;
    padding: 0.62rem 1rem;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: var(--accent-hover);
    border-color: var(--accent-hover);
    color: #111315;
    transform: translateY(-1px);
}

.stButton > button:focus,
.stDownloadButton > button:focus {
    box-shadow: 0 0 0 2px var(--bg),
                0 0 0 4px var(--accent);
}

/* Data tables */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid var(--border);
}

/* Dividers */
hr {
    border-color: var(--border) !important;
}

/* Sliders */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent);
}

/* Selection highlight */
::selection {
    background: #F97316;
    color: #111315;
}

/* Mobile-friendly spacing */
@media (max-width: 768px) {
    .block-container {
        padding: 1.2rem 1rem 2rem;
    }

    [data-testid="stMetric"] {
        padding: 14px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# LOAD MODELS
# -------------------------------

@st.cache_resource
def load_model(path):
    return joblib.load(path)


# -------------------------------
# PREDICTION HELPER
# -------------------------------

def predict_transactions(model, model_name, X):
    raw_predictions = model.predict(X)

    if model_name == "Isolation Forest":
        # Isolation Forest: -1 = anomaly, 1 = normal
        return (raw_predictions == -1).astype(int)

    # Classifiers: 0 = legitimate, 1 = fraud
    return np.asarray(raw_predictions, dtype=int)


# -------------------------------
# HEADER
# -------------------------------

st.markdown("# 🛡️ FraudLens")
st.markdown("### Intelligent Credit Card Fraud Analytics")
st.caption(
    "Machine learning portfolio project · Batch transaction analysis"
)

st.divider()


# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.markdown("## 🛡️ FraudLens")
st.sidebar.caption("Transaction intelligence")

available_models = [
    name for name, path in MODEL_PATHS.items()
    if path.exists()
]

if not available_models:
    st.error(
        "No saved models were found. Run project.py first "
        "and check that the model files are in this folder."
    )
    st.stop()

selected_model_name = st.sidebar.selectbox(
    "Select detection model",
    available_models
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model information")

model_descriptions = {
    "Logistic Regression": "Supervised linear classification",
    "Random Forest": "Supervised ensemble of decision trees",
    "XGBoost": "Supervised gradient-boosted trees",
    "Isolation Forest": "Unsupervised anomaly detection"
}

st.sidebar.write(model_descriptions[selected_model_name])

if selected_model_name == "Isolation Forest":
    st.sidebar.info(
        "Anomalies are treated as potential fraud. "
        "An anomaly is not necessarily fraudulent."
    )

try:
    model = load_model(str(MODEL_PATHS[selected_model_name]))
except Exception as exc:
    st.error(f"Could not load the selected model: {exc}")
    st.stop()


# -------------------------------
# CSV UPLOAD
# -------------------------------

st.markdown("## Upload transactions")
st.write(
    "Upload a CSV containing Time, V1–V28 and Amount. "
    "An optional Class column can be included for evaluation."
)

uploaded_file = st.file_uploader(
    "Choose a transaction CSV",
    type=["csv"],
    key="transaction_upload"
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin analysis.")
    st.stop()

try:
    uploaded_df = pd.read_csv(uploaded_file)

    if uploaded_df.empty:
        st.error("The uploaded CSV is empty.")
        st.stop()

    if "Class" in uploaded_df.columns:
        actual_labels = uploaded_df["Class"].copy()
        input_df = uploaded_df.drop(columns=["Class"]).copy()
    else:
        actual_labels = None
        input_df = uploaded_df.copy()

    expected_features = list(model.feature_names_in_)

    missing = [
        col for col in expected_features
        if col not in input_df.columns
    ]

    extra = [
        col for col in input_df.columns
        if col not in expected_features
    ]

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.stop()

    if extra:
        st.warning(
            "Extra columns will be ignored: " + ", ".join(extra)
        )

    X_upload = input_df[expected_features].copy()
    X_upload = X_upload.apply(pd.to_numeric, errors="coerce")

    if X_upload.isna().any().any():
        st.error(
            "The feature columns contain missing or non-numeric values. "
            "Correct the CSV and upload it again."
        )
        st.stop()

    if not np.isfinite(X_upload.to_numpy(dtype=float)).all():
        st.error("The CSV contains infinite numeric values.")
        st.stop()

    if actual_labels is not None:
        actual_labels = pd.to_numeric(
            actual_labels, errors="coerce"
        )

        if (
            actual_labels.isna().any()
            or not actual_labels.isin([0, 1]).all()
        ):
            st.error("Class must contain only 0 or 1.")
            st.stop()

        actual_labels = actual_labels.astype(int)

    st.markdown("### Dataset preview")
    st.dataframe(
        uploaded_df.head(10),
        use_container_width=True,
        hide_index=True
    )

    if st.button(
        "Analyze transactions",
        use_container_width=True,
        type="primary"
    ):
        predictions = predict_transactions(
            model, selected_model_name, X_upload
        )

        results = uploaded_df.copy()
        results["Predicted Class"] = predictions
        results["Prediction"] = np.where(
            predictions == 1,
            "Potential Fraud",
            "Legitimate"
        )

        st.session_state["fraudlens_results"] = results
        st.session_state["fraudlens_model"] = selected_model_name
        st.session_state["fraudlens_labels"] = (
            actual_labels.to_numpy()
            if actual_labels is not None
            else None
        )

except Exception as exc:
    st.error(f"Could not process this CSV: {exc}")
    st.stop()


# -------------------------------
# DISPLAY SAVED ANALYSIS
# -------------------------------

results = st.session_state.get("fraudlens_results")
analysis_model = st.session_state.get("fraudlens_model")

if results is None or analysis_model != selected_model_name:
    st.info("Click **Analyze transactions** to generate predictions.")
    st.stop()

predictions = results["Predicted Class"].to_numpy()

total = len(results)
fraud_count = int((predictions == 1).sum())
legitimate_count = total - fraud_count
fraud_rate = fraud_count / total * 100


# -------------------------------
# KPI METRICS
# -------------------------------

st.markdown("## Analysis overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Transactions analyzed", f"{total:,}")
c2.metric("Potential fraud", f"{fraud_count:,}")
c3.metric("Legitimate predictions", f"{legitimate_count:,}")
c4.metric("Predicted fraud rate", f"{fraud_rate:.2f}%")

if fraud_count:
    st.warning(
        f"{fraud_count:,} transaction(s) were flagged for review."
    )
else:
    st.success("No transactions were flagged in this batch.")


# -------------------------------
# INTERACTIVE CHARTS
# -------------------------------

st.markdown("## Transaction analytics")

chart1, chart2 = st.columns(2)

with chart1:
    st.markdown("#### Prediction distribution")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        ["Legitimate", "Potential Fraud"],
        [legitimate_count, fraud_count]
    )
    ax.set_ylabel("Transactions")
    ax.set_title("Predicted transaction classes")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with chart2:
    st.markdown("#### Transaction amount distribution")

    if "Amount" in results.columns:
        plot_df = results.copy()
        plot_df["Amount"] = pd.to_numeric(
            plot_df["Amount"], errors="coerce"
        )
        plot_df = plot_df.dropna(subset=["Amount"])

        amount_chart = (
            plot_df.groupby("Prediction")["Amount"]
            .mean()
            .reindex(["Legitimate", "Potential Fraud"])
            .dropna()
        )

        st.bar_chart(amount_chart)
        st.caption("Mean transaction amount by predicted class.")
    else:
        st.info("Amount column is unavailable for this chart.")


# -------------------------------
# OPTIONAL MODEL EVALUATION
# -------------------------------

actual = st.session_state.get("fraudlens_labels")

if actual is not None:
    st.markdown("## Evaluation against supplied labels")
    st.caption(
        "Use trusted ground-truth labels for meaningful evaluation. "
        "Do not evaluate on labels used to train the model."
    )

    e1, e2, e3, e4 = st.columns(4)

    e1.metric(
        "Accuracy",
        f"{accuracy_score(actual, predictions) * 100:.2f}%"
    )
    e2.metric(
        "Fraud precision",
        f"{precision_score(actual, predictions, zero_division=0):.3f}"
    )
    e3.metric(
        "Fraud recall",
        f"{recall_score(actual, predictions, zero_division=0):.3f}"
    )
    e4.metric(
        "Fraud F1-score",
        f"{f1_score(actual, predictions, zero_division=0):.3f}"
    )

    st.markdown("#### Confusion matrix")

    cm = confusion_matrix(actual, predictions, labels=[0, 1])

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    )

    st.dataframe(cm_df, use_container_width=True)


# -------------------------------
# FILTER AND REVIEW TRANSACTIONS
# -------------------------------

st.markdown("## Transaction explorer")

filter_choice = st.selectbox(
    "Filter transactions",
    [
        "All transactions",
        "Potential fraud only",
        "Legitimate only"
    ]
)

filtered_results = results.copy()

if filter_choice == "Potential fraud only":
    filtered_results = filtered_results[
        filtered_results["Predicted Class"] == 1
    ]
elif filter_choice == "Legitimate only":
    filtered_results = filtered_results[
        filtered_results["Predicted Class"] == 0
    ]

if "Amount" in filtered_results.columns:
    amounts = pd.to_numeric(
        filtered_results["Amount"], errors="coerce"
    )

    valid_amounts = amounts.dropna()

    if not valid_amounts.empty:
        min_amount = float(valid_amounts.min())
        max_amount = float(valid_amounts.max())

        if min_amount < max_amount:
            amount_range = st.slider(
                "Transaction amount range",
                min_value=min_amount,
                max_value=max_amount,
                value=(min_amount, max_amount)
            )

            filtered_results = filtered_results[
                amounts.between(
                    amount_range[0], amount_range[1]
                )
            ]

st.caption(f"Showing {len(filtered_results):,} transaction(s).")

st.dataframe(
    filtered_results,
    use_container_width=True,
    hide_index=True
)


# -------------------------------
# DOWNLOAD RESULTS
# -------------------------------

csv_bytes = results.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download all prediction results",
    data=csv_bytes,
    file_name="fraud_predictions.csv",
    mime="text/csv",
    use_container_width=True
)


# -------------------------------
# FOOTER
# -------------------------------

st.divider()
st.caption(
    f"Active model: {selected_model_name} · "
    "FraudLens is a machine learning demonstration. "
    "Predictions are not proof of fraud; review flagged transactions "
    "before taking action."
)
