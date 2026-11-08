"""Score an uploaded test CSV with the five saved classifiers."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Cell nucleus diagnosis", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

MODEL_NOTES = {
    "Logistic Regression": "Draws one straight boundary between benign and malignant.",
    "Decision Tree": "Asks a chain of yes/no questions about the cell measurements.",
    "KNN": "Finds the 5 most similar training samples and takes a vote.",
    "Naive Bayes": "Uses probability and treats each measurement as independent.",
    "Support Vector Machine": "Fits a curved boundary, then calibrates it into probabilities.",
}


def evaluate(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


@st.cache_resource
def load_artifacts():
    metadata = joblib.load(MODEL_DIR / "metadata.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }
    return metadata, scaler, models


try:
    metadata, scaler, models = load_artifacts()
except FileNotFoundError:
    st.error("The saved models are missing. Run train_models.py in this folder first.")
    st.stop()

feature_names = metadata["feature_names"]
target_name = metadata["target_name"]
class_names = metadata["class_names"]

st.title("Will this cell nucleus be called malignant?")
st.write(
    "Upload the held-out test CSV. This page loads the five models trained "
    "earlier, prepares the 30 measurements with the saved scaler, and scores "
    "the file."
)
st.caption(
    f"Author: {metadata['student_name']}. Random seed: {metadata['random_state']}. "
    "In the diagnosis column, 1 means malignant and 0 means benign."
)

uploaded = st.file_uploader("Upload test_data.csv", type=["csv"])
if uploaded is None:
    st.info("Upload the test CSV to see all five models, then pick one for the matrix.")
    st.stop()

uploaded_table = pd.read_csv(uploaded)
if uploaded_table.empty:
    st.error("The uploaded file has no rows.")
    st.stop()

required_columns = feature_names + [target_name]
missing_columns = [column for column in required_columns if column not in uploaded_table.columns]
if missing_columns:
    st.error("Missing columns: " + ", ".join(missing_columns))
    st.stop()

measurements = uploaded_table[feature_names]
diagnosis = uploaded_table[target_name]
seen_labels = set(diagnosis.unique())
if not seen_labels.issubset({0, 1}):
    st.error("The diagnosis column must contain only 0 and 1.")
    st.stop()
if len(seen_labels) < 2:
    st.error("AUC needs both classes in the uploaded file.")
    st.stop()

prepared = scaler.transform(measurements)

comparison_rows = []
stored_predictions = {}
for name, model in models.items():
    predicted = model.predict(prepared)
    probability = model.predict_proba(prepared)[:, 1]
    stored_predictions[name] = predicted
    row = {"ML Model Name": name}
    row.update(evaluate(diagnosis, predicted, probability))
    comparison_rows.append(row)

st.subheader("All five models on this file")
st.dataframe(pd.DataFrame(comparison_rows).round(4))

chosen_name = st.selectbox("Choose a model to inspect", list(MODEL_FILES))
st.caption(MODEL_NOTES[chosen_name])
chosen_pred = stored_predictions[chosen_name]
chosen_metrics = next(row for row in comparison_rows if row["ML Model Name"] == chosen_name)

st.subheader(f"Evaluation metrics: {chosen_name}")
metric_columns = st.columns(6)
for column, metric_name in zip(
    metric_columns, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
):
    column.metric(metric_name, f"{chosen_metrics[metric_name]:.4f}")

matrix_column, report_column = st.columns(2)

with matrix_column:
    st.subheader("Confusion matrix")
    matrix = confusion_matrix(diagnosis, chosen_pred, labels=[0, 1])
    figure, axis = plt.subplots()
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    st.pyplot(figure)
    plt.close(figure)

with report_column:
    st.subheader("Classification report")
    report = classification_report(
        diagnosis,
        chosen_pred,
        labels=[0, 1],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(4))