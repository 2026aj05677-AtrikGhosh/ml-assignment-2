"""Train five classifiers, score them, and save them for the web app."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

STUDENT_NAME = "Atrik Ghosh"
RANDOM_STATE = 677

TARGET_NAME = "diagnosis"
TEST_SIZE = 0.20

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def evaluate(y_true, y_pred, y_score):
    """Score one model on the held-out exam rows."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    dataset = load_breast_cancer()
    feature_names = list(dataset.feature_names)

    X = pd.DataFrame(dataset.data, columns=feature_names)
    # The source file codes 0 as malignant and 1 as benign.
    # We flip that so 1 means malignant, the class we want to find.
    y = pd.Series(np.where(dataset.target == 0, 1, 0), name=TARGET_NAME)

    print("Student:", STUDENT_NAME)
    print("Random seed:", RANDOM_STATE)
    print("Rows:", X.shape[0], "Features:", X.shape[1])
    print("First five features:", feature_names[:5])
    print(y.value_counts().rename({0: "benign (0)", 1: "malignant (1)"}))

    if X.shape[0] < 500 or X.shape[1] < 12:
        raise SystemExit("Dataset is below the assignment minimum.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=RANDOM_STATE),
            ensemble=False,
        ),

    }

    rows = []
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        fitted_models[name] = model

        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        test_score = model.predict_proba(X_test_scaled)[:, 1]

        metrics = {"ML Model Name": name}
        metrics.update(evaluate(y_test, test_pred, test_score))
        metrics["Train Accuracy"] = accuracy_score(y_train, train_pred)
        rows.append(metrics)

        joblib.dump(model, MODEL_DIR / MODEL_FILES[name])

        print("\n" + name)
        print(
            classification_report(
                y_test,
                test_pred,
                labels=[0, 1],
                target_names=["benign", "malignant"],
                digits=4,
                zero_division=0,
            )
        )
        print("Confusion matrix [actual rows, predicted columns]:")
        print(confusion_matrix(y_test, test_pred, labels=[0, 1]))

    results = pd.DataFrame(rows)
    print("\nTest-set comparison. Precision, Recall, and F1 refer to malignant = 1.")
    print(results.round(4).to_string(index=False))

    test_frame = X_test.copy()
    test_frame[TARGET_NAME] = y_test.to_numpy()
    test_path = BASE_DIR / "test_data.csv"
    test_frame.to_csv(test_path, index=False)

    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(
        {
            "feature_names": feature_names,
            "target_name": TARGET_NAME,
            "class_names": ["benign", "malignant"],
            "student_name": STUDENT_NAME,
            "random_state": RANDOM_STATE,
        },
        MODEL_DIR / "metadata.joblib",
    )

    reloaded = joblib.load(MODEL_DIR / "logistic_regression.joblib")
    fresh_pred = fitted_models["Logistic Regression"].predict(X_test_scaled)
    saved_pred = reloaded.predict(X_test_scaled)
    if not np.array_equal(fresh_pred, saved_pred):
        raise SystemExit("The saved model did not reproduce its own predictions.")

    print("\nSaved logistic regression reloaded and matched.")
    print("Test file:", test_path, "rows:", len(test_frame))


if __name__ == "__main__":
    main()