
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay
)

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing Subscription Prediction")

st.write(
    "Compare five machine learning models to predict whether "
    "a customer will subscribe to a term deposit."
)

MODEL_DIR = "."

@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load(
            os.path.join(MODEL_DIR, "logistic_regression.joblib")
        ),
        "Decision Tree": joblib.load(
            os.path.join(MODEL_DIR, "decision_tree.joblib")
        ),
        "K-Nearest Neighbors": joblib.load(
            os.path.join(MODEL_DIR, "knn.joblib")
        ),
        "Naive Bayes": joblib.load(
            os.path.join(MODEL_DIR, "naive_bayes.joblib")
        ),
        "Support Vector Machine": joblib.load(
            os.path.join(MODEL_DIR, "support_vector_machine.joblib")
        )
    }

    preprocessor = joblib.load(
        os.path.join(MODEL_DIR, "preprocessor.joblib")
    )

    return models, preprocessor


try:
    models, preprocessor = load_models()
    st.success("Trained models loaded successfully.")

except Exception as e:
    st.error(f"Unable to load models: {e}")
    st.stop()


selected_model_name = st.selectbox(
    "Select a Machine Learning Model",
    list(models.keys())
)

uploaded_file = st.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(data.head())

    if "y" not in data.columns:
        st.error("The uploaded CSV must contain the target column 'y'.")
        st.stop()

    y_true = (
        data["y"]
        .astype(str)
        .str.lower()
        .map({"no": 0, "yes": 1, "0": 0, "1": 1})
    )

    if y_true.isna().any():
        st.error("Target column 'y' must contain yes/no or 1/0 values.")
        st.stop()

    X = data.drop(columns=["y"])

    try:
        # Apply the saved preprocessing
        X_processed = preprocessor.transform(X)

        # Select the trained model
        model = models[selected_model_name]

        # Generate predictions
        raw_predictions = model.predict(X_processed)

        # Convert predictions into numeric labels
        predictions = (
            pd.Series(raw_predictions)
            .astype(str)
            .str.lower()
            .map({"no": 0, "yes": 1, "0": 0, "1": 1})
            .to_numpy()
        )

        # Calculate scores for AUC
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_processed)[:, 1]
        else:
            scores = model.decision_function(X_processed)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, predictions)

        auc = roc_auc_score(y_true, scores)

        precision = precision_score(
            y_true, predictions, zero_division=0
        )

        recall = recall_score(
            y_true, predictions, zero_division=0
        )

        f1 = f1_score(
            y_true, predictions, zero_division=0
        )

        mcc = matthews_corrcoef(y_true, predictions)

        # Display results
        st.subheader(f"Performance: {selected_model_name}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # Display confusion matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots()

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No", "Yes"]
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False
        )

        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
