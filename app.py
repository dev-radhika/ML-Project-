
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "defect_model.joblib"
DATA_PATH = BASE_DIR / "data" / "poi_3_0.csv"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"
IMPORTANCE_PATH = BASE_DIR / "outputs" / "feature_importance.csv"
CONFUSION_PATH = BASE_DIR / "outputs" / "confusion_matrix.png"
COMPARISON_PATH = BASE_DIR / "outputs" / "model_comparison.csv"

FEATURES = [
    "CBO",
    "DCC",
    "ExportCoupling",
    "ImportCoupling",
    "NOM",
    "WMC",
]

st.set_page_config(
    page_title="Software Defect Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .sub-title {
        color: #9CA3AF;
        font-size: 15px;
        margin-bottom: 28px;
    }
    .section-gap {
        margin-top: 28px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_class_one_probability(model, data):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)
        classes = list(model.classes_)
        if 1 in classes:
            return probabilities[:, classes.index(1)]
        return probabilities[:, -1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(data)
        return 1 / (1 + np.exp(-scores))

    return None


def read_uploaded_dataset(uploaded_file):
    extension = Path(uploaded_file.name).suffix.lower()

    if extension == ".csv":
        return pd.read_csv(uploaded_file)

    if extension == ".xlsx":
        return pd.read_excel(uploaded_file)

    raise ValueError("Please upload a CSV or XLSX file.")


def validate_prediction_data(data):
    missing_columns = [
        column for column in FEATURES if column not in data.columns
    ]

    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"

    feature_data = data[FEATURES].copy()

    for column in FEATURES:
        feature_data[column] = pd.to_numeric(
            feature_data[column], errors="coerce"
        )

    if feature_data.isnull().any().any():
        return False, "The dataset contains missing or non-numeric feature values."

    if not np.isfinite(feature_data.to_numpy()).all():
        return False, "The dataset contains infinite or invalid values."

    return True, feature_data


def display_external_evaluation(actual, predictions, probabilities):
    actual = pd.to_numeric(actual, errors="coerce")

    if actual.isnull().any():
        st.warning("The Defect column contains missing or invalid labels.")
        return

    if not set(actual.unique()).issubset({0, 1}):
        st.warning("The Defect column must contain only 0 and 1.")
        return

    if len(actual.unique()) != 2:
        st.warning("Evaluation requires both defect classes: 0 and 1.")
        return

    actual = actual.astype(int)

    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(actual, predictions, zero_division=0)
    recall = recall_score(actual, predictions, zero_division=0)
    f1 = f1_score(actual, predictions, zero_division=0)
    balanced_accuracy = balanced_accuracy_score(actual, predictions)

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")
    col4.metric("F1 Score", f"{f1:.3f}")
    col5.metric("Balanced Accuracy", f"{balanced_accuracy:.3f}")

    if probabilities is not None:
        auc = roc_auc_score(actual, probabilities)
        st.metric("ROC-AUC", f"{auc:.3f}")

    matrix = confusion_matrix(
    actual,
    predictions,
    labels=[0, 1]
)

    st.subheader("Confusion Matrix")

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Non-defective", "Actual Defective"],
        columns=["Predicted Non-defective", "Predicted Defective"],
    )

    st.dataframe(matrix_df, use_container_width=True)


model = load_model()
dataset = load_dataset()
metrics = load_metrics()

if model is None:
    st.error("Trained model not found. Please run the training script first.")
    st.code("python src/train_model.py")
    st.stop()

with st.sidebar:
    st.markdown("## Project information")
    st.write("Dataset: PROMISE / Apache POI 3.0")
    st.write("Prediction unit: software class")
    st.write("Target: historical defect label (0/1)")

    st.markdown("---")
    st.markdown("### Navigate")

    page = st.radio(
        "Select a page",
        [
            "Overview",
            "Explore dataset",
            "Predict defects",
            "Model explanation",
            "Evaluation",
        ],
        label_visibility="collapsed",
    )


if page == "Overview":
    st.markdown(
        '<div class="main-title">Software Defect Prediction</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">A machine-learning decision-support system for prioritising software classes for testing.</div>',
        unsafe_allow_html=True,
    )

    st.header("What does this system do?")

    st.write(
        "It estimates whether a software class is defect-prone using six "
        "class-level software metrics. The output can help a team prioritise "
        "code review and testing. It does not locate bugs or prove that a "
        "class contains a defect."
    )

    if not dataset.empty:
        total_rows = len(dataset)
        defective_count = int(dataset["Defect"].sum())
        defect_percentage = defective_count / total_rows * 100

        col1, col2, col3 = st.columns(3)

        col1.metric("Dataset rows", total_rows)
        col2.metric("Defective classes", defective_count)
        col3.metric("Defect prevalence", f"{defect_percentage:.1f}%")

    st.markdown("### Selected model")
    st.write("Random Forest")

    st.info(
        "The reported test metrics come from a held-out stratified test set. "
        "Model selection used cross-validation on training data only."
    )

    st.markdown("### Features used")

    feature_descriptions = {
        "CBO": "Coupling Between Objects",
        "DCC": "Direct Class Coupling",
        "ExportCoupling": "Export Coupling",
        "ImportCoupling": "Import Coupling",
        "NOM": "Number of Methods",
        "WMC": "Weighted Methods per Class",
    }

    feature_df = pd.DataFrame(
        {
            "Feature": feature_descriptions.keys(),
            "Description": feature_descriptions.values(),
        }
    )

    st.dataframe(feature_df, use_container_width=True)


elif page == "Explore dataset":
    st.title("Explore Dataset")

    if dataset.empty:
        st.error("Dataset file not found.")
    else:
        st.write("The project uses the PROMISE Apache POI 3.0 dataset.")

        col1, col2 = st.columns(2)

        col1.metric("Total records", len(dataset))
        col2.metric("Total features", len(FEATURES))

        st.subheader("Dataset preview")
        st.dataframe(dataset.head(20), use_container_width=True)

        st.subheader("Dataset information")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Dataset shape")
            st.write(dataset.shape)

        with col2:
            st.write("Missing values")
            st.dataframe(
                dataset.isnull().sum().rename("Missing values"),
                use_container_width=True,
            )

        if "Defect" in dataset.columns:
            st.subheader("Defect distribution")

            distribution = (
                dataset["Defect"]
                .value_counts()
                .sort_index()
                .rename(index={0: "Non-defective", 1: "Defective"})
            )

            st.bar_chart(distribution)

        st.subheader("Feature statistics")
        st.dataframe(
            dataset[FEATURES].describe().T,
            use_container_width=True,
        )


elif page == "Predict defects":
    st.title("Predict Software Defects")

    st.write(
        "Enter the six software metrics below to predict whether a software "
        "class is likely to be defect-prone."
    )

    st.subheader("Individual prediction")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            cbo = st.number_input("CBO", min_value=0.0, value=5.0)
            dcc = st.number_input("DCC", min_value=0.0, value=3.0)

        with col2:
            export_coupling = st.number_input(
                "ExportCoupling", min_value=0.0, value=2.0
            )
            import_coupling = st.number_input(
                "ImportCoupling", min_value=0.0, value=2.0
            )

        with col3:
            nom = st.number_input("NOM", min_value=0.0, value=10.0)
            wmc = st.number_input("WMC", min_value=0.0, value=15.0)

        submitted = st.form_submit_button(
            "Predict Defect",
            use_container_width=True,
        )

    if submitted:
        input_data = pd.DataFrame(
            [
                {
                    "CBO": cbo,
                    "DCC": dcc,
                    "ExportCoupling": export_coupling,
                    "ImportCoupling": import_coupling,
                    "NOM": nom,
                    "WMC": wmc,
                }
            ]
        )

        prediction = int(model.predict(input_data)[0])
        probability = get_class_one_probability(model, input_data)

        st.subheader("Prediction result")

        if prediction == 1:
            st.error("Predicted result: Defective class")
        else:
            st.success("Predicted result: Non-defective class")

        if probability is not None:
            st.metric(
                "Defect probability",
                f"{float(probability[0]) * 100:.2f}%",
            )

    st.markdown("---")
    st.subheader("External dataset testing")

    st.write(
        "Upload your own CSV or Excel (.xlsx) dataset and get defect predictions. "
        "The uploaded dataset must contain the six required software metric columns."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or XLSX file",
        type=["csv", "xlsx"],
    )

    if uploaded_file is not None:
        try:
            external_data = read_uploaded_dataset(uploaded_file)

            st.subheader("Uploaded dataset preview")
            st.dataframe(external_data.head(), use_container_width=True)

            is_valid, result = validate_prediction_data(external_data)

            if not is_valid:
                st.error(result)
            else:
                feature_data = result

                predictions = model.predict(feature_data).astype(int)
                probabilities = get_class_one_probability(
                    model, feature_data
                )

                results = external_data.copy()
                results["Predicted_Defect"] = predictions
                results["Prediction_Label"] = np.where(
                    predictions == 1,
                    "Defective",
                    "Non-defective",
                )

                if probabilities is not None:
                    results["Defect_Probability"] = probabilities

                results["Risk_Level"] = np.where(
                    predictions == 1,
                    "High Risk",
                    "Low Risk",
                )

                st.success("Predictions generated successfully.")

                col1, col2 = st.columns(2)

                col1.metric("Total classes", len(results))
                col2.metric(
                    "Predicted defective classes",
                    int((predictions == 1).sum()),
                )

                st.subheader("Prediction results")
                st.dataframe(results, use_container_width=True)

                csv_data = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download prediction results",
                    data=csv_data,
                    file_name="defect_predictions.csv",
                    mime="text/csv",
                )

                if "Defect" in external_data.columns:
                    st.markdown("---")
                    st.subheader("External dataset evaluation")

                    display_external_evaluation(
                        external_data["Defect"],
                        predictions,
                        probabilities,
                    )

        except Exception as error:
            st.error(f"Unable to process the uploaded dataset: {error}")


elif page == "Model explanation":
    st.title("Model Explanation")

    st.write(
        "This section explains which software metrics influenced the "
        "trained model's predictions."
    )

    if IMPORTANCE_PATH.exists():
        importance = pd.read_csv(IMPORTANCE_PATH)

        st.subheader("Feature importance")
        st.dataframe(importance, use_container_width=True)

        numeric_columns = importance.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:
            value_column = numeric_columns[-1]
            name_column = importance.columns[0]

            chart_data = importance.set_index(name_column)[value_column]
            st.bar_chart(chart_data)

    else:
        st.warning("Feature importance file not found.")

    if not dataset.empty:
        st.subheader("Example software class")

        selected_index = st.number_input(
            "Select dataset row",
            min_value=0,
            max_value=len(dataset) - 1,
            value=0,
            step=1,
        )

        selected_row = dataset.iloc[[int(selected_index)]][FEATURES]

        st.dataframe(selected_row, use_container_width=True)

        prediction = int(model.predict(selected_row)[0])
        probability = get_class_one_probability(model, selected_row)

        if prediction == 1:
            st.error("Model prediction: Defective")
        else:
            st.success("Model prediction: Non-defective")

        if probability is not None:
            st.write(
                f"Estimated defect probability: {float(probability[0]) * 100:.2f}%"
            )


elif page == "Evaluation":
    st.title("Model Evaluation")

    st.write(
        "The following results represent the trained model's evaluation "
        "on the held-out test dataset."
    )

    if metrics:
        metric_labels = [
    ("Accuracy", "test_accuracy"),
    ("Precision", "test_precision"),
    ("Recall", "test_recall"),
    ("F1 Score", "test_f1"),
    ("Balanced Accuracy", "test_balanced_accuracy"),
    ("ROC-AUC", "test_roc_auc"),
]

        available_metrics = [
            (label, key)
            for label, key in metric_labels
            if key in metrics
        ]

        for start in range(0, len(available_metrics), 3):
            columns = st.columns(3)

            for column, (label, key) in zip(
                columns,
                available_metrics[start:start + 3],
            ):
                column.metric(label, f"{float(metrics[key]):.3f}")

    else:
        st.warning("Evaluation metrics file not found.")

    if CONFUSION_PATH.exists():
        st.subheader("Confusion Matrix")
        st.image(
            str(CONFUSION_PATH),
            use_container_width=True,
        )
    else:
        st.warning("Confusion matrix image not found.")

    if COMPARISON_PATH.exists():
        st.subheader("Model Comparison")

        comparison = pd.read_csv(COMPARISON_PATH)
        st.dataframe(comparison, use_container_width=True)

        numeric_columns = comparison.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:
            st.bar_chart(
                comparison.set_index(comparison.columns[0])[numeric_columns]
            )

    st.markdown("---")
    st.caption(
        "Accuracy and F1 Score are different evaluation metrics. "
        "Results depend on the dataset and test split."
    )


st.markdown("---")
st.caption("Software Defect Prediction")
