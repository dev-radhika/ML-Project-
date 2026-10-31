#  Software Defect Prediction

A Machine Learning project that predicts whether a software class is likely to contain defects based on software code metrics. It also provides defect probability and risk level through an interactive Streamlit dashboard.

## Dataset

- **Dataset:** PROMISE/BPD – Apache POI 3.0
- **Total records:** 442 software classes
- **Target column:** `Defect` (0 = Non-defective, 1 = Defective)
- **Features used:** CBO, DCC, ExportCoupling, ImportCoupling, NOM, WMC

## Technologies & Libraries

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Streamlit

## Machine Learning

The project compares Logistic Regression, Decision Tree, and Random Forest using GridSearchCV with 5-fold stratified cross-validation. Random Forest was selected as the best-performing model.

## Model Performance

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 79.8% |
| Precision | 83.1% |
| Recall    | 86.0% |
| F1-score  | 84.5% |
| ROC-AUC   | 83.2% |

## Key Features

- Predicts defective or non-defective software classes.
- Displays defect probability and risk level.
- Provides feature-importance explanations.
- Allows CSV and Excel dataset uploads.
- Displays evaluation metrics and confusion matrix when actual labels are available.
- Supports downloading prediction results as CSV.

## Run Locally

```bash
git clone <your-repository-url>
cd Software-Defect-Prediction

python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and run:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Project Goal

To demonstrate how machine learning and explainability techniques can help identify potentially defective software components during software development.
