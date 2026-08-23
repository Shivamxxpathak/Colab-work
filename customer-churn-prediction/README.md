# Customer Churn Prediction & Retention Analytics

An end-to-end supervised machine learning project that predicts telecom customer churn, compares multiple classification algorithms, evaluates model reliability, and exposes the best model through a Streamlit prediction app.

## Project objective

Customer churn is a binary classification problem with a direct business objective: identify customers who are likely to leave so a retention team can intervene before churn occurs.

This project is designed as a portfolio-quality ML workflow rather than a single `model.fit()` notebook.

## Dataset

The project is designed for the **IBM Telco Customer Churn** dataset. The commonly used public dataset contains 7,043 customer records and 21 columns, with `Churn` as the target.

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Do not commit private customer data to GitHub.

## ML workflow

```text
Raw Customer Data
       ↓
Data Cleaning
       ↓
Train/Test Split
       ↓
ColumnTransformer
       ├── Numeric: median imputation + scaling
       └── Categorical: mode imputation + one-hot encoding
       ↓
Model Comparison
       ├── Logistic Regression
       ├── KNN
       ├── Naive Bayes
       ├── Decision Tree
       ├── Random Forest
       ├── SVM
       ├── AdaBoost
       └── Gradient Boosting
       ↓
Soft Voting Ensemble
       ↓
ROC-AUC / F1 / Precision / Recall / Accuracy
       ↓
Best Model
       ↓
Churn Probability + Risk Segment
       ↓
Streamlit App
```

## Models

| Model | Role |
|---|---|
| Logistic Regression | Interpretable baseline |
| KNN | Distance-based classifier |
| Naive Bayes | Probabilistic baseline |
| Decision Tree | Non-linear, interpretable model |
| Random Forest | Bagging ensemble |
| SVM | Margin-based classifier |
| AdaBoost | Adaptive boosting |
| Gradient Boosting | Sequential boosting |
| Voting Ensemble | Combines complementary models |

## Evaluation

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- 5-fold stratified cross-validation ROC-AUC

For churn, recall and ROC-AUC should be considered alongside accuracy because missing a high-risk customer can be more costly than contacting an additional customer.

## Project structure

```text
customer-churn-prediction/
├── app.py
├── data/
│   └── README.md
├── models/
├── outputs/
├── scripts/
│   └── train.py
├── src/
│   ├── churn_pipeline.py
│   └── visualize.py
├── .gitignore
├── LICENSE
├── PROJECT_STRUCTURE.md
├── PROJECT_SUMMARY.md
├── QUICKSTART.md
├── README.md
└── requirements.txt
```

## Run locally

### 1. Create an environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Save the Telco CSV as:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 4. Train and evaluate

```bash
python scripts/train.py
```

The pipeline creates:

```text
outputs/model_comparison.csv
outputs/model_comparison.png
outputs/best_churn_model.joblib
outputs/test_predictions.csv
```

### 5. Launch the app

```bash
streamlit run app.py
```

## Risk interpretation

- **High risk:** ≥ 70%
- **Medium risk:** 40–69.9%
- **Low risk:** < 40%

These thresholds are demonstration business rules, not calibrated operational guarantees.

## Portfolio talking points

This project demonstrates:

- End-to-end supervised ML
- Leakage-aware preprocessing with `Pipeline` and `ColumnTransformer`
- Categorical encoding with `OneHotEncoder`
- Model benchmarking across classical algorithms and ensembles
- Stratified cross-validation
- Business-oriented churn-risk interpretation
- Model persistence with Joblib
- Interactive ML deployment with Streamlit

## Next improvements

- XGBoost comparison
- Hyperparameter optimization with `RandomizedSearchCV`
- Probability calibration
- SHAP-based explainability
- Threshold optimization based on retention cost
- MLflow experiment tracking
- Docker deployment
- Automated CI testing
