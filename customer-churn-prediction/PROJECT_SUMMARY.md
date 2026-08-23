# Portfolio Upgrade Summary

## What changed

- Replaced label-based categorical preprocessing with `ColumnTransformer` + `Pipeline` + `OneHotEncoder`.
- Added Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, SVM, AdaBoost and Gradient Boosting.
- Added a soft Voting Ensemble.
- Added stratified 5-fold cross-validation.
- Added reproducible train/test splitting.
- Added model comparison CSV and chart generation.
- Added Joblib model persistence.
- Added a Streamlit prediction application.
- Added explicit high/medium/low churn-risk interpretation.
- Removed synthetic data from the portfolio benchmark; the workflow expects the IBM Telco Churn dataset.
