"""End-to-end customer churn training pipeline."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def load_telco_data(path: str | Path) -> pd.DataFrame:
    """Load and clean the IBM Telco Customer Churn CSV."""
    df = pd.read_csv(path)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if "Churn" not in df.columns:
        raise ValueError("Dataset must contain a 'Churn' target column.")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("Int64")
    if df["Churn"].isna().any():
        raise ValueError("Churn contains values other than Yes/No.")
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)])


def build_models(random_state: int = RANDOM_STATE) -> Dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=random_state, n_jobs=-1),
        "SVM": SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=random_state),
    }


def train_and_evaluate(df: pd.DataFrame, output_dir: str | Path = "outputs") -> Tuple[pd.DataFrame, Pipeline]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    models = build_models()
    results = []
    fitted = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        results.append({"model": name, "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0), "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0), "roc_auc": roc_auc_score(y_test, proba), "cv_roc_auc_mean": cv_auc.mean(), "cv_roc_auc_std": cv_auc.std()})
        fitted[name] = pipe

    ensemble = VotingClassifier(estimators=[
        ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=RANDOM_STATE)),
    ], voting="soft")
    ensemble_pipe = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", ensemble)])
    ensemble_pipe.fit(X_train, y_train)
    pred = ensemble_pipe.predict(X_test)
    proba = ensemble_pipe.predict_proba(X_test)[:, 1]
    results.append({"model": "Voting Ensemble", "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0), "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0), "roc_auc": roc_auc_score(y_test, proba), "cv_roc_auc_mean": np.nan, "cv_roc_auc_std": np.nan})
    fitted["Voting Ensemble"] = ensemble_pipe

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)
    best_name = results_df.iloc[0]["model"]
    joblib.dump(fitted[best_name], output_dir / "best_churn_model.joblib")
    pd.DataFrame({"y_true": y_test, "churn_probability": fitted[best_name].predict_proba(X_test)[:, 1]}).to_csv(output_dir / "test_predictions.csv", index=False)
    return results_df, fitted[best_name]
