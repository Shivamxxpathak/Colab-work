"""
Support Vector Machine (SVM) Practice
=====================================
Hands-on practice covering:
- Linear SVM classification
- Feature scaling
- Train/test split
- Accuracy and classification report
- Confusion matrix
- Hyperparameter tuning with GridSearchCV

Dataset: Breast Cancer dataset from scikit-learn
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scaling is important for SVM because the optimization depends on distances.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=1.0))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("=== Linear SVM ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=data.target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Compare multiple SVM configurations using cross-validation.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC())
    ])

    param_grid = {
        "svm__kernel": ["linear", "rbf"],
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", "auto"],
    }

    search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")
    search.fit(X_train, y_train)

    print("\n=== Hyperparameter Tuning ===")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Test accuracy: {search.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
