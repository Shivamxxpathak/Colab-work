"""Portfolio-ready evaluation charts."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_model_comparison(results: pd.DataFrame, out: str | Path) -> None:
    results = results.sort_values("roc_auc")
    plt.figure(figsize=(10, 6))
    plt.barh(results["model"], results["roc_auc"])
    plt.xlabel("ROC-AUC")
    plt.title("Customer Churn Model Comparison")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
