"""Train the churn models from the command line."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.churn_pipeline import load_telco_data, train_and_evaluate
from src.visualize import plot_model_comparison
DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT = ROOT / "outputs"
if __name__ == "__main__":
    if not DATA.exists():
        raise SystemExit(f"Dataset not found. Save the Telco CSV as {DATA}")
    df = load_telco_data(DATA)
    results, _ = train_and_evaluate(df, OUTPUT)
    plot_model_comparison(results, OUTPUT / "model_comparison.png")
    print("\nModel comparison:\n")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest model: {results.iloc[0]['model']}")
