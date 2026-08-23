# Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` inside `data/`, then:

```bash
python scripts/train.py
streamlit run app.py
```
