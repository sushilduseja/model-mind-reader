# Model Mind Reader

An ML interpretability tool with a FastAPI backend and Next.js frontend. Train models, generate LIME/SHAP explanations, and evaluate fairness metrics.

## Project Structure

```
├── app.py                 # FastAPI server entry point
├── config.py              # Constants and configuration
├── requirements.txt       # Python dependencies
├── modules/
│   ├── data_ingestion.py      # CSV upload & schema validation
│   ├── model_training.py      # Train DecisionTree/LogisticRegression
│   ├── explainability.py      # LIME & SHAP explanations
│   └── fairness_metrics.py    # Accuracy/precision/recall metrics
├── tests/                 # pytest test suite
└── front-end/             # Next.js UI (see front-end/README.md)
```

## Backend Setup

```bash
python -m pip install -r requirements.txt
python app.py
```

Available endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/train_model` | POST | Train a DecisionTree or LogisticRegression model |
| `/generate_explanations` | POST | Generate LIME explanations for predictions |
| `/calculate_fairness_metrics` | POST | Calculate accuracy, precision, recall |

## Running Tests

```bash
PYTHONPATH=. pytest tests/
```

## Frontend

See [front-end/README.md](front-end/README.md) for setup instructions.
