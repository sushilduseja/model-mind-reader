import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Initialize FastAPI app (if not already initialized in another module)
app = FastAPI()


# Define request model
class FairnessMetricsRequest(BaseModel):
    data: list  # List of dictionaries representing rows of data
    target_column: str
    predictions: list  # List of model predictions


@app.post("/calculate_fairness_metrics")
def calculate_fairness_metrics(request: FairnessMetricsRequest):
    try:
        # Convert data to DataFrame
        data = pd.DataFrame(request.data)
        y_true = data[request.target_column]

        # Calculate fairness metrics
        preds = request.predictions
        avg = "weighted"
        metrics = {
            "accuracy": accuracy_score(y_true, preds),
            "precision": precision_score(y_true, preds, average=avg),
            "recall": recall_score(y_true, preds, average=avg),
        }

        return {"fairness_metrics": metrics}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def validate_protected_attribute(data: pd.DataFrame, attribute: str) -> None:
    """Check the protected attribute exists in the dataset."""
    if attribute not in data.columns:
        msg = f"Protected attribute '{attribute}' not found in dataset."
        raise ValueError(msg)


def compute_rate_safe(numerator: int, denominator: int) -> float:
    """Compute a rate, returning 0 on zero denominator."""
    return numerator / denominator if denominator != 0 else 0
