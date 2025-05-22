from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from typing import List

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
        metrics = {
            "accuracy": accuracy_score(y_true, request.predictions),
            "precision": precision_score(y_true, request.predictions, average='weighted'),
            "recall": recall_score(y_true, request.predictions, average='weighted')
        }

        return {"fairness_metrics": metrics}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def validate_protected_attribute(data: pd.DataFrame, protected_attribute: str) -> None:
    """Validate that the protected attribute exists in the dataset."""
    if protected_attribute not in data.columns:
        raise ValueError(f"Protected attribute '{protected_attribute}' not found in dataset.")

def compute_rate_safe(numerator: int, denominator: int) -> float:
    """Compute rates safely, avoiding division by zero."""
    return numerator / denominator if denominator != 0 else 0
