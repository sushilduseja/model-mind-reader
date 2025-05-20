import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Initialize FastAPI app
app = FastAPI()

def preprocess_data(data, target_column):
    """Preprocesses the data by encoding categorical columns and dropping irrelevant ones."""
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical columns
    for column in X.select_dtypes(include=['object']).columns:
        X[column] = LabelEncoder().fit_transform(X[column])

    # Ensure target column is numeric
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    return X, y

# Define request model
class TrainModelRequest(BaseModel):
    data: List[dict]
    target_column: str
    model_type: Optional[str] = "Decision Tree"

@app.post("/train_model")
def train_model(request: TrainModelRequest):
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame(request.data)
        target_column = request.target_column
        model_type = request.model_type

        # Preprocess the data
        X, y = preprocess_data(data, target_column)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Select model type
        if model_type == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_type == "Logistic Regression":
            model = LogisticRegression()
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")

        # Train the model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate performance metrics
        performance = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average='weighted'),
            "recall": recall_score(y_test, predictions, average='weighted')
        }

        return {"model_type": model_type, "performance": performance}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
