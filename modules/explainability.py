from typing import List

import lime.lime_tabular
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Initialize FastAPI app (if not already initialized in another module)
app = FastAPI()


def preprocess_data_for_explainer(data, target_column):
    """Preprocess data to numeric form for LIME/SHAP explainers."""
    data = data.copy()
    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = LabelEncoder().fit_transform(data[column])

    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


# Define request model
class ExplainabilityRequest(BaseModel):
    data: list  # List of dictionaries representing rows of data
    target_column: str
    model_type: str  # "Decision Tree" or "Logistic Regression"


@app.post("/generate_explanations")
def generate_explanations(request: ExplainabilityRequest):
    try:
        # Convert data to DataFrame
        data = pd.DataFrame(request.data)

        # Preprocess the data
        X, y = preprocess_data_for_explainer(data, request.target_column)

        # Load the model (this assumes a pre-trained model is available)
        if request.model_type == "Decision Tree":
            model = DecisionTreeClassifier()
        elif request.model_type == "Logistic Regression":
            model = LogisticRegression()
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")

        # Generate explanations using LIME or SHAP
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=X.columns.tolist(),
            class_names=["Class 0", "Class 1"],
            mode="classification",
        )
        proba = model.predict_proba
        explanation = explainer.explain_instance(X.iloc[0].values, proba)

        return {"explanation": explanation.as_list()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def validate_feature_alignment(
    training_features: List[str], explanation_features: List[str]
) -> None:
    """Check explanation input features match the training features."""
    if set(training_features) != set(explanation_features):
        raise ValueError("Feature names do not match training features.")
    if len(training_features) != len(explanation_features):
        raise ValueError("Feature dimensions mismatch.")
