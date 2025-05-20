import streamlit as st
import lime.lime_tabular
import shap
import numpy as np
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Initialize FastAPI app (if not already initialized in another module)
app = FastAPI()

def preprocess_data_for_explainer(data, target_column):
    """Preprocesses data to ensure it is numeric for LIME and SHAP explainers."""
    data = data.copy()
    for column in data.select_dtypes(include=['object']).columns:
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
        explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names=X.columns.tolist(), class_names=["Class 0", "Class 1"], mode="classification")
        explanation = explainer.explain_instance(X.iloc[0].values, model.predict_proba)

        return {"explanation": explanation.as_list()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_explanations(model, data, predictions):
    """Generates LIME and SHAP explanations."""
    st.subheader("LIME Explanations")

    # Preprocess data for LIME
    numeric_data = preprocess_data_for_explainer(data)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(numeric_data),
        feature_names=numeric_data.columns,
        class_names=["Class 0", "Class 1"],
        mode="classification"
    )

    instance_idx = st.number_input("Enter the index of the instance to explain", min_value=0, max_value=len(data)-1, step=1)
    explanation = lime_explainer.explain_instance(numeric_data.iloc[instance_idx].values, model.predict_proba)
    st.write(explanation.as_list())

    st.subheader("SHAP Summary Plot")
    shap_explainer = shap.Explainer(model, numeric_data)
    shap_values = shap_explainer(numeric_data)
    shap.summary_plot(shap_values, numeric_data, show=False)
    st.pyplot(bbox_inches='tight')
