from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import lime.lime_tabular

# Initialize FastAPI app
app = FastAPI()

# Request models
class TrainModelRequest(BaseModel):
    data: list
    target_column: str
    model_type: str

class ExplainabilityRequest(BaseModel):
    data: list
    target_column: str
    model_type: str

class FairnessMetricsRequest(BaseModel):
    data: list
    target_column: str
    predictions: list

# Endpoints
@app.post("/train_model")
def train_model(request: TrainModelRequest):
    try:
        data = pd.DataFrame(request.data)
        X = data.drop(columns=[request.target_column])
        y = data[request.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if request.model_type == "Decision Tree":
            model = DecisionTreeClassifier()
        elif request.model_type == "Logistic Regression":
            model = LogisticRegression()
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        performance = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average='weighted'),
            "recall": recall_score(y_test, predictions, average='weighted')
        }

        return {"model_type": request.model_type, "performance": performance}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_explanations")
def generate_explanations(request: ExplainabilityRequest):
    try:
        data = pd.DataFrame(request.data)
        X = data.drop(columns=[request.target_column])
        y = data[request.target_column]

        # Initialize and train the model
        if request.model_type == "Decision Tree":
            model = DecisionTreeClassifier()
        elif request.model_type == "Logistic Regression":
            model = LogisticRegression()
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")

        model.fit(X, y)  # Train the model

        # Dynamically determine class names
        class_names = [str(cls) for cls in model.classes_]

        # Initialize LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=X.columns.tolist(),
            class_names=class_names,
            mode="classification"
        )

        # Generate explanation for the first instance
        explanation = explainer.explain_instance(
            X.iloc[0].values, model.predict_proba
        )

        return {"explanation": explanation.as_list()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate_fairness_metrics")
def calculate_fairness_metrics(request: FairnessMetricsRequest):
    try:
        data = pd.DataFrame(request.data)
        y_true = data[request.target_column]

        metrics = {
            "accuracy": accuracy_score(y_true, request.predictions),
            "precision": precision_score(y_true, request.predictions, average='weighted'),
            "recall": recall_score(y_true, request.predictions, average='weighted')
        }

        return {"fairness_metrics": metrics}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
