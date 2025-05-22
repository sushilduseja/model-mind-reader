from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import lime.lime_tabular
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import logging

def create_app() -> FastAPI:
    app = FastAPI()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # Allow requests from the frontend
        allow_credentials=True,
        allow_methods=["*"],  # Allow all HTTP methods
        allow_headers=["*"],  # Allow all headers
    )

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    # Define request models
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

    # Define endpoints
    @app.post("/train_model")
    def train_model(request: TrainModelRequest):
        try:
            # Convert input data to DataFrame
            data = pd.DataFrame(request.data)

            # Check if target column exists
            if request.target_column not in data.columns:
                raise HTTPException(status_code=400, detail="Target column not found in data")

            # Separate features and target
            X = data.drop(columns=[request.target_column])
            y = data[request.target_column]

            # Train the model
            if request.model_type == "Decision Tree":
                model = DecisionTreeClassifier()
            elif request.model_type == "Logistic Regression":
                model = LogisticRegression()
            else:
                raise HTTPException(status_code=400, detail="Invalid model type")

            model.fit(X, y)

            # Return success response
            return {"message": "Model trained successfully", "model_type": request.model_type}

        except Exception as e:
            # Log the error and return a 500 response
            logger.error("Error in /train_model", exc_info=True)
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
            logger.error("Error in /generate_explanations", exc_info=True)
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
            logger.error("Error in /calculate_fairness_metrics", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.exception_handler(Exception)
    def global_exception_handler(request, exc):
        logger.error("Unhandled exception occurred", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "details": str(exc)}
        )

    @app.exception_handler(HTTPException)
    def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail}
        )

    return app

# Create the app instance
app = create_app()
