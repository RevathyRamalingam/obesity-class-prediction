import pickle
from fastapi import FastAPI, HTTPException
import uvicorn
from typing import Dict, Literal
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
import numpy as np
import pandas as pd

# Pydantic model for the customer data
class Customer(BaseModel):
    gender: Literal["male", "female"]
    family_history_with_overweight: Literal["yes", "no"]
    favc: Literal["yes", "no"]
    scc: Literal["yes", "no"]
    smoke: Literal["yes", "no"]
    caec: Literal["no", "sometimes", "frequently", "always"]
    calc: Literal["no", "sometimes", "frequently", "always"]
    mtrans: Literal[
        "public_transportation",
        "automobile",
        "walking",
        "motorbike",
        "bike"
    ]
    age: int = Field(..., ge=5)
    height: float = Field(..., ge=0)
    weight: float = Field(..., ge=3)
    fcvc: float = Field(..., ge=1.0, le=3.0)
    ncp: float = Field(..., ge=1.0, le=4.0)
    ch2o: float = Field(..., ge=1.0, le=3.0)
    faf: float = Field(..., ge=0.0, le=3.0)
    tue: float = Field(..., ge=0.0, le=2.0)

    # Pydantic v2.0 uses model_config and ConfigDict to enforce validation for extra fields
    model_config = ConfigDict(extra='forbid')  # Rejects any extra fields in the input JSON

# Response model for the prediction result
class PredictResponse(BaseModel):
    Health_category: str  # Predicted class (e.g., "obese", "overweight", etc.)
    probabilities: Dict[str, float]  # Dictionary of class probabilities

# Initialize FastAPI app
app = FastAPI(title="Obesity Prediction API", version="1.0")

# Load the pre-trained model
def load_model():
    with open('logistic_regression_model.bin', 'rb') as f_in:
        pipeline = pickle.load(f_in)
    return pipeline

# Load the model when the app starts
pipeline = load_model()

# Prediction function
def predict_obesity(customer ,pipeline) -> (np.ndarray, np.ndarray):
    y_pred = pipeline.predict(customer)  # Wrap it in a list to match the input shape
    y_pred_proba = pipeline.predict_proba(customer)
    return y_pred, y_pred_proba


@app.post("/predict", response_model=PredictResponse)
def make_prediction(customer: Customer) -> PredictResponse:
    try:
        ypred, y_pred_proba = predict_obesity(customer.model_dump(), pipeline)

        classes = ['insufficient_weight', 'normal_weight', 'obesity_type_i',
                   'obesity_type_ii', 'obesity_type_iii', 'overweight_level_i',
                   'overweight_level_ii']

        if isinstance(ypred[0], str):  # If it's already a class label
            predicted_class = ypred[0]  # Directly use the class label
        else:  # If it's an index, map it to the class label
            predicted_class = classes[ypred[0]]

        class_probabilities = {
            classes[i]: round(y_pred_proba[0][i] * 100, 4) for i in range(len(classes))
        }

        response = PredictResponse(
            Health_category=predicted_class,
            probabilities=class_probabilities
        )

        return response

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )


# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9696)
