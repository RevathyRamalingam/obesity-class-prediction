import pickle
#import numpy as np
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any

app=FastAPI()

def load_model():
    with open('model/logistic_regression_model.bin', 'rb') as f_in:
        pipeline = pickle.load(f_in)
    return pipeline


pipeline = load_model()
    
def predict_obesity(customer, pipeline):
    y_pred = pipeline.predict(customer)
    y_pred_proba = pipeline.predict_proba(customer)
    return y_pred, y_pred_proba

customer = {
  "age": 22.956845,
  "gender": "female",
  "height": 1.618686,
  "weight": 81.281578,
  "calc": "no",
  "favc": "yes",
  "fcvc": 2.396265,
  "ncp": 1.073421,
  "scc": "no",
  "smoke": "no",
  "ch2o": 1.979833,
  "family_history_with_overweight": "yes",
  "faf": 0.022598,
  "tue": 0.061282,
  "caec": "sometimes",
  "mtrans": "public_transportation"
}


@app.post("/predict")
def make_prediction(customer: Dict[str, Any]):
    ypred,y_pred_proba = predict_obesity(customer,pipeline)
    classes = ['insufficient_weight', 'normal_weight', 'obesity_type_i',
       'obesity_type_ii', 'obesity_type_iii', 'overweight_level_i',
       'overweight_level_ii']
    return {
        "Health_category": ypred[0],
        "probabilities": {c: round(y_pred_proba[0][classes.index(c)]*100, 4) for c in classes}
    }
if __name__== "__main__":
    uvicorn.run(app,host="localhost",port=9696)
