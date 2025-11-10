# 🧠 Obesity Multiclass Classification Problem

Obesity is a significant health issue affecting millions globally. This project leverages machine learning to classify individuals into various obesity categories based on multiple features. By analyzing factors such as age, gender, physical activity, diet, and medical history, the model aims to predict whether a person is underweight, normal weight, overweight, or obese.

## 🎯 Problem Statement

Given a set of features (e.g., age, gender, lifestyle habits, diet), the goal is to predict one of the following obesity categories:

| Class | Category             |
|-------|----------------------|
| 0     | insufficient_weight  |
| 1     | normal_weight        |
| 2     | obesity_type_i       |
| 3     | obesity_type_ii      |
| 4     | obesity_type_iii     |
| 5     | overweight_level_i   |
| 6     | overweight_level_ii  |

This is a **multiclass classification** problem where the model must predict one of the seven classes.

## 📊 Approach

The classification task is addressed using several machine learning algorithms and evaluated using relevant performance metrics.

## 📁 Project Structure

The project folder consists of the following files and directories:

### Root-level Files:
- `.python_version` — Specifies the Python version for reproducibility.
- `Dockerfile` — Used for deploying the FASTAPI ObesityPrediction service in Docker.
- `main.py` — Contains the FASTAPI app to make predictions, similar to `predict.py`.
- `requirements.txt` — Lists all the required Python libraries (e.g., pandas, numpy, scikit-learn, fastapi, pydantic) for cloud deployment on Render.
- `train.py` — Saves the final model as `logistic_regression_model.bin` in the `model` directory after dataset cleaning, hyperparameter tuning, and model training (done in the Jupyter notebook `notebook.ipynb`).
- `notebook.ipynb` — Jupyter notebook with the following steps:
  - Fetch dataset, preprocess, and clean data.
  - Perform correlation and mutual information analysis.
  - Check target distribution.
  - Hyperparameter tuning with cross-validation.
  - Evaluate models like Logistic Regression, Decision Tree, and Random Forest.
- `uv.lock` — Locks the specific versions of dependencies to ensure reproducibility.
- `pyproject.toml` — Contains the dependencies for the Uvicorn project.
- `ObesityDataSet.csv` — UCI obesity dataset.

### Files Inside Folders:
- **model/**: Contains the pickle-saved model named `logistic_regression_model.bin`.
- **data/**: Contains the UCI obesity dataset `ObesityDataSet.csv`.
- **visual_graphs/**:contains graphs to evaluate and compare the metric such as precision, recall, f1, correlation, mi scores, accuracy and roc_auc scores
contains distribution of target graph, correlation matrix, Confusion matrix, DecisionTree Hyperparameter tuning graphs and random Forest  Hyperparameter tuning heatmap and Model comparision.
- **output_screenshots/** : contains the screenshots of output
---
🎥 Demo Videos on Cloud Deployment

Part 1: Cloud Deployment Overview and testing[https://www.loom.com/share/6632698ca88f424c886b87c38bfc676a]

Part 2: Pydantic validation on API[https://www.loom.com/share/4cefb723b71d4d3088d9f00128ad1358]

---
TRAINING THE MODEL

You can run the notebook.py script to see how the model is trained and hyperparameter tuning is done for several algorithms such as LinearRegression, DecisionTree and randomForestRegressor.
> python notebook.py
---

## 🛠 Steps to Run the Project Locally

### Option 1: Using Docker

1. Clone the repository:
    ```bash
    git clone https://github.com/RevathyRamalingam/obesity-class-prediction.git
    ```

2. Build the Docker image:
    ```bash
    docker build -t obesity-prediction .
    ```

3. Run the Docker container:
    ```bash
    docker run -it -p 9696:9696 obesity-prediction:latest
    ```

4. Open your browser and navigate to:
    ```
    http://127.0.0.1:9696/docs
    ```

5. Provide the following JSON input to test the prediction:
    ```json
    {
      "age": 39,
      "gender": "female",
      "height": 1.51,
      "weight": 72,
      "calc": "no",
      "favc": "yes",
      "fcvc": 2.396265,
      "ncp": 1.073421,
      "scc": "no",
      "smoke": "no",
      "ch2o": 1.5,
      "family_history_with_overweight": "yes",
      "faf": 0.022598,
      "tue": 0.061282,
      "caec": "sometimes",
      "mtrans": "automobile"
    }
    ```

6. The output will look like this:
    ```json
    {
      "Health_category": "overweight_level_ii",
      "probabilities": {
        "insufficient_weight": 0,
        "normal_weight": 0.1317,
        "obesity_type_i": 31.7571,
        "obesity_type_ii": 0,
        "obesity_type_iii": 0,
        "overweight_level_i": 4.5125,
        "overweight_level_ii": 63.5986
      }
    }
    ```

Alternatively, you can also use a **curl** command to see the output in the CLI:
```bash
curl -X 'POST' \
  'http://localhost:9696/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "age":39,
    "gender": "female",
    "height": 1.51,
    "weight": 72,
    "calc": "no",
    "favc": "yes",
    "fcvc": 2.396265,
    "ncp": 1.073421,
    "scc": "no",
    "smoke": "no",
    "ch2o": 1.5,
    "family_history_with_overweight": "yes",
    "faf": 0.022598,
    "tue": 0.061282,
    "caec": "sometimes",
    "mtrans": "automobile"
  }'
Option 2: Using Uvicorn
Clone the repository:

bash

git clone https://github.com/RevathyRamalingam/obesity-class-prediction.git
cd obesity-class-prediction
Install Uvicorn and FastAPI:
pip install uvicorn fastapi
Run the Uvicorn server:
uvicorn main:app --host 0.0.0.0 --port 9696
Open your browser and go to:

http://127.0.0.1:9696/docs
Provide the same JSON input as mentioned earlier to get the obesity prediction.

☁️ Steps to Deploy the Project in the Cloud (Render)
Go to Render.

Sign in to your account or create a new one.

Once logged in, go to the Dashboard on Render.

Click the New button in the top-right corner and select Web Service (or another appropriate service type).

You'll be prompted to Connect GitHub to Render if you haven't done so already. Follow the steps to authorize Render to access your GitHub repositories.

After connecting GitHub, select the repository:

https://github.com/RevathyRamalingam/obesity-class-prediction
In Settings, fill in the following:

General:

Name: Choose a name for your project.

Region: Select the region closest to your geographical location.

Deploy & Build:

Repository: https://github.com/RevathyRamalingam/obesity-class-prediction

Branch: master

Build Command: echo "Hello"

Start Command: pip install -r requirements.txt && uvicorn main:app --host 0.0.0.0 --port 9696 --reload

Choose the default settings for the rest of the fields.

After deployment, you will see the URL where the service is running in the cloud.

Click on the link to access the Prediction Service and test it with the sample JSON input provided earlier.

Error Handling (Example)
If you provide an extra field in the input JSON, you'll receive a 422 Unprocessable Entity error:

{
  "detail": [
    {
      "type": "extra_forbidden",
      "loc": ["body", "new"],
      "msg": "Extra inputs are not permitted",
      "input": 9
    }
  ]
}
📝 Conclusion
This project demonstrates how to use machine learning to predict obesity categories using multiclass classification model based on various health and lifestyle factors. It provides a simple web API using FastAPI that can be deployed locally with Docker or in the cloud with Render.

