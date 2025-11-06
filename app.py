from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.responses import JSONResponse
import shap

app = FastAPI()

# Load trained model
model = joblib.load("loan_model.pkl")

# Load training data for SHAP background (add your actual training data file)
try:
    train_data = pd.read_csv("train_u6lujuX_CVtuZ9i")  # Use your processed training file
    background = shap.sample(train_data.drop('Loan_Status', axis=1, errors='ignore'), 100)
except:
    background = None

# Root route for health check
@app.get("/")
def home():
    return JSONResponse(content={"message": "Loan Eligibility API is running ✅"})

# Define input schema
class LoanInput(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

@app.post("/predict")
def predict(data: LoanInput):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Feature engineering
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanAmount_log"] = np.log(df["LoanAmount"] + 1)
    df["Total_Income_log"] = np.log(df["Total_Income"] + 1)

    # Encode categorical variables
    df = pd.get_dummies(df)

    # Match model training columns
    model_features = list(model.feature_names_in_)
    df = df.reindex(columns=model_features, fill_value=0)

    # Predict
    prediction = model.predict(df)[0]
    result = "Approved" if prediction == 1 else "Not Approved"

    # Get confidence score
    try:
        confidence = float(np.max(model.predict_proba(df)))
    except AttributeError:
        confidence = None

    # Compute SHAP values with TreeExplainer
    try:
        explainer = shap.TreeExplainer(model, background if background is not None else df)
        shap_values_raw = explainer.shap_values(df)
        
        # Handle both binary and multiclass output
        if isinstance(shap_values_raw, list):
            shap_vals = shap_values_raw[1][0]  # Use positive class
        else:
            shap_vals = shap_values_raw[0]
            
        shap_values = {
            feature: float(shap_vals[i])
            for i, feature in enumerate(df.columns)
        }
    except Exception as e:
        print(f"SHAP error: {e}")
        shap_values = {feature: 0.0 for feature in df.columns}

    return {
        "eligibilityStatus": result,
        "confidence": confidence,
        "shapValues": shap_values
    }
