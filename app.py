from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.responses import JSONResponse

app = FastAPI()

# Load trained model
model = joblib.load("loan_model.pkl")

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

    # Get confidence score if model supports it
    try:
        confidence = float(np.max(model.predict_proba(df)))
    except AttributeError:
        confidence = None  # Model does not support predict_proba

    # Placeholder for SHAP values (can compute later)
    shap_values = {col: 0.0 for col in df.columns}  # default zero

    return {
        "eligibilityStatus": result,
        "confidence": confidence,
        "shapValues": shap_values
    }
