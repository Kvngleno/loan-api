from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import math
from fastapi.responses import JSONResponse
import shap

app = FastAPI()

# Load trained model
model = joblib.load("loan_model.pkl")

# Load background data for SHAP
try:
    train_data = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
    if 'Loan_Status' in train_data.columns:
        background = train_data.drop('Loan_Status', axis=1)
    else:
        background = train_data.copy()
    background = pd.get_dummies(background)
except Exception as e:
    print(f"Background data load error: {e}")
    background = None


@app.get("/")
def home():
    return JSONResponse(content={"message": "Loan Eligibility API is running ✅"})


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
    df = pd.DataFrame([data.dict()])

    # Feature engineering
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanAmount_log"] = np.log(df["LoanAmount"] + 1)
    df["Total_Income_log"] = np.log(df["Total_Income"] + 1)

    # Encode categorical variables
    df = pd.get_dummies(df)

    # Align with model features
    model_features = list(model.feature_names_in_)
    df = df.reindex(columns=model_features, fill_value=0)

    # Make prediction
    prediction = model.predict(df)[0]
    result = "Approved" if prediction == 1 else "Not Approved"

    # Confidence
    try:
        confidence = float(np.max(model.predict_proba(df)))
    except AttributeError:
        confidence = None

    # ✅ Fix SHAP explanation
    shap_values = {}
    try:
        if hasattr(model, "coef_"):
            # Align background perfectly
            if background is not None:
                background_aligned = background.reindex(columns=model_features, fill_value=0)
                explainer = shap.LinearExplainer(model, masker=background_aligned)
            else:
                explainer = shap.LinearExplainer(model, masker=df)

            shap_vals = explainer(df)
            raw_values = shap_vals.values[0] if hasattr(shap_vals, "values") else shap_vals[0]
            shap_values = {
                feature: float(raw_values[i]) if not (math.isnan(raw_values[i]) or math.isinf(raw_values[i])) else 0.0
                for i, feature in enumerate(df.columns)
            }

        else:
            # For tree models
            background_aligned = background.reindex(columns=model_features, fill_value=0) if background is not None else df
            explainer = shap.TreeExplainer(model, background_aligned)
            shap_vals = explainer.shap_values(df)
            shap_output = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
            shap_values = {
                feature: float(shap_output[i]) if not (math.isnan(shap_output[i]) or math.isinf(shap_output[i])) else 0.0
                for i, feature in enumerate(df.columns)
            }

        # ✅ Sort SHAP values by absolute magnitude (so you can see strongest impact)
        shap_values = dict(sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True))

    except Exception as e:
        print(f"SHAP error: {e}")
        shap_values = {feature: 0.0 for feature in df.columns}

    return {
        "eligibilityStatus": result,
        "confidence": confidence,
        "shapValues": shap_values
    }

