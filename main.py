import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from tabpfn_client import set_access_token

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="TabPFN Fraud Detection API")

# Configure TabPFN Access Token
api_key = os.getenv("PRIORLABS_API_KEY")
if api_key:
    set_access_token(api_key)
    print("✅ TabPFN Access token configured.")
else:
    print("⚠️ PRIORLABS_API_KEY not found in .env")

# Load Models and Scalers
try:
    model = joblib.load('TabPFN Model.pkl')
    robust_scaler = joblib.load('Robust Scaler (1).pkl')
    standard_scaler = joblib.load('Standard Scaler.pkl')
    print("✅ Models and Scalers loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    robust_scaler = None
    standard_scaler = None

class Transaction(BaseModel):
    step: int
    amount: float
    balanceDiffOrig: float
    balanceDiffDest: float
    destIsMerchant: int
    senderTxnCount: int
    receiverTxnCount: int
    type_CASH_IN: int
    type_CASH_OUT: int
    type_DEBIT: int
    type_PAYMENT: int
    type_TRANSFER: int

@app.post("/predict")
async def predict(data: Transaction):
    if model is None:
        print("gagal: Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Create DataFrame directly from the input model
        input_dict = data.dict()
            
        # Create DataFrame to match training columns order
        feature_order = [
            'step', 'amount', 'balanceDiffOrig', 'balanceDiffDest', 
            'destIsMerchant', 'type_CASH_IN', 'type_CASH_OUT', 
            'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER', 
            'senderTxnCount', 'receiverTxnCount'
        ]
        
        df = pd.DataFrame([input_dict])[feature_order]
        
        # Preprocessing
        numerical_features = ["amount", "balanceDiffOrig", "balanceDiffDest", "senderTxnCount", "receiverTxnCount"]
        
        # Apply Robust Scaler (using .values to avoid feature name warning/error)
        df[numerical_features] = robust_scaler.transform(df[numerical_features].values)
        
        # Apply Standard Scaler to 'step'
        df[['step']] = standard_scaler.transform(df[['step']].values)
        
        # Prediction
        # Convert to numpy/values to match the expected format (0-11 indices)
        probs = model.predict_proba(df.values)
        prob_fraud = float(probs[0][1])
        prediction = int(prob_fraud > 0.5)
        
        return {
            "is_fraud": prediction,
            "probability": prob_fraud,
            "model_name": "TabPFN"
        }
        
    except Exception as e:
        print(f"gagal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
