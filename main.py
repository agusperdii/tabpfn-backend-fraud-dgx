import os
from pathlib import Path

# --- AGGRESSIVE MONKEY PATCH FOR VERCEL READ-ONLY FS ---
import os
from pathlib import Path

# Redirect TabPFN to use /tmp for everything
os.environ["TABPFN_HOME"] = "/tmp/.tabpfn"
os.environ["TABPFN_MODEL_CACHE_DIR"] = "/tmp/.tabpfn/models"
os.environ["TABPFN_DATASET_CACHE_DIR"] = "/tmp/.tabpfn/datasets"

original_mkdir = os.mkdir
original_makedirs = os.makedirs

def patched_mkdir(path, mode=0o777):
    path_str = str(path)
    if "/var/task" in path_str and not path_str.startswith("/tmp"):
        print(f"🚫 Blocking mkdir on read-only path: {path}")
        return
    return original_mkdir(path, mode)

def patched_makedirs(name, mode=0o777, exist_ok=False):
    name_str = str(name)
    if "/var/task" in name_str and not name_str.startswith("/tmp"):
        print(f"🚫 Blocking makedirs on read-only path: {name}")
        return
    return original_makedirs(name, mode, exist_ok)

os.mkdir = patched_mkdir
os.makedirs = patched_makedirs
# -------------------------------------------------------

try:
    import tabpfn_client
    from tabpfn_client.service_wrapper import UserAuthenticationClient
    # Force token file to /tmp
    UserAuthenticationClient.CACHED_TOKEN_FILE = Path("/tmp/.tabpfn_token")
except Exception:
    pass

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# Import this after setting environment variables
try:
    from tabpfn_client import set_access_token
except ImportError:
    set_access_token = None

# Load environment variables from .env if present
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="TabPFN Fraud Detection API")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure TabPFN Access Token
api_key = os.getenv("PRIORLABS_API_KEY")
if api_key:
    # Set the token in environment variables as backup
    os.environ["TABPFN_TOKEN"] = str(api_key)
    
    # Always call set_access_token because our monkey patch makes it safe for Vercel (/tmp)
    try:
        if set_access_token:
            set_access_token(str(api_key))
            print("✅ TabPFN Access token configured successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not set token: {e}")
else:
    print("⚠️ PRIORLABS_API_KEY not found in environment variables")

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
