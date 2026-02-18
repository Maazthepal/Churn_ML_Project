import pandas as pd
from pathlib import Path
import joblib
import os
import requests

# ─── Configuration ────────────────────────────────────────────────────────────
# Set this environment variable in Koyeb dashboard
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://your-bucket.s3.amazonaws.com/model.joblib"  # Replace with your actual URL
)

LOCAL_MODEL_PATH = Path("/tmp/model.joblib")  # Use /tmp for serverless/container environments

# ─── Model Loading ────────────────────────────────────────────────────────────
def download_model():
    """Download model from cloud storage if not already cached"""
    if LOCAL_MODEL_PATH.exists():
        print(f"✅ Model already cached at {LOCAL_MODEL_PATH}")
        return
    
    print(f"📥 Downloading model from {MODEL_URL}...")
    try:
        response = requests.get(MODEL_URL, timeout=60)
        response.raise_for_status()
        
        # Ensure directory exists
        LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        LOCAL_MODEL_PATH.write_bytes(response.content)
        print(f"✅ Model downloaded successfully ({len(response.content) / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise RuntimeError(f"Could not download model from {MODEL_URL}") from e

def get_model():
    """Load model lazily on first call"""
    if not hasattr(get_model, "model"):
        # Download if needed
        if not LOCAL_MODEL_PATH.exists():
            download_model()
        
        # Load model
        print(f"📦 Loading model from {LOCAL_MODEL_PATH}...")
        get_model.model = joblib.load(LOCAL_MODEL_PATH)
        print(f"✅ Model loaded: {type(get_model.model).__name__}")
    
    return get_model.model

# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_input(form_data: dict) -> pd.DataFrame:
    """Transform form data into model-ready format"""
    
    # Normalize keys and values (case insensitive, defaults)
    def get_val(key, default=None):
        val = form_data.get(key, form_data.get(key.lower(), default))
        if val is None:
            return default
        if isinstance(val, str):
            val = val.strip().title()
        return val

    data = {
        'gender': 1 if get_val('gender') == 'Male' else 0,
        'SeniorCitizen': int(get_val('SeniorCitizen', 0)),
        'Partner': 1 if get_val('Partner') == 'Yes' else 0,
        'Dependents': 1 if get_val('Dependents') == 'Yes' else 0,
        'tenure': int(get_val('tenure', 0)),
        'PhoneService': 1 if get_val('PhoneService') == 'Yes' else 0,
        'MultipleLines': 1 if get_val('MultipleLines') == 'Yes' else 0,
        'OnlineSecurity': 1 if get_val('OnlineSecurity') == 'Yes' else 0,
        'OnlineBackup': 1 if get_val('OnlineBackup') == 'Yes' else 0,
        'DeviceProtection': 1 if get_val('DeviceProtection') == 'Yes' else 0,
        'TechSupport': 1 if get_val('TechSupport') == 'Yes' else 0,
        'StreamingTV': 1 if get_val('StreamingTV') == 'Yes' else 0,
        'StreamingMovies': 1 if get_val('StreamingMovies') == 'Yes' else 0,
        'PaperlessBilling': 1 if get_val('PaperlessBilling') == 'Yes' else 0,
        'MonthlyCharges': float(get_val('MonthlyCharges', 0.0)),
        'TotalCharges': float(get_val('TotalCharges', 0.0)),
    }
    
    df = pd.DataFrame([data])
    
    # One-hot encode
    internet_service = get_val('InternetService', 'No').title()
    df['InternetService_Fiber optic'] = 1 if 'Fiber' in internet_service else 0
    df['InternetService_No'] = 1 if internet_service == 'No' else 0
    
    contract = get_val('Contract', 'Month-to-month').title()
    df['Contract_One year'] = 1 if 'One' in contract else 0
    df['Contract_Two year'] = 1 if 'Two' in contract else 0
    
    payment = get_val('PaymentMethod', '').title()
    df['PaymentMethod_Credit card (automatic)'] = 1 if 'Credit Card' in payment else 0
    df['PaymentMethod_Electronic check'] = 1 if 'Electronic' in payment else 0
    df['PaymentMethod_Mailed check'] = 1 if 'Mailed' in payment else 0
    
    # Expected columns (must match training)
    expected_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'InternetService_Fiber optic', 'InternetService_No',
        'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check'
    ]
    
    df = df[expected_cols]
    
    return df