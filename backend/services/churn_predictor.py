# backend/services/churn_predictor.py
import pandas as pd
from pathlib import Path
import joblib

# Load model once (global, loaded on startup)
MODEL_PATH = Path("../artifacts/model_trainer/model.joblib")

def get_model():
    """Load model lazily on first call"""
    if not hasattr(get_model, "model"):
        get_model.model = joblib.load(MODEL_PATH)
    return get_model.model

# backend/services/churn_predictor.py (replace the whole function)

def preprocess_input(form_data: dict) -> pd.DataFrame:
    """Transform form data into model-ready format - robust version"""
    
    # Normalize keys and values (case insensitive, defaults)
    def get_val(key, default=None):
        val = form_data.get(key.lower(), default)  # lowercase key
        if val is None:
            return default
        if isinstance(val, str):
            val = val.strip().title()  # e.g. 'yes' → 'Yes', 'male' → 'Male'
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
    
    # One-hot encode (make case insensitive)
    internet_service = get_val('InternetService', 'No').title()
    df['InternetService_Fiber optic'] = 1 if internet_service == 'Fiber Optic' else 0
    df['InternetService_No'] = 1 if internet_service == 'No' else 0
    
    contract = get_val('Contract', 'Month-to-month').title()
    df['Contract_One year'] = 1 if contract == 'One Year' else 0
    df['Contract_Two year'] = 1 if contract == 'Two Year' else 0
    
    payment = get_val('PaymentMethod', '').title()
    df['PaymentMethod_Credit card (automatic)'] = 1 if 'Credit Card' in payment else 0
    df['PaymentMethod_Electronic check'] = 1 if 'Electronic' in payment else 0
    df['PaymentMethod_Mailed check'] = 1 if 'Mailed' in payment else 0
    
    # Expected columns
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