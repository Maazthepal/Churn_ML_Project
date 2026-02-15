from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Load the trained model
MODEL_PATH = Path("artifacts/model_trainer/model.joblib")
model = joblib.load(MODEL_PATH)

print("✅ Model loaded successfully!")

def preprocess_input(form_data):
    """Transform form data into model-ready format"""
    
    # Create dataframe from form data
    data = {
        'gender': 1 if form_data['gender'] == 'Male' else 0,
        'SeniorCitizen': int(form_data['SeniorCitizen']),
        'Partner': 1 if form_data['Partner'] == 'Yes' else 0,
        'Dependents': 1 if form_data['Dependents'] == 'Yes' else 0,
        'tenure': int(form_data['tenure']),
        'PhoneService': 1 if form_data['PhoneService'] == 'Yes' else 0,
        'MultipleLines': 1 if form_data['MultipleLines'] == 'Yes' else 0,
        'OnlineSecurity': 1 if form_data['OnlineSecurity'] == 'Yes' else 0,
        'OnlineBackup': 1 if form_data['OnlineBackup'] == 'Yes' else 0,
        'DeviceProtection': 1 if form_data['DeviceProtection'] == 'Yes' else 0,
        'TechSupport': 1 if form_data['TechSupport'] == 'Yes' else 0,
        'StreamingTV': 1 if form_data['StreamingTV'] == 'Yes' else 0,
        'StreamingMovies': 1 if form_data['StreamingMovies'] == 'Yes' else 0,
        'PaperlessBilling': 1 if form_data['PaperlessBilling'] == 'Yes' else 0,
        'MonthlyCharges': float(form_data['MonthlyCharges']),
        'TotalCharges': float(form_data['TotalCharges']),
    }
    
    df = pd.DataFrame([data])
    
    # One-hot encode categorical variables
    internet_service = form_data['InternetService']
    df['InternetService_Fiber optic'] = 1 if internet_service == 'Fiber optic' else 0
    df['InternetService_No'] = 1 if internet_service == 'No' else 0
    
    contract = form_data['Contract']
    df['Contract_One year'] = 1 if contract == 'One year' else 0
    df['Contract_Two year'] = 1 if contract == 'Two year' else 0
    
    payment = form_data['PaymentMethod']
    df['PaymentMethod_Credit card (automatic)'] = 1 if payment == 'Credit card (automatic)' else 0
    df['PaymentMethod_Electronic check'] = 1 if payment == 'Electronic check' else 0
    df['PaymentMethod_Mailed check'] = 1 if payment == 'Mailed check' else 0
    
    # Ensure correct column order (same as training)
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Preprocess
        processed_data = preprocess_input(form_data)
        
        # Predict
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        # Prepare result
        result = {
            'prediction': int(prediction),
            'churn': 'Yes - Customer will likely churn' if prediction == 1 else 'No - Customer will likely stay',
            'churn_probability': float(probability[1]) * 100,
            'no_churn_probability': float(probability[0]) * 100,
            'confidence': float(max(probability)) * 100,
            'risk_level': 'High Risk' if probability[1] > 0.7 else 'Medium Risk' if probability[1] > 0.4 else 'Low Risk'
        }
        
        return render_template('results.html', result=result, form_data=form_data)
        
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)