# backend/main.py
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from services.churn_predictor import get_model, preprocess_input
from typing import Dict

# ── This is the line that defines 'app' ──
app = FastAPI(title="Churn Prediction API")

# Allow frontend to call the backend (localhost:3000 for Next.js dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Backend is running!"}

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>Churn Prediction API</title></head>
        <body>
            <h1>Churn Prediction API is Running! 🚀</h1>
            <p>Go to <a href="/docs">/docs</a> for interactive API docs (Swagger UI)</p>
            <p>Or test the prediction endpoint via POST to <code>/predict</code></p>
        </body>
    </html>
    """

@app.post("/predict")
async def predict_churn(
    gender: str = Form(...),
    SeniorCitizen: int = Form(...),
    Partner: str = Form(...),
    Dependents: str = Form(...),
    tenure: int = Form(...),
    PhoneService: str = Form(...),
    MultipleLines: str = Form(...),
    InternetService: str = Form(...),
    OnlineSecurity: str = Form(...),
    OnlineBackup: str = Form(...),
    DeviceProtection: str = Form(...),
    TechSupport: str = Form(...),
    StreamingTV: str = Form(...),
    StreamingMovies: str = Form(...),
    Contract: str = Form(...),
    PaperlessBilling: str = Form(...),
    PaymentMethod: str = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...)
) -> Dict:
    form_data: Dict = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }
    

    try:
        model = get_model()
        processed = preprocess_input(form_data)
        prediction = int(model.predict(processed)[0])
        probability = float(model.predict_proba(processed)[0][1]) * 100

        return {
            "prediction": prediction,
            "churn": "Yes - Customer will likely churn" if prediction == 1 else "No - Customer will likely stay",
            "churn_probability": round(probability, 1),
            "risk_level": "High Risk" if probability > 70 else "Medium Risk" if probability > 40 else "Low Risk"
        }
    except Exception as e:
        return {"error": str(e)}