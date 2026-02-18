from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from services.churn_predictor import get_model, preprocess_input
from typing import Dict

app = FastAPI(
    title="Churn Prediction API",
    description="ML-powered customer churn prediction with cloud-hosted model",
    version="1.0.0"
)

# CORS - allow your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-vercel-app.vercel.app",  # Replace with your actual Vercel URL
        "*"  # Remove this in production, be specific
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup Event: Preload Model ────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Download and load model on startup to avoid cold start delays"""
    try:
        print("🚀 Starting up FastAPI server...")
        model = get_model()  # This triggers download if needed
        print(f"✅ Model ready: {type(model).__name__}")
        print("✅ Server ready to accept requests")
    except Exception as e:
        print(f"❌ CRITICAL: Failed to load model on startup")
        print(f"Error: {e}")
        # Don't raise - let server start so /health still works
        # But predictions will fail until model loads

# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Check if API and model are ready"""
    try:
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_type": type(model).__name__
        }
    except Exception as e:
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": str(e)
        }

# ─── Root Endpoint ────────────────────────────────────────────────────────────
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>StayFlow Churn Prediction API</title>
            <style>
                body { font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; }
                code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
                .status { display: inline-block; padding: 4px 12px; border-radius: 4px; }
                .healthy { background: #d4edda; color: #155724; }
            </style>
        </head>
        <body>
            <h1>🚀 StayFlow Churn Prediction API</h1>
            <p class="status healthy">✓ Running on Koyeb</p>
            
            <h2>Endpoints:</h2>
            <ul>
                <li><code>GET /health</code> - Check API and model status</li>
                <li><code>POST /predict</code> - Get churn prediction (19 fields required)</li>
                <li><code>GET /docs</code> - Interactive API documentation (Swagger UI)</li>
            </ul>
            
            <h2>Quick Links:</h2>
            <p>
                <a href="/docs">📖 API Docs (Swagger)</a> | 
                <a href="/health">🏥 Health Check</a>
            </p>
            
            <hr>
            <p style="color: #666; font-size: 14px;">
                Powered by FastAPI + Scikit-learn | Model hosted on cloud storage
            </p>
        </body>
    </html>
    """

# ─── Prediction Endpoint ──────────────────────────────────────────────────────
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
    """
    Predict customer churn probability
    
    Returns:
        - prediction: 0 (will stay) or 1 (will churn)
        - churn: Human-readable prediction
        - churn_probability: Percentage (0-100)
        - risk_level: High Risk / Medium Risk / Low Risk
    """
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
        return {
            "error": str(e),
            "message": "Prediction failed. Check model and input data."
        }