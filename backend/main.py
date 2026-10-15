from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.predictor import CHURN_THRESHOLD, ChurnPredictor
from backend.schemas import ChurnPredictionRequest, ChurnPredictionResponse

app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI backend for predicting customer churn using a trained ML model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ChurnPredictor()


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "churn-prediction-api"}


@app.post("/api/predict", response_model=ChurnPredictionResponse)
def predict_churn(payload: ChurnPredictionRequest) -> ChurnPredictionResponse:
    try:
        prediction, churn_probability = predictor.predict(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return ChurnPredictionResponse(
        prediction=prediction,
        churn_probability=churn_probability,
        threshold=CHURN_THRESHOLD,
    )
