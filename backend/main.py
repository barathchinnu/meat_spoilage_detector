import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mutton-freshness-backend")

app = FastAPI(
    title="Mutton Freshness Monitoring System",
    description="IoT and Machine Learning API for real-time mutton spoilage detection using ESP32, DHT22, and MQ-135 sensors.",
    version="2.0.0"
)

# Allow dashboard and external clients to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# ML Model Loading
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "freshness_model.pkl"

model: Any = None
feature_names: List[str] = ["temperature", "humidity", "mq135"]
classes: List[str] = ["Fresh", "Normal", "Spoilage"]
target_column: str = "stage"
metrics: Dict[str, Any] = {}
model_loaded: bool = False

try:
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        if isinstance(bundle, dict) and "model" in bundle:
            model = bundle["model"]
            feature_names = bundle.get("feature_names", feature_names)
            classes = [str(c) for c in bundle.get("classes", getattr(model, "classes_", classes))]
            target_column = bundle.get("target_column", target_column)
            metrics = bundle.get("metrics", {})
        else:
            model = bundle
            feature_names = list(getattr(model, "feature_names_in_", feature_names))
            classes = [str(c) for c in getattr(model, "classes_", classes)]
        model_loaded = True
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        logger.info(f"Feature order: {feature_names}")
        logger.info(f"Classes: {classes}")
    else:
        logger.error(f"Model file not found at: {MODEL_PATH}")
except Exception as e:
    logger.exception(f"Error loading model from {MODEL_PATH}: {e}")
    model_loaded = False


# -----------------------------
# Schemas
# -----------------------------
class SensorData(BaseModel):
    temperature: float = Field(..., description="Temperature in Celsius from DHT22")
    humidity: float = Field(..., description="Relative humidity in percentage from DHT22")
    mq135: int = Field(..., description="MQ-135 gas sensor raw ADC value or PPM")

    model_config = {
        "json_schema_extra": {
            "example": {
                "temperature": 30.5,
                "humidity": 68.5,
                "mq135": 1150
            }
        }
    }


# Initial state for latest sensor reading
latest_data: Dict[str, Any] = {
    "temperature": 0.0,
    "humidity": 0.0,
    "mq135": 0,
    "prediction": "Awaiting Data",
    "stage": "Awaiting Data",
    "confidence": 0.0,
    "probabilities": {cls: 0.0 for cls in classes},
    "recommendation": "Waiting for sensor readings from ESP32...",
    "severity": "unknown",
    "color": "#64748B",
    "timestamp": None
}


# -----------------------------
# Inference Helper
# -----------------------------
def predict_freshness(temperature: float, humidity: float, mq135: int) -> Dict[str, Any]:
    """
    Generate freshness prediction and class probabilities using the loaded ML model.
    """
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded. Please ensure {MODEL_PATH.name} exists in backend/model/."
        )

    # DataFrame with explicit column names matching training features
    input_df = pd.DataFrame(
        [[float(temperature), float(humidity), int(mq135)]],
        columns=feature_names
    )

    predicted_stage = str(model.predict(input_df)[0])
    raw_probabilities = model.predict_proba(input_df)[0]
    model_classes = [str(c) for c in getattr(model, "classes_", classes)]

    prob_dict = {
        cls: round(float(prob), 4)
        for cls, prob in zip(model_classes, raw_probabilities)
    }

    max_prob = float(max(raw_probabilities))
    confidence_pct = round(max_prob * 100, 2)

    if predicted_stage.lower() == "fresh":
        recommendation = "Mutton is fresh and safe for consumption. Store properly under refrigeration."
        severity = "safe"
        color = "#10B981"
    elif predicted_stage.lower() == "normal":
        recommendation = "Freshness is acceptable. Consume promptly or freeze immediately."
        severity = "warning"
        color = "#F59E0B"
    else:  # Spoilage
        recommendation = "Spoilage detected! High bacterial decomposition and gas emission. Do NOT consume."
        severity = "danger"
        color = "#EF4444"

    return {
        "prediction": predicted_stage,
        "stage": predicted_stage,
        "confidence": confidence_pct,
        "probabilities": prob_dict,
        "recommendation": recommendation,
        "severity": severity,
        "color": color
    }


# -----------------------------
# Endpoints
# -----------------------------

@app.get("/")
def home():
    """API health check and information"""
    return {
        "project": "Mutton Freshness Monitoring System",
        "status": "running",
        "model_loaded": model_loaded,
        "model_file": MODEL_PATH.name,
        "feature_names": feature_names,
        "classes": classes,
        "metrics": metrics,
        "docs_url": "/docs"
    }


@app.get("/model-info")
def get_model_info():
    """Retrieve detailed model architecture, features, and evaluation metrics"""
    return {
        "model_loaded": model_loaded,
        "model_type": type(model).__name__ if model else None,
        "feature_names": feature_names,
        "target_column": target_column,
        "classes": classes,
        "metrics": metrics,
        "path": str(MODEL_PATH)
    }


@app.post("/sensor")
def receive_sensor_data(data: SensorData):
    """
    Receive live telemetry from ESP32, run ML classification immediately,
    and update the latest status.
    """
    try:
        prediction_info = predict_freshness(data.temperature, data.humidity, data.mq135)
    except Exception as e:
        logger.error(f"Prediction error during /sensor ingest: {e}")
        prediction_info = {
            "prediction": "Error",
            "stage": "Error",
            "confidence": 0.0,
            "probabilities": {},
            "recommendation": f"Model inference error: {str(e)}",
            "severity": "danger",
            "color": "#EF4444"
        }

    latest_data["temperature"] = round(data.temperature, 2)
    latest_data["humidity"] = round(data.humidity, 2)
    latest_data["mq135"] = int(data.mq135)
    latest_data["prediction"] = prediction_info["prediction"]
    latest_data["stage"] = prediction_info["stage"]
    latest_data["confidence"] = prediction_info["confidence"]
    latest_data["probabilities"] = prediction_info["probabilities"]
    latest_data["recommendation"] = prediction_info["recommendation"]
    latest_data["severity"] = prediction_info["severity"]
    latest_data["color"] = prediction_info["color"]
    latest_data["timestamp"] = datetime.now().isoformat()

    return {
        "message": "Sensor data received and processed",
        "data": latest_data
    }


@app.get("/sensor")
def get_sensor_data():
    """
    Fetch the latest sensor reading along with ML freshness classification.
    """
    return latest_data


@app.post("/predict")
def predict_endpoint(data: SensorData, update_latest: bool = Query(False, description="Optionally update live dashboard state")):
    """
    Direct prediction endpoint for manual testing or simulation.
    """
    result = predict_freshness(data.temperature, data.humidity, data.mq135)
    response_payload = {
        "temperature": data.temperature,
        "humidity": data.humidity,
        "mq135": data.mq135,
        "timestamp": datetime.now().isoformat(),
        **result
    }

    if update_latest:
        latest_data.update(response_payload)

    return response_payload


@app.get("/dashboard")
def dashboard():
    """
    Combined endpoint for frontend dashboard polling.
    """
    return {
        "project": "Mutton Freshness Monitoring System",
        "status": "online",
        "temperature": latest_data["temperature"],
        "humidity": latest_data["humidity"],
        "mq135": latest_data["mq135"],
        "prediction": latest_data["prediction"],
        "stage": latest_data["stage"],
        "confidence": latest_data["confidence"],
        "probabilities": latest_data["probabilities"],
        "recommendation": latest_data["recommendation"],
        "severity": latest_data["severity"],
        "color": latest_data["color"],
        "timestamp": latest_data["timestamp"],
        "model_metrics": metrics
    }