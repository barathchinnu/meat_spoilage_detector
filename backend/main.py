from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Meat Spoilage Detection API")

# Load ML model
model = joblib.load("model/meat_spoilage_model.pkl")


class SensorData(BaseModel):
    temperature_c: float
    humidity_pct: float
    mq8_d0: int


@app.get("/")
def home():
    return {
        "message": "Meat Spoilage Detection API is running"
    }


@app.post("/predict")
def predict(data: SensorData):

    features = [[
        data.temperature_c,
        data.humidity_pct,
        data.mq8_d0
    ]]

    prediction = model.predict(features)[0]

    return {
        "temperature_c": data.temperature_c,
        "humidity_pct": data.humidity_pct,
        "mq8_d0": data.mq8_d0,
        "prediction": prediction
    }