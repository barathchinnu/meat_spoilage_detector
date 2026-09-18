from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(
    title="Mutton Freshness Monitoring System",
    version="1.0"
)

# Allow dashboard to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Sensor Data Model
# -----------------------------

class SensorData(BaseModel):
    temperature: float
    humidity: float
    mq135: int


# Store latest sensor reading
latest_data = {
    "temperature": 0.0,
    "humidity": 0.0,
    "mq135": 0,
    "timestamp": None
}


# -----------------------------
# Home
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Mutton Freshness Monitoring API",
        "status": "running"
    }


# -----------------------------
# Receive ESP32 Sensor Data
# -----------------------------

@app.post("/sensor")
def receive_sensor_data(data: SensorData):

    latest_data["temperature"] = data.temperature
    latest_data["humidity"] = data.humidity
    latest_data["mq135"] = data.mq135
    latest_data["timestamp"] = datetime.now().isoformat()

    return {
        "message": "Sensor data received",
        "data": latest_data
    }


# -----------------------------
# Get Latest Sensor Data
# -----------------------------

@app.get("/sensor")
def get_sensor_data():

    return latest_data


# -----------------------------
# Dashboard Data
# -----------------------------

@app.get("/dashboard")
def dashboard():

    return {
        "project": "Mutton Freshness Monitoring System",
        "temperature": latest_data["temperature"],
        "humidity": latest_data["humidity"],
        "mq135": latest_data["mq135"],
        "timestamp": latest_data["timestamp"]
    }