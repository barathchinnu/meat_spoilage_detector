# 🥩 IoT Mutton Freshness Monitoring System

An end-to-end IoT and Machine Learning system that monitors mutton and meat freshness in real-time using **ESP32**, **DHT22** (temperature & humidity), and **MQ-135** (air quality / ammonia / spoilage gases) sensors paired with a trained **Random Forest Classifier**.

---

## 📦 ML Model Specification (`freshness_model.pkl`)

The model bundle is located at [`backend/model/freshness_model.pkl`](file:///c:/meat-spoilage-project/backend/model/freshness_model.pkl) and saved as a dictionary containing:

| Key | Type | Description |
|---|---|---|
| `model` | `RandomForestClassifier` | Scikit-learn Random Forest model trained on sensor data |
| `feature_names` | `list` | Ordered input features: `['temperature', 'humidity', 'mq135']` |
| `classes` | `list` | Target classification stages: `['Fresh', 'Normal', 'Spoilage']` |
| `target_column` | `str` | `"stage"` |
| `metrics` | `dict` | Evaluation metrics (Accuracy: `1.0`, F1: `1.0`, Precision: `1.0`, Recall: `1.0`) |

### Classification Stages & Interpretation

- 🟢 **Fresh**: Low temperature, standard humidity, clean air (low MQ-135 reading). Safe for consumption.
- 🟡 **Normal / Acceptable**: Moderate temperature or aging indicators. Acceptable for consumption; consume promptly.
- 🔴 **Spoilage**: High temperature/humidity or elevated MQ-135 reading (ammonia, sulfides, VOCs). Hazardous; do not consume.

---

## 🚀 Getting Started

### 1. Install Backend Dependencies

Navigate to the `backend` folder and install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the FastAPI Backend Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- Alternative Redoc: `http://127.0.0.1:8000/redoc`

### 3. Open the Dashboard

Open [`dashboard/index.html`](file:///c:/meat-spoilage-project/dashboard/index.html) directly in any web browser, or serve it with Python:

```bash
cd dashboard
python -m http.server 3000
```
Then visit `http://localhost:3000`.

---

## 📡 API Endpoints

### 1. Root / Health Check
- **`GET /`**
  - Returns API status, loaded model file, expected features, classes, and evaluation metrics.

### 2. Sensor Telemetry Ingestion (ESP32)
- **`POST /sensor`**
  - Payload:
    ```json
    {
      "temperature": 30.5,
      "humidity": 68.5,
      "mq135": 1150
    }
    ```
  - Behavior: Ingests readings, executes ML classification, updates latest telemetry, and returns prediction with confidence and probability distribution.

### 3. Get Latest Telemetry & Freshness
- **`GET /sensor`**
  - Returns the latest sensor readings along with the ML stage, recommendation, confidence score, and per-class probabilities.

### 4. Direct Model Prediction
- **`POST /predict`**
  - Query Param: `update_latest` (optional boolean, default `false`)
  - Payload:
    ```json
    {
      "temperature": 31.8,
      "humidity": 72.0,
      "mq135": 1950
    }
    ```
  - Response:
    ```json
    {
      "temperature": 31.8,
      "humidity": 72.0,
      "mq135": 1950,
      "timestamp": "2026-10-22T20:45:00.000000",
      "prediction": "Spoilage",
      "stage": "Spoilage",
      "confidence": 95.0,
      "probabilities": {
        "Fresh": 0.0,
        "Normal": 0.05,
        "Spoilage": 0.95
      },
      "recommendation": "Spoilage detected! High bacterial decomposition and gas emission. Do NOT consume.",
      "severity": "danger",
      "color": "#EF4444"
    }
    ```

### 5. Full Dashboard Feed
- **`GET /dashboard`**
  - Returns project metadata, latest readings, freshness status, and model metrics for frontend consumption.

### 6. Model Metadata
- **`GET /model-info`**
  - Returns full metadata about the serialized model bundle.

---

## 🔌 ESP32 Arduino / C++ Integration Example

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

#define DHTPIN 4
#define DHTTYPE DHT22
#define MQ135_PIN 34

DHT dht(DHTPIN, DHTTYPE);
const char* serverUrl = "http://<YOUR_BACKEND_IP>:8000/sensor";

void sendSensorData() {
    float temp = dht.readTemperature();
    float hum = dht.readHumidity();
    int mq135Val = analogRead(MQ135_PIN);

    if (isnan(temp) || isnan(hum)) return;

    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    String jsonPayload = "{\"temperature\":" + String(temp, 2) + 
                         ",\"humidity\":" + String(hum, 2) + 
                         ",\"mq135\":" + String(mq135Val) + "}";

    int httpResponseCode = http.POST(jsonPayload);
    http.end();
}
```
