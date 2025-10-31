import os
# Set TF env vars before importing tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, Query
from pydantic import BaseModel, validator
import numpy as np
import joblib
import pandas as pd
import tensorflow as tf
from typing import List

# Load model and scaler
model = tf.keras.models.load_model("multistep_human_model.h5", compile=False)
scaler = joblib.load("multistep_human_scaler.pkl")

app = FastAPI()

# Human safety thresholds (OSHA/EPA/WHO)
THRESHOLDS = {
    "NH3": {"warning": 25.0, "danger": 50.0},
    "CH4": {"warning": 1000.0, "danger": 50000.0},
    "CO": {"warning": 35.0, "danger": 200.0},
    "Temp": {"warning": 30.0, "danger": 35.0},
    "Humidity": {"warning_low": 30.0, "danger_low": 20.0, "warning_high": 70.0, "danger_high": 80.0}
}

# Input data format
class SensorData(BaseModel):
    nh3: List[float]
    ch4: List[float]
    co: List[float]
    temp: List[float]
    humidity: List[float]

    @validator('nh3', 'ch4', 'co', 'temp', 'humidity')
    def check_length(cls, v):
        if len(v) != 18:
            raise ValueError('Each list must have exactly 18 values (3 hours @10-min)')
        return v

@app.get("/")
def read_root():
    return {"message": "🧑 Human Air Safety API is up and running! Predicts NH3, CH4, CO, Temp, Humidity hazards."}

@app.post("/predict")
def predict_gas_levels(data: SensorData, summary: bool = Query(False, description="Return condensed summary for Reticulum?")):
    try:
        # Stack to (18, 5)
        input_data = np.column_stack([
            data.nh3, data.ch4, data.co, data.temp, data.humidity
        ])
        print(f"Input data shape: {input_data.shape}")

        # Scale & predict (same as before)
        input_scaled = scaler.transform(input_data)
        model_input = input_scaled.reshape(1, 18, 5)
        predictions_scaled = model.predict(model_input)
        predictions_flat = predictions_scaled[0]
        predictions_reshaped = predictions_flat.reshape(36, 5)
        predictions = scaler.inverse_transform(predictions_reshaped)
        predictions = np.maximum(predictions, 0)  # Clamp negatives
        print(f"Inverse transformed predictions: {predictions.shape}")

        if not summary:
            # Full 36-timestep output (original)
            alerts = []
            for idx in range(36):
                row = predictions[idx]
                nh3, ch4, co, temp, hum = row
                status = get_detailed_alerts(row)  # Full alerts per timestep
                alerts.append({
                    "timestep": idx + 1,
                    "prediction": {
                        "NH3": round(float(nh3), 3),
                        "CH4": round(float(ch4), 3),
                        "CO": round(float(co), 3),
                        "Temp": round(float(temp), 3),
                        "Humidity": round(float(hum), 3)
                    },
                    "alerts": status
                })
            return {"predictions": alerts}
        else:
            # Condensed summary for Reticulum (new logic)
            return get_reticulum_summary(predictions, input_data[-1])  # Last input as "current"

    except Exception as e:
        print("❌ Error during prediction:", e)
        return {"error": str(e)}

# New endpoint for Reticulum export (call after /predict for summary JSON to paste in chat)
@app.post("/export_reticulum")
def export_to_reticulum(data: SensorData):
    try:
        # Same prediction as /predict
        input_data = np.column_stack([data.nh3, data.ch4, data.co, data.temp, data.humidity])
        input_scaled = scaler.transform(input_data)
        model_input = input_scaled.reshape(1, 18, 5)
        predictions_scaled = model.predict(model_input)
        predictions_flat = predictions_scaled[0]
        predictions_reshaped = predictions_flat.reshape(36, 5)
        predictions = scaler.inverse_transform(predictions_reshaped)
        predictions = np.maximum(predictions, 0)

        # Get condensed summary
        summary = get_reticulum_summary(predictions, input_data[-1])
        print("Reticulum-ready JSON:", summary)  # Copy this to Reticulum chat
        return summary
    except Exception as e:
        print("❌ Error during export:", e)
        return {"error": str(e)}

# Helper: Detailed alerts per row (for full output)
def get_detailed_alerts(row):
    nh3, ch4, co, temp, hum = row
    status = []
    if nh3 > THRESHOLDS["NH3"]["danger"]:
        status.append("🚨 NH3 critical—Evacuate immediately!")
    elif nh3 > THRESHOLDS["NH3"]["warning"]:
        status.append("⚠️ NH3 high—Ventilate & monitor!")

    if ch4 > THRESHOLDS["CH4"]["danger"]:
        status.append("🚨 CH4 explosive risk—Evacuate!")
    elif ch4 > THRESHOLDS["CH4"]["warning"]:
        status.append("⚠️ CH4 elevated—Increase ventilation!")

    if co > THRESHOLDS["CO"]["danger"]:
        status.append("🚨 CO life-threatening—Seek fresh air now!")
    elif co > THRESHOLDS["CO"]["warning"]:
        status.append("⚠️ CO rising—Check ventilation!")

    if temp > THRESHOLDS["Temp"]["danger"]:
        status.append("🔥 Extreme heat—Cool down & hydrate!")
    elif temp > THRESHOLDS["Temp"]["warning"]:
        status.append("🌡️ Warm—Monitor for heat stress.")

    if hum < THRESHOLDS["Humidity"]["danger_low"]:
        status.append("💧 Very dry—Risk of dehydration!")
    elif hum < THRESHOLDS["Humidity"]["warning_low"]:
        status.append("💧 Dry air—Increase humidity.")
    elif hum > THRESHOLDS["Humidity"]["danger_high"]:
        status.append("💦 Very humid—Mold risk, dehumidify!")
    elif hum > THRESHOLDS["Humidity"]["warning_high"]:
        status.append("💦 Humid—Ventilate to reduce moisture.")

    return status or ["✅ All levels safe."]

# New Helper: Condensed summary for Reticulum (current + 6-hour forecast + status/alerts)
def get_reticulum_summary(predictions, current_row):
    nh3, ch4, co, temp, hum = current_row  # Current (last input)
    current = {
        "current": {
            "NH3": round(float(nh3), 2),
            "CH4": round(float(ch4), 2),
            "CO": round(float(co), 2),
            "Temp": round(float(temp), 2),
            "Humidity": round(float(hum), 2)
        }
    }

    # Aggregate to 6 hourly averages (36 timesteps / 6 = 6 hours)
    hourly_forecast = []
    for h in range(6):
        start = h * 6
        end = start + 6
        hour_preds = predictions[start:end]
        avg_hour = np.mean(hour_preds, axis=0)  # Avg (NH3, CH4, CO, Temp, Hum)
        hourly_forecast.append({
            "hour": h + 1,
            "avg_NH3": round(float(avg_hour[0]), 2),
            "avg_CH4": round(float(avg_hour[1]), 2),
            "avg_CO": round(float(avg_hour[2]), 2),
            "avg_Temp": round(float(avg_hour[3]), 2),
            "avg_Humidity": round(float(avg_hour[4]), 2)
        })

    # Overall status (scan all predictions)
    status = "Safe"
    danger_count = 0
    for row in predictions:
        if any(get_detailed_alerts(row)):  # If any alert
            danger_count += 1
    if danger_count > 18:  # >50% timesteps have alerts
        status = "Danger"
    elif danger_count > 6:  # >16%
        status = "Warning"

    # Key alerts (top 3 from all timesteps)
    all_alerts = []
    for row in predictions:
        all_alerts.extend(get_detailed_alerts(row))
    key_alerts = list(set(all_alerts))[:3]  # Unique, top 3

    return {
        "timestamp": pd.Timestamp.now().isoformat(),
        "status": status,
        "current": current["current"],
        "forecast": hourly_forecast,
        "alerts": key_alerts or ["✅ No hazards detected."]
    }

@app.post("/alert")
def get_alert_summary(data: SensorData):
    try:
        input_data = np.column_stack([data.nh3, data.ch4, data.co, data.temp, data.humidity])
        input_scaled = scaler.transform(input_data)
        model_input = input_scaled.reshape(1, 18, 5)
        predictions_scaled = model.predict(model_input)
        predictions_flat = predictions_scaled[0]
        predictions_reshaped = predictions_flat.reshape(36, 5)
        predictions = scaler.inverse_transform(predictions_reshaped)
        predictions = np.maximum(predictions, 0)

        # Critical alerts (danger only, from summary logic)
        summary = get_reticulum_summary(predictions, input_data[-1])
        critical_alerts = [a for a in summary['alerts'] if '🚨' in a]

        status = "🚨 Critical Alerts" if critical_alerts else "✅ All safe"
        return {
            "status": status,
            "alerts": critical_alerts or ["Air quality within safe thresholds."]
        }
    except Exception as e:
        print("❌ Error during alert generation:", e)
        return {"error": str(e)}