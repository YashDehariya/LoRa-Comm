import serial.tools.list_ports
from serial import Serial
import json
import pandas as pd
from datetime import datetime
import requests
import os
import sys

FASTAPI_URL = "http://127.0.0.1:8001"
MESHCHAT_API = "http://localhost:8000/api/v1/lxmf-messages/send"
COM_PORT = "COM10"
BAUD_RATE = 9600
TARGET_HASH = "1583f76976bcb3747199f86b9ae9e9f6"

os.makedirs("exports", exist_ok=True)
os.makedirs("summaries", exist_ok=True)

def send_meshchat_alert(message, destination_hash=None):
    if destination_hash is None:
        destination_hash = TARGET_HASH

    payload = {
        "lxmf_message": {
            "destination_hash": destination_hash,
            "content": message,
            "fields": {}
        }
    }

    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Sending alert to MeshChat...")
        response = requests.post(MESHCHAT_API, json=payload, timeout=10)

        if response.status_code == 200:
            print("Alert sent successfully.")
            print(f"To: {destination_hash[:16]}...")
        else:
            print(f"Failed to send alert (status {response.status_code})")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("MeshChat not reachable. Start it with:")
        print("python meshchat.py --headless --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"Error sending alert: {e}")

ports = serial.tools.list_ports.comports()
print("Available COM ports:")
for port in ports:
    print(f"- {port.device}: {port.description}")

try:
    ser = Serial(COM_PORT, BAUD_RATE, timeout=30)
    print(f"Connected to {COM_PORT}")
    print("Waiting for JSON payloads...")

    all_readings = []

    while True:
        data, json_started, brace_count = "", False, 0

        while True:
            line = ser.readline().decode("utf-8").strip()
            if not line:
                continue
            print(f"Received: {line}")

            if line.startswith("{"):
                json_started, brace_count, data = True, 1, line
            elif json_started:
                data += line
                brace_count += line.count("{") - line.count("}")
                if brace_count == 0:
                    break

        data = data.replace("'", '"')
        print(f"Parsing JSON: {data}")

        try:
            json_data = json.loads(data)

            processed = {
                "nh3": json_data.get("nh3", []),
                "ch4": json_data.get("ch4", []),
                "co": json_data.get("co", []),
                "temp": json_data.get("temp", []),
                "humidity": json_data.get("humidity", [])
            }

            all_readings.append(processed)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            df = pd.DataFrame(all_readings, columns=["nh3", "ch4", "co", "temp", "humidity"])
            csv_file = f"exports/sensor_readings_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"Saved {csv_file}")

            print("Sending data to FastAPI...")
            r = requests.post(f"{FASTAPI_URL}/predict", json=processed, timeout=30)
            if r.status_code == 200:
                print("/predict OK")
            else:
                print(f"FastAPI error {r.status_code}")

            print("Requesting Reticulum summary...")
            s = requests.post(f"{FASTAPI_URL}/export_reticulum", json=processed, timeout=30)
            if s.status_code != 200:
                print(f"/export_reticulum failed ({s.status_code})")
                continue

            summary = s.json()
            summary_file = f"summaries/reticulum_summary_{timestamp}.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Saved summary: {summary_file}")

            message = (
                f"[EcoSenseNet Update @ {timestamp}]\n"
                f"Status: {summary.get('status', 'Unknown')}\n"
                f"NH3: {summary['current'].get('NH3', 'N/A')} ppm | "
                f"CH4: {summary['current'].get('CH4', 'N/A')} ppm | "
                f"CO: {summary['current'].get('CO', 'N/A')} ppm\n"
                f"Temp: {summary['current'].get('Temp', 'N/A')}°C | "
                f"Humidity: {summary['current'].get('Humidity', 'N/A')}%\n"
                f"Alerts: {', '.join(summary.get('alerts', []))}"
            )

            send_meshchat_alert(message, destination_hash=TARGET_HASH)

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw data: {data}")
            continue

except serial.SerialException as e:
    print(f"Serial port error: {e}")
except KeyboardInterrupt:
    print("\nStopped manually.")
finally:
    if "ser" in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")
