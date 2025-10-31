import serial.tools.list_ports
from serial import Serial
import json
import pandas as pd
from datetime import datetime
import requests
import os
import sys

# -------------------------------
# 🔧 CONFIGURATION
# -------------------------------
FASTAPI_URL = "http://127.0.0.1:8001"
MESHCHAT_API = "http://localhost:8000/api/v1/lxmf-messages/send"
COM_PORT = "COM5"
BAUD_RATE = 9600
TARGET_HASH = "1583f76976bcb3747199f86b9ae9e9f6"

os.makedirs("exports", exist_ok=True)
os.makedirs("summaries", exist_ok=True)

# -------------------------------
# 💬 MESHCHAT ALERT FUNCTION
# -------------------------------
def send_meshchat_alert(message: str, destination_hash: str = None):
    """
    Send alert through MeshChat API running on localhost
    """

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
            print("✅ Alert sent successfully!")
            print(f"   Message: {message}")
            print(f"   To: {destination_hash[:16]}...")
        else:
            print(f"❌ Failed to send alert — Status {response.status_code}")
            print(f"   Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ MeshChat not reachable. Make sure it's running with:")
        print("   python meshchat.py --headless --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"❌ Error sending alert: {e}")

# -------------------------------
# 📡 SERIAL + FASTAPI LOOP
# -------------------------------
ports = serial.tools.list_ports.comports()
print("Available COM ports:")
for port in ports:
    print(f"- {port.device}: {port.description}")

try:
    ser = Serial(COM_PORT, BAUD_RATE, timeout=30)
    print(f"✅ Connected to {COM_PORT}")
    print("Waiting for JSON payload from Arduino...")

    all_readings = []

    while True:
        data = ""
        json_started = False
        brace_count = 0

        # ---- Read full JSON block from Arduino ----
        while True:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue
            print(f"Received data: {line}")

            if line.startswith("{"):
                json_started = True
                brace_count = 1
                data = line
            elif json_started:
                data += line
                brace_count += line.count("{")
                brace_count -= line.count("}")
                if brace_count == 0:
                    break

        data = data.replace("'", '"')
        print(f"Attempting to parse JSON: {data}")

        try:
            json_data = json.loads(data)

            # Extract arrays
            nh3 = json_data.get("nh3", [])
            ch4 = json_data.get("ch4", [])
            co = json_data.get("co", [])
            temp = json_data.get("temp", [])
            humidity = json_data.get("humidity", [])

            processed = {
                "nh3": nh3,
                "ch4": ch4,
                "co": co,
                "temp": temp,
                "humidity": humidity
            }

            all_readings.append(processed)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # ---- Save CSV ----
            df = pd.DataFrame(all_readings, columns=["nh3", "ch4", "co", "temp", "humidity"])
            csv_filename = f"exports/sensor_readings_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"💾 Saved {csv_filename}")

            # ---- Step 1: Send to FastAPI ----
            print("📡 Sending data to /predict endpoint...")
            response = requests.post(f"{FASTAPI_URL}/predict", json=processed, timeout=30)
            if response.status_code == 200:
                print("✅ /predict Response OK")
            else:
                print(f"⚠️ /predict returned {response.status_code}: {response.text}")

            # ---- Step 2: Get Reticulum summary ----
            print("🚀 Requesting Reticulum summary from FastAPI...")
            summary_resp = requests.post(f"{FASTAPI_URL}/export_reticulum", json=processed, timeout=30)
            if summary_resp.status_code != 200:
                print(f"⚠️ /export_reticulum failed: {summary_resp.status_code}")
                continue

            summary_data = summary_resp.json()
            summary_filename = f"summaries/reticulum_summary_{timestamp}.json"
            with open(summary_filename, "w") as f:
                json.dump(summary_data, f, indent=2)
            print(f"📦 Saved summary: {summary_filename}")

            # ---- Step 3: Build alert message ----
            message_text = (
                f"[EcoSenseNet Update @ {timestamp}]\n"
                f"Status: {summary_data.get('status', 'Unknown')}\n"
                f"NH3: {summary_data['current'].get('NH3', 'N/A')} ppm | "
                f"CH4: {summary_data['current'].get('CH4', 'N/A')} ppm | "
                f"CO: {summary_data['current'].get('CO', 'N/A')} ppm\n"
                f"Temp: {summary_data['current'].get('Temp', 'N/A')}°C | "
                f"Humidity: {summary_data['current'].get('Humidity', 'N/A')}%\n"
                f"Alerts: {', '.join(summary_data.get('alerts', []))}"
            )

            # ---- Step 4: Send MeshChat Alert ----
            send_meshchat_alert(message_text, destination_hash=TARGET_HASH)

        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            print(f"Raw data received: {data}")
            continue

except serial.SerialException as e:
    print(f"⚠️ Serial Port Error: {e}")
except KeyboardInterrupt:
    print("\n🛑 Stopping data collection manually...")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("🔒 Serial port closed.")
