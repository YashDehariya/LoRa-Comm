#include <ArduinoJson.h>
#include <DHT.h>

#define MQ4_PIN A0
#define MQ7_PIN A1
#define MQ135_PIN A2
#define DHT_PIN 2
#define DHTTYPE DHT11

DHT dht(DHT_PIN, DHTTYPE);

// Calibration constants
const float RL = 10.0;   // Load resistance in kΩ
const float R0_MQ135 = 10.0;
const float R0_MQ7   = 10.0;
const float R0_MQ4   = 10.0;
const float VC = 5.0;    // Supply voltage

// Arrays for 18 samples
float nh3_values[18];
float ch4_values[18];
float co_values[18];
float temp_values[18];
float humidity_values[18];

// --- Conversion helper ---
float voltageToPPM(int rawValue, int sensorType) {
  float vout = (rawValue / 1023.0) * VC;
  if (vout == 0) return 0;
  float rs = RL * ((VC - vout) / vout);
  float ratio, ppm;

  switch (sensorType) {
    case 135:  // MQ135 (NH3)
      ratio = rs / R0_MQ135;
      ppm = 102.2 * pow(ratio, -2.473);
      break;
    case 7:    // MQ7 (CO)
      ratio = rs / R0_MQ7;
      ppm = 99.042 * pow(ratio, -1.518);
      break;
    case 4:    // MQ4 (CH4)
      ratio = rs / R0_MQ4;
      ppm = 4.4 * pow(ratio, -2.2);
      break;
    default:
      ppm = 0;
  }
  return ppm;
}

void setup() {
  Serial.begin(9600);
  dht.begin();
  Serial.println("Starting data collection (18 samples every 10s, PPM output)...");
}

void loop() {
  for (int i = 0; i < 18; i++) {
    int raw_mq135 = analogRead(MQ135_PIN);
    int raw_mq4   = analogRead(MQ4_PIN);
    int raw_mq7   = analogRead(MQ7_PIN);

    // Convert analog readings → PPM
    nh3_values[i] = voltageToPPM(raw_mq135, 135);
    ch4_values[i] = voltageToPPM(raw_mq4, 4);
    co_values[i]  = voltageToPPM(raw_mq7, 7);

    // Read DHT11 values
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    temp_values[i] = isnan(t) ? -1 : t;
    humidity_values[i] = isnan(h) ? -1 : h;

    Serial.print("Sample "); Serial.print(i + 1);
    Serial.print(": NH3="); Serial.print(nh3_values[i]);
    Serial.print("ppm, CH4="); Serial.print(ch4_values[i]);
    Serial.print("ppm, CO="); Serial.print(co_values[i]);
    Serial.print("ppm, Temp="); Serial.print(temp_values[i]);
    Serial.print("°C, Hum="); Serial.print(humidity_values[i]);
    Serial.println("%");

    delay(1000); // 10 seconds
  }

  // Build JSON
  StaticJsonDocument<2048> doc;
  JsonArray nh3 = doc.createNestedArray("nh3");
  JsonArray ch4 = doc.createNestedArray("ch4");
  JsonArray co  = doc.createNestedArray("co");
  JsonArray temp = doc.createNestedArray("temp");
  JsonArray humidity = doc.createNestedArray("humidity");

  for (int i = 0; i < 18; i++) {
    nh3.add(nh3_values[i]);
    ch4.add(ch4_values[i]);
    co.add(co_values[i]);
    temp.add(temp_values[i]);
    humidity.add(humidity_values[i]);
  }

  // Print JSON to Serial
  serializeJsonPretty(doc, Serial);
  Serial.println();
  Serial.println("----- JSON transmission complete -----");
  
  // Optional pause before next batch
  delay(10000);
}
