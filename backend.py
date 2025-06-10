from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import requests

app = Flask(__name__)
CORS(app)

# Load trained models
solar_model = joblib.load("solar.pkl")  # Solar model
wind_model = joblib.load("wind.pkl")       # Wind model

# API key for OpenWeatherMap
OPENWEATHER_API_KEY = "ade1b5741945c6a047fdea1f9ffdec61"

@app.route('/predict', methods=['POST'])
def predict_power_from_location():
    data = request.json
    place = data.get('place')

    # Get weather data
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={place}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(weather_url)
    if response.status_code != 200:
        return jsonify({"error": "Place not found or weather API failed"}), 400

    weather_data = response.json()
    cloud_cover = weather_data['clouds']['all']
    wind_speed = weather_data['wind']['speed']
    temperature = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    air_density = weather_data['main'].get('pressure', 1013) / (287.05 * (temperature + 273.15))  # Ideal gas law approx
    solar_irradiance = max(0, 1000 - cloud_cover * 10)  # Approximation

    # Solar prediction
    solar_features = np.array([[solar_irradiance, temperature, wind_speed, humidity]])
    predicted_power_per_m2 = solar_model.predict(solar_features)[0]
    panel_area = 5.0  # m²
    predicted_total_power = predicted_power_per_m2 * panel_area

    # Wind prediction
    wind_features = np.array([[wind_speed, air_density, temperature, humidity]])
    predicted_wind_pct = wind_model.predict(wind_features)[0]

    return jsonify({
        'place': place,
        # Solar output
        'solar_irradiance': solar_irradiance,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'cloud_cover': cloud_cover,
        'panel_area': panel_area,
         'air_density': round(air_density, 4),
        'predicted_power_per_m2': predicted_power_per_m2,
        'predicted_total_power': predicted_total_power,
        # Wind output
        'predicted_wind_power_pct': predicted_wind_pct
    })

if __name__ == '__main__':
    app.run(debug=True)
