import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(42)
data_size = 500

# Simulate input features
df = pd.DataFrame({
    'solar_irradiance': np.random.uniform(0, 1000, data_size),  # W/m²
    'temperature': np.random.uniform(15, 40, data_size),        # °C
    'wind_speed': np.random.uniform(0, 20, data_size),          # m/s
    'humidity': np.random.uniform(10, 90, data_size),           # %
    'panel_area': np.random.uniform(1, 10, data_size),          # m², varying panel size
})

# Simulate total power output (W) - real power output depends on irradiance, area and efficiency
df['power_output'] = (
    df['solar_irradiance'] * df['panel_area'] * 0.18 +   # Assume 18% efficiency
    4.5 * df['wind_speed'] -
    0.3 * df['humidity'] +
    np.random.normal(0, 30, data_size)  # noise
)

# Calculate power output per square meter (W/m²)
df['power_per_m2'] = df['power_output'] / df['panel_area']

# Features and target for model (predict power per m²)
features = ['solar_irradiance', 'temperature', 'wind_speed', 'humidity']
X = df[features]
y = df['power_per_m2']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

import joblib

# Save the trained RandomForestRegressor model to a file
joblib.dump(model, 'rf_model.pkl')
print("✅ Model saved as 'rf_model.pkl'")


# Predict power per m² on test set
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.2f} W/m²")
print(f"RMSE: {rmse:.2f} W/m²")

# Example: Predict total power output for test samples by multiplying predicted power/m² by actual panel area
panel_area_test = df.loc[y_test.index, 'panel_area']
total_power_pred = y_pred * panel_area_test.values


# Predict power per m² on test set
y_pred = model.predict(X_test)

# Calculate R² score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.2f} W/m²")
print(f"RMSE: {rmse:.2f} W/m²")


# Save actual and predicted total power output to CSV
results_df = pd.DataFrame({
    'solar_irradiance': X_test['solar_irradiance'].values,
    'temperature': X_test['temperature'].values,
    'wind_speed': X_test['wind_speed'].values,
    'humidity': X_test['humidity'].values,
    'panel_area': panel_area_test.values,
    'actual_power_per_m2': y_test.values,
    'predicted_power_per_m2': y_pred,
    'actual_total_power': y_test.values * panel_area_test.values,
    'predicted_total_power': total_power_pred
})

results_df.to_csv("solar_power_predictions.csv", index=False)
print("✅ Predictions saved to 'solar_power_predictions.csv'")

# Plot actual vs predicted power per m2 (first 50 samples)
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:50], label="Actual power/m²")
plt.plot(y_pred[:50], label="Predicted power/m²")
plt.xlabel("Sample")
plt.ylabel("Power Output per m² (W/m²)")
plt.title("Actual vs Predicted Power Output per m²")
plt.legend()
plt.grid(True)
plt.show()