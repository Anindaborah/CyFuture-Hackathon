import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib  # <-- for saving the model

# Set seed and simulate data
np.random.seed(42)
data_size = 500

# Simulated input features for wind power prediction
df = pd.DataFrame({
    'wind_speed': np.random.uniform(0, 25, data_size),     # m/s
    'air_density': np.random.uniform(1.0, 1.3, data_size), # kg/m³
    'temperature': np.random.uniform(-10, 40, data_size),  # °C
    'humidity': np.random.uniform(10, 90, data_size),      # %
})

# Define the power percentage curve
def wind_power_pct(ws):
    if ws < 3:
        return 0
    elif ws > 25:
        return 0
    elif ws <= 12:
        return (ws / 12) ** 3 * 100
    else:
        return 100

df['power_pct'] = df['wind_speed'].apply(wind_power_pct)
df['power_pct'] *= (df['air_density'] / 1.225)
df['power_pct'] += np.random.normal(0, 5, data_size)
df['power_pct'] = df['power_pct'].clip(0, 100)

# Define features and target
features = ['wind_speed', 'air_density', 'temperature', 'humidity']
X = df[features]
y = df['power_pct']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to disk as 'wind.pkl'
joblib.dump(model, 'wind.pkl')
print("✅ Model saved as 'wind.pkl'")

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f} %")
print(f"RMSE: {rmse:.2f} %")
print(f"R² Score: {r2:.4f}")

# Save predictions to CSV
results_df = pd.DataFrame({
    'wind_speed': X_test['wind_speed'],
    'air_density': X_test['air_density'],
    'temperature': X_test['temperature'],
    'humidity': X_test['humidity'],
    'actual_power_pct': y_test,
    'predicted_power_pct': y_pred
})
results_df.to_csv("wind_power_predictions.csv", index=False)
print("✅ Predictions saved to 'wind_power_predictions.csv'")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:50], label='Actual Power %')
plt.plot(y_pred[:50], label='Predicted Power %')
plt.xlabel('Sample')
plt.ylabel('Power Generation (%)')
plt.title('Actual vs Predicted Wind Power Generation (%)')
plt.legend()
plt.grid(True)
plt.show()
