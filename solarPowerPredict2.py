# -----------------------------
# Step 0: Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# -----------------------------
# Step 1: Simulate Dataset
# -----------------------------
np.random.seed(42)
data_size = 500

df = pd.DataFrame({
    'solar_irradiance': np.random.uniform(0, 1000, data_size),
    'temperature': np.random.uniform(15, 40, data_size),
    'wind_speed': np.random.uniform(0, 20, data_size),
    'humidity': np.random.uniform(10, 90, data_size),
})

# Realistic nonlinear formula for power output
df['power_output'] = (
    0.55 * df['solar_irradiance'] +
    4.5 * df['wind_speed'] -
    0.3 * df['humidity'] +
    0.1 * df['temperature'] * df['wind_speed'] -
    0.2 * df['temperature']**1.2 +
    np.random.normal(0, 30, data_size)  # reduced noise for better signal
)

# -----------------------------
# Step 2: Feature Engineering
# -----------------------------
df['temp_irr'] = df['temperature'] * df['solar_irradiance']
df['temp_sq'] = df['temperature']**2
df['irr_sq'] = df['solar_irradiance']**2
df['log_irr'] = np.log1p(df['solar_irradiance'])  # log(irr + 1)
df['wind_humidity'] = df['wind_speed'] * df['humidity']

# -----------------------------
# Step 3: Train-Test Split
# -----------------------------
X = df.drop('power_output', axis=1)
y = df['power_output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 4: XGBoost + Grid Search
# -----------------------------
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.03, 0.05],
    'max_depth': [3, 4],
}

grid = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# -----------------------------
# Step 5: Evaluation
# -----------------------------
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Best Parameters:", grid.best_params_)
print(f"MAE: {mae:.2f} kW")
print(f"RMSE: {rmse:.2f} kW")
print(f"R² Score: {r2:.3f}")

# -----------------------------
# Step 6: Visualizations
# -----------------------------

# Actual vs Predicted Line Plot
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:50], label="Actual", marker='o')
plt.plot(y_pred[:50], label="Predicted", marker='x')
plt.xlabel("Sample")
plt.ylabel("Power Output (kW)")
plt.title("Actual vs Predicted Power Output")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Scatter")
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

# Feature Importance
importances = best_model.feature_importances_
features = X.columns
plt.figure(figsize=(7,4))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance (XGBoost)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Step 7: Save Predictions to CSV
# -----------------------------
results_df = pd.DataFrame({
    'actual_power_output': y_test.values,
    'predicted_power_output': y_pred
})

results_df.to_csv("predicted_results.csv", index=False)
print("✅ Results saved to 'predicted_results.csv'")
