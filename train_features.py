import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# Load dataset
# ===============================
df = pd.read_csv("drug_properties_with_binding_affinity.csv")

# ===============================
# Let's Predict Binding Affinity
# (example continuous target)
# ===============================
X = df[["Lipophilicity_LogP"]]
y = df["Binding_Affinity_kcal_per_mol"]

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# ===============================
# Train Model (REGRESSION)
# ===============================
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# Predictions
# ===============================
y_pred = model.predict(X_test)

# ===============================
# Evaluation Metrics
# ===============================
metrics = {
    "MAE": round(mean_absolute_error(y_test, y_pred), 3),
    "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
    
}

print("\nRegression Model Results")
for k, v in metrics.items():
    print(f"{k}: {v}")

# ===============================
# Save model & metrics
# ===============================
joblib.dump(model, "drug_risk_model.pkl")
joblib.dump(metrics, "drug_model_metrics.pkl")

print("\nModel saved successfully.")
