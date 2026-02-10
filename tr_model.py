import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load datasets
features_df = pd.read_csv("drug_features_dataset.csv")
prob_df = pd.read_csv("drug_side_effect_probability.csv")

# Merge features + labels
df = features_df.merge(prob_df, on="Drug_name")
df = df.dropna()

# Encode categorical features
le_abs = LabelEncoder()
le_sol = LabelEncoder()
le_tox = LabelEncoder()

df["abs_enc"] = le_abs.fit_transform(df["absorption"])
df["sol_enc"] = le_sol.fit_transform(df["solubility"])
df["tox_enc"] = le_tox.fit_transform(df["toxicity"])

# Features and target
X = df[["abs_enc", "sol_enc", "tox_enc"]]
y = df["Side_Effect_Probability(%)"]

# 80–20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = np.clip(model.predict(X_test), 0, 100)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 3))

# Save model and encoders
joblib.dump(model, "rf_probability_model.pkl")
joblib.dump(le_abs, "le_abs.pkl")
joblib.dump(le_sol, "le_sol.pkl")
joblib.dump(le_tox, "le_tox.pkl")

metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
joblib.dump(metrics, "regression_metrics.pkl")

print("✅ Model and metrics saved")
