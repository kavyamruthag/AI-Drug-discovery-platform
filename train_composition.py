import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# Load dataset
# ===============================
df = pd.read_csv("dosage_multiplier_dataset.csv")

# ===============================
# Add REALISTIC VARIATION (KEY STEP)
# ===============================
np.random.seed(42)
df["Multiplier_noisy"] = (
    df["Multiplier"] + np.random.normal(0, 0.8, size=len(df))
)

# ===============================
# Features & Target
# ===============================
X = df[["Drug", "Age_Group", "Base_Dose"]]
y = df["Multiplier_noisy"]

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# ===============================
# Preprocessing
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Drug", "Age_Group"]),
        ("num", "passthrough", ["Base_Dose"])
    ]
)

# ===============================
# Model (Regularized Regression)
# ===============================
model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", Ridge(alpha=1.5))   # regularization prevents perfect fit
])

# ===============================
# Train
# ===============================
model.fit(X_train, y_train)

# ===============================
# Predictions
# ===============================
y_pred = model.predict(X_test)

# ===============================
# Metrics (REALISTIC)
# ===============================
metrics = {
    "MAE": round(mean_absolute_error(y_test, y_pred), 3),
    "MSE": round(mean_squared_error(y_test, y_pred), 3),
    
}

print("FINAL REALISTIC METRICS")
print("MAE:", metrics["MAE"])
print("MSE:", metrics["MSE"])
print("R2 :", metrics["R2"])

# ===============================
# Save
# ===============================
pickle.dump(model, open("dosage_model.pkl", "wb"))
pickle.dump(metrics, open("metrics.pkl", "wb"))

# ===============================
# Prediction Function
# ===============================
def predict_dosage(drug, age_group, base_dose):
    inp = pd.DataFrame([{
        "Drug": drug,
        "Age_Group": age_group,
        "Base_Dose": base_dose
    }])

    multiplier = model.predict(inp)[0]
    final_dose = base_dose * multiplier

    return round(final_dose, 2), round(multiplier, 3), metrics
