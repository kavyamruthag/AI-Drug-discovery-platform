import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("drug_properties_with_binding_affinity.csv")

# Create target (simple rule-based risk for ML demo)
df["Risk_Class"] = np.where(
    (df["Binding_Affinity_kcal_per_mol"] < -7) & (df["Lipophilicity_LogP"] < 3),
    0,  # LOW RISK
    1   # HIGH RISK
)

X = df[["Lipophilicity_LogP", "Binding_Affinity_kcal_per_mol"]]
y = df["Risk_Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

# Save model and accuracy
joblib.dump(model, "drug_risk_model.pkl")
joblib.dump(round(accuracy, 2), "model_accuracy.pkl")

print("Model trained | Accuracy:", round(accuracy, 2))
