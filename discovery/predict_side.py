import pandas as pd
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model & encoders
model = joblib.load(os.path.join(BASE_DIR, "../rf_probability_model.pkl"))
le_abs = joblib.load(os.path.join(BASE_DIR, "../le_abs.pkl"))
le_sol = joblib.load(os.path.join(BASE_DIR, "../le_sol.pkl"))
le_tox = joblib.load(os.path.join(BASE_DIR, "../le_tox.pkl"))

# Load feature dataset
features_df = pd.read_csv(
    os.path.join(BASE_DIR, "../drug_features_dataset.csv")
)

def predict_side_effect_percentage(drug_name):
    row = features_df[
        features_df["Drug_name"].str.lower() == drug_name.lower()
    ]

    if row.empty:
        return None

    absorption = row.iloc[0]["absorption"]
    solubility = row.iloc[0]["solubility"]
    toxicity = row.iloc[0]["toxicity"]

    abs_enc = le_abs.transform([absorption])[0]
    sol_enc = le_sol.transform([solubility])[0]
    tox_enc = le_tox.transform([toxicity])[0]

    prob = model.predict([[abs_enc, sol_enc, tox_enc]])[0]
    prob = np.clip(prob, 0, 100)

    return round(prob, 2)
