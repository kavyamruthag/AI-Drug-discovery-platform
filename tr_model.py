import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

data = pd.read_csv("drug_side_effect_probability.csv")

data.columns = ["Drug_Name", "Side_Effect_Probability"]

# Clean data
data = data.dropna()
data["Side_Effect_Probability"] = pd.to_numeric(
    data["Side_Effect_Probability"], errors="coerce"
)
data = data.dropna()

# --------------------------------------------------
# TF-IDF Feature Engineering
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(data["Drug_Name"])
y = data["Side_Effect_Probability"]

print("Feature count:", X.shape[1])

# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Train Model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------------------------
# Evaluate
# --------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance")
print("Train R2 :", round(model.score(X_train, y_train), 3))
print("Test R2  :", round(r2, 3))
print("MAE      :", round(mae, 2))
print("RMSE     :", round(rmse, 2))

# --------------------------------------------------
# Save NEW Model Files (New Names)
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

joblib.dump(model, os.path.join(BASE_DIR, "drug_rf_tfidf_model.pkl"))
joblib.dump(vectorizer, os.path.join(BASE_DIR, "drug_tfidf_vectorizer.pkl"))

metrics = {
    "MAE": round(mae, 2),
    "RMSE": round(rmse, 2),
    
}

joblib.dump(metrics, os.path.join(BASE_DIR, "drug_model_metrics.pkl"))

print("\n✅ New model, vectorizer, and metrics saved successfully!")
