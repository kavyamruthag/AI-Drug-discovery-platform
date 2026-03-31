import joblib
import os
import numpy as np
from django.shortcuts import render

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "drug_rf_tfidf_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "drug_tfidf_vectorizer.pkl")
metrics_path = os.path.join(BASE_DIR, "drug_model_metrics.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

try:
    metrics = joblib.load(metrics_path)
except:
    metrics = {"MAE": "-", "RMSE": "-"}

def drug_predictor(request):
    result = None

    if request.method == "POST":
        drug_name = request.POST.get("drug")

        if drug_name:
            input_vector = vectorizer.transform([drug_name])
            prediction = model.predict(input_vector)
            probability = np.clip(prediction[0], 0, 100)

            result = {
                "drug": drug_name,
                "probability": round(float(probability), 2)
            }

    return render(
        request,
        "drug_predictor.html",
        {
            "result": result,
            "metrics": metrics
        }
    )
