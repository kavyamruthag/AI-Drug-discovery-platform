import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "../disease_deficiency_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "../tfidf_vectorizer.pkl"))
metrics = joblib.load(os.path.join(BASE_DIR, "../model_metrics.pkl"))

def predict_lack_of_component(disease_name):
    disease_vec = vectorizer.transform([disease_name])
    prediction = model.predict(disease_vec)[0]

    return prediction, metrics
