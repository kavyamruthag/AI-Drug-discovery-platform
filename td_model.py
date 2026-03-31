import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("disease_lack_of_component.csv")
df = df.dropna()

# Features and labels
X = df["Disease"]
y = df["Lack_of_Component"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions (ONLY for metric calculation)
y_pred = model.predict(X_test_vec)

# Calculate metrics
metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
    "precision": round(precision_score(y_test, y_pred, average="weighted"), 3),
    "recall": round(recall_score(y_test, y_pred, average="weighted"), 3),
    "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 3)
}

# Save model, vectorizer, metrics
joblib.dump(model, "disease_deficiency_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(metrics, "model_metrics.pkl")

print("✅ Model, vectorizer, and metrics saved")
