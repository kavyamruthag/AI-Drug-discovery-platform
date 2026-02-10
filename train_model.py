import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\dheer\OneDrive\Documents\DRUG-DISCOVERY\drug_disease.csv")
df.dropna(subset=['Disease_name', 'Drug_name'], inplace=True)

# -----------------------------
# 2️⃣ Normalize text
# -----------------------------
df['Disease_name'] = df['Disease_name'].str.strip().str.lower()
df['Drug_name'] = df['Drug_name'].str.strip().str.lower()

# -----------------------------
# 3️⃣ Group drugs by disease
# -----------------------------
df_grouped = df.groupby('Disease_name')['Drug_name'].apply(lambda x: list(x.unique())).reset_index()

# -----------------------------
# 4️⃣ Encode drugs (multi-label)
# -----------------------------
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(df_grouped['Drug_name'])

# -----------------------------
# 5️⃣ Create embeddings for disease names
# -----------------------------
print("🔹 Generating embeddings...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
X_embed = embed_model.encode(df_grouped['Disease_name'].tolist(), batch_size=32, show_progress_bar=True)

# -----------------------------
# 6️⃣ Train classifier
# -----------------------------
#X_train, X_test, y_train, y_test = train_test_split(X_embed, y_encoded, test_size=0.2, random_state=42)

print("🔹 Training RandomForest model...")
clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
clf.fit(X_embed, y_encoded)
#clf.fit(X_train, y_train)
# -----------------------------
# 7️⃣ Create 'models' directory
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

# -----------------------------
# 8️⃣ Save models
# -----------------------------
pickle.dump(clf, open(os.path.join(models_dir, "drug_model.pkl"), "wb"))
pickle.dump(mlb, open(os.path.join(models_dir, "mlb.pkl"), "wb"))
pickle.dump(embed_model, open(os.path.join(models_dir, "embedding_model.pkl"), "wb"))

# -----------------------------
# 9️⃣ Save known diseases and embeddings (for unseen disease prediction)
# -----------------------------
known_diseases = df_grouped['Disease_name'].tolist()
np.save(os.path.join(models_dir, "known_disease_embeddings.npy"), X_embed)
with open(os.path.join(models_dir, "known_diseases.pkl"), "wb") as f:
    pickle.dump(known_diseases, f)

print("✅ Training completed successfully!")
print("✅ Models and reference data saved in the 'models' folder.")
