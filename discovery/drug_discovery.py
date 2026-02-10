import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Load your dataset
df = pd.read_csv('drug_disease.csv')  # Make sure this file is in the same folder
df = df[['Disease_Name', 'Drug_Name']].drop_duplicates()

# Step 2: Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
disease_embeddings = model.encode(df['Disease_Name'].tolist())

# Step 3: Function to predict drugs
def predict_drugs(input_disease):
    input_emb = model.encode([input_disease])
    similarities = cosine_similarity(input_emb, disease_embeddings)[0]
    best_idx = np.argmax(similarities)
    matched_disease = df.iloc[best_idx]['Disease_Name']
    drugs = df[df['Disease_Name'] == matched_disease]['Drug_Name'].tolist()
    return {
        "Input Disease": input_disease,
        "Matched Disease": matched_disease,
        "Drugs": drugs
        
    }

# Step 4: Example usage

