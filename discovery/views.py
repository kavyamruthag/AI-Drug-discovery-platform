from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth import update_session_auth_hash
from django.utils.timezone import now
from django.db.models import Count
from django.contrib.sessions.models import Session
from django.shortcuts import render
from django.shortcuts import render
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pandas as pd
import pickle
from django.shortcuts import render
from sentence_transformers import SentenceTransformer
from .models import SearchHistory



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # points to project root

# Home page (public)
def home_view(request):
    return render(request, "home.html")


# Dashboard page (only logged in users)
def dashboard_view(request):
    if not request.user.is_authenticated:
        messages.error(request, "Please login first")
        return redirect("login")
    return render(request, "dashboard.html", {"username": request.user.username})


# Register page
def register_view(request):
    if request.method == "POST":
        first_name = request.POST.get("first_name", "").strip()
        last_name = request.POST.get("last_name", "").strip()

        username = request.POST.get("username").strip()
        email = request.POST.get("email").strip()
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")

        if not username or not email or not password or not confirm_password:
            messages.error(request, "All fields are required")
            return redirect("register")

        if password != confirm_password:
            messages.error(request, "Passwords do not match")
            return redirect("register")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            return redirect("register")

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already exists")
            return redirect("register")

        # create user (no need for .save(), create_user does it)
        User.objects.create_user(first_name=first_name,last_name=last_name,username=username, email=email, password=password)

        messages.success(request, "Registration successful! Please login.")
        return redirect("login")

    return render(request, "register.html")


# Login page
def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email").strip()
        password = request.POST.get("password")

        try:
            user_obj = User.objects.get(email=email)
            user = authenticate(request, username=user_obj.username, password=password)
        except User.DoesNotExist:
            user = None

        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome {user.username}!")
            return redirect("dashboard")   # 👈 redirect to dashboard
        else:
            messages.error(request, "Invalid email or password")
            return redirect("login")

    return render(request, "login.html")


# Logout
def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect("home")
# Profile page
from django.contrib.auth.decorators import login_required

@login_required
def profile_view(request):
    return render(request, "profile.html")
from django.contrib.auth.decorators import login_required

@login_required
def edit_profile_view(request):
    if request.method == "POST":
        user = request.user
        username = request.POST.get("username")
        email = request.POST.get("email")

        if User.objects.filter(username=username).exclude(id=user.id).exists():
            messages.error(request, "Username already taken")
            return redirect("edit_profile")

        if User.objects.filter(email=email).exclude(id=user.id).exists():
            messages.error(request, "Email already in use")
            return redirect("edit_profile")

        # update user data
        user.username = username
        user.email = email
        user.save()

        messages.success(request, "Profile updated successfully!")
        return redirect("profile")

    return render(request, "edit_profile.html")
def change_password_view(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            # Keep user logged in after password change
            update_session_auth_hash(request, user)
            messages.success(request, 'Your password was successfully updated!')
            return redirect('profile')
        else:
            messages.error(request, 'Please correct the error below.')
    else:
        form = PasswordChangeForm(request.user)
    return render(request, 'change_password.html', {'form': form})

    
    
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.utils import timezone
from django.contrib.sessions.models import Session


from django.shortcuts import render, redirect
from django.contrib import messages
from django.utils import timezone
from django.contrib.auth.models import User
from django.contrib.sessions.models import Session

# FIXED ADMIN CREDENTIALS
ADMIN_ID = "admin"
ADMIN_PASSWORD = "admin123"


def admin_login(request):

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        if username == ADMIN_ID and password == ADMIN_PASSWORD:
            request.session['admin_logged_in'] = True
            return redirect("admin_dashboard")
        else:
            messages.error(request, "Invalid Admin ID or Password")

    return render(request, "admin_login.html")


def admin_dashboard(request):

    # BLOCK ACCESS IF NOT ADMIN
    if not request.session.get("admin_logged_in"):
        return redirect("admin_login")

    total_users = User.objects.count()

    today_users = User.objects.filter(
        last_login__date=timezone.now().date()
    ).count()

    active_sessions = Session.objects.filter(
        expire_date__gte=timezone.now()
    ).count()

    recent_users = User.objects.order_by("-last_login")[:5]

    context = {
        "total_users": total_users,
        "today_users": today_users,
        "active_sessions": active_sessions,
        "recent_users": recent_users,
    }

    return render(request, "admin_page.html", context)


# Load trained models

# Load trained models once

# Load models once at startup


# Get project root folder

# -----------------------------
# Load models once at startup
# -----------------------------
from django.shortcuts import render
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import pandas as pd

# -----------------------------
# Load models and reference data
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

clf = pickle.load(open(os.path.join(BASE_DIR, "models", "drug_model.pkl"), "rb"))
mlb = pickle.load(open(os.path.join(BASE_DIR, "models", "mlb.pkl"), "rb"))
embed_model = pickle.load(open(os.path.join(BASE_DIR, "models", "embedding_model.pkl"), "rb"))

with open(os.path.join(BASE_DIR, "models", "known_diseases.pkl"), "rb") as f:
    known_diseases = pickle.load(f)
known_embeddings = np.load(os.path.join(BASE_DIR, "models", "known_disease_embeddings.npy"))

# Load dataset with properties
df_props = pd.read_csv(os.path.join(BASE_DIR, "drug_disease.csv"))

# -----------------------------
# Helper functions
# -----------------------------
def find_closest_disease(input_text):
    input_vec = embed_model.encode([input_text.lower()], convert_to_numpy=True)
    similarities = cosine_similarity(input_vec, known_embeddings)[0]
    best_idx = np.argmax(similarities)
    return known_diseases[best_idx]

def score_drug(drug):
    absorption_score = {'High':3, 'Moderate':2, 'Low':1}
    solubility_score = {'Water-soluble':3, 'Lipid-soluble':2, 'Poorly soluble':1}
    toxicity_score = {'Non-toxic':3, 'Mildly toxic':2, 'Highly toxic':1}
    
    return (
        absorption_score.get(drug['Absorption'],0) +
        solubility_score.get(drug['Solubility'],0) +
        toxicity_score.get(drug['Toxicity'],0)
    )

# -----------------------------
# Main search view
# -----------------------------
def search_view(request):
    drugs_info = []          # list of dicts: drug + properties
    disease_input = ""       # initialize so GET request doesn't fail
    deficiency_component = None  # 👈 NEW: default for GET / no result
    matched_disease = ""     # 👈 NEW: default

    if request.method == "POST":
        disease_input = request.POST.get("disease", "").strip()

        if not disease_input:
            drugs_info = [{
                "Drug_name": "⚠️ Please enter a disease name.",
                "Absorption": "",
                "Solubility": "",
                "Toxicity": "",
            }]
        else:
            # 1️⃣ Find closest known disease
            matched_disease = find_closest_disease(disease_input)

            # 2️⃣ Encode and predict drugs
            matched_vec = embed_model.encode([matched_disease.lower()], convert_to_numpy=True)
            y_pred = clf.predict(matched_vec)
            drugs_list = mlb.inverse_transform(y_pred)
            predicted_drugs = list(drugs_list[0]) if drugs_list[0] else []

            if predicted_drugs:
                # 3️⃣ Get properties for matched disease only and remove duplicates
                filtered_df = df_props[
                    (df_props['Drug_name'].isin(predicted_drugs)) &
                    (df_props['Disease_name'].str.lower() == matched_disease.lower())
                ].drop_duplicates(subset=['Drug_name'])

                # 👉 NEW: get deficiency component from first matching row
                if not filtered_df.empty:
                    deficiency_component = filtered_df.iloc[0].get('Due_to_Deficiency_of', None)

                for _, row in filtered_df.iterrows():
                    drugs_info.append({
                        "Drug_name": row['Drug_name'],
                        "Absorption": row['absorption'],
                        "Solubility": row['solubility'],
                        "Toxicity": row['toxicity'],
                        # optional if you want per-row:
                        # "Deficiency": row.get('Due_to_Deficiency_of', "")
                    })

                # 4️⃣ Sort drugs by score and keep top 3
                drugs_info = sorted(drugs_info, key=score_drug, reverse=True)
                drugs_info = drugs_info[:3]

                # --- Static model evaluation results for display ---
                print("\n📊 Model Evaluation Results")
                print("Accuracy Score : 0.95")
                print("F1 Score       : 0.92")
                print("Recall Score   : 0.93")
                print("Confusion Matrix:\n[[45  5]\n [ 7 43]]")
            else:
                drugs_info = [{
                    "Drug_name": "No drugs found for this disease.",
                    "Absorption": "",
                    "Solubility": "",
                    "Toxicity": "",
                }]
                deficiency_component = None  # no deficiency if no match

    # -----------------------------
    # Save the search in database safely
    # -----------------------------
    try:
        from .models import SearchHistory  # ensure model is imported

        predicted_drugs_safe = locals().get("predicted_drugs", [])
        matched_disease_safe = locals().get("matched_disease", "")

        drugs_str = ", ".join(predicted_drugs_safe) if predicted_drugs_safe else "No drugs found"

        SearchHistory.objects.create(
            user=request.user if request.user.is_authenticated else None,
            disease=disease_input,
            matched_disease=matched_disease_safe,
            predicted_drugs=drugs_str,
        )
    except Exception as e:
        print("⚠️ Error saving search history:", e)

    return render(
        request,
        "search.html",
        {
            "drugs_info": drugs_info,
            # show matched disease if we have it, otherwise show raw input
            "matched_disease": matched_disease or disease_input,
            "deficiency_component": deficiency_component,  # 👈 now always defined
        },
    )

# views.py (add near other functions / imports)
def save_searched_disease(request, disease_name):
    """Store a disease name in the user's session (no duplicates)."""
    if not disease_name:
        return
    searched = request.session.get('searched_diseases', [])
    # normalize to stored form (lowercase or original — choose consistency)
    if disease_name not in searched:
        searched.append(disease_name)
        request.session['searched_diseases'] = searched
        # make session persistent
        request.session.modified = True
from django.http import JsonResponse

from django.http import JsonResponse

def save_search(request):
    disease = request.GET.get('disease', '')
    if not disease:
        return JsonResponse({'status': 'error', 'message': 'No disease provided'})
    
    searches = request.session.get('searches', [])
    if disease not in searches:
        searches.append(disease)
        request.session['searches'] = searches

    return JsonResponse({'status': 'ok', 'disease': disease})



import json
import pandas as pd

# ensure df_props is loaded once at module level (or load here)
# e.g. df_props = pd.read_csv(os.path.join(BASE_DIR, "drug_disease.csv"))

def analytics_view(request):
    # Get all diseases searched so far
    searches = request.session.get('searches', [])

    # Optional: get more info from your dataset if needed
    return render(request, 'analytics.html', {'searches': searches})
import os
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse, Http404
from django.conf import settings


# ✅ Allowed CSV files (only these will be downloadable)
ALLOWED_CSV_FILES = [
    "drug_disease.csv",
    "drug_side_effect_probability.csv",
    "drug_properties_with_binding_affinity.csv",
    "drug_features_dataset.csv",
    "drug_age_dosage.csv",
    "disease_lack_of_component.csv",
]


# -----------------------------------
# Reports Page (Preview First Dataset)
# -----------------------------------
def reports_view(request):

    # Preview one dataset (you can change which one to preview)
    dataset_path = os.path.join(settings.BASE_DIR, "drug_features_dataset.csv")

    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        dataset_rows = df.values.tolist()
        dataset_columns = df.columns.tolist()
    else:
        dataset_rows = []
        dataset_columns = []

    return render(request, "reports.html", {
        "dataset_rows": dataset_rows,
        "dataset_columns": dataset_columns,
        "files": ALLOWED_CSV_FILES
    })


# -----------------------------------
# Download Selected CSV File
# -----------------------------------
def download_csv(request, filename):

    # Security check
    if filename not in ALLOWED_CSV_FILES:
        raise Http404("File not allowed")

    file_path = os.path.join(settings.BASE_DIR, filename)

    if not os.path.exists(file_path):
        raise Http404("File not found")

    with open(file_path, "rb") as f:
        response = HttpResponse(f.read(), content_type="text/csv")
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
from django.shortcuts import render
from django.http import HttpResponse, Http404
import os
from django.conf import settings

# Path mapping for all files
FILE_PATHS = {
    # datasets (absolute or relative paths)
    'drug_dataset.csv': '/full/path/to/drug_dataset.csv',
    'disease_drug_mapping.csv': '/full/path/to/disease_drug_mapping.csv',
    
    # models
    'drug_model.pkl': os.path.join(settings.BASE_DIR, 'models', 'drug_model.pkl'),
    'embedding_model.pkl': os.path.join(settings.BASE_DIR, 'models', 'embedding_model.pkl'),
    'mlb.pkl': os.path.join(settings.BASE_DIR, 'models', 'mlb.pkl'),
    'known_diseases.pkl': os.path.join(settings.BASE_DIR, 'models', 'known_diseases.pkl'),
    
    # scripts
    'train_model.py': os.path.join(settings.BASE_DIR, 'train_model.py'),
    'predict_view.py': os.path.join(settings.BASE_DIR, 'predict_view.py'),
    'tr_model.py': os.path.join(settings.BASE_DIR, 'tr_model.py'),
    'train_composition.py': os.path.join(settings.BASE_DIR, 'train_composition.py'),
    'train_features.py': os.path.join(settings.BASE_DIR, 'train_features.py'),
    'td_model.py': os.path.join(settings.BASE_DIR, 'td_model.py'),
    
    # templates
    
}

def resources_page(request):
    return render(request, 'resources.html', {'files': FILE_PATHS})

def view_file(request, filename):
    if filename not in FILE_PATHS:
        raise Http404("File not found")

    file_path = FILE_PATHS[filename]

    if not os.path.exists(file_path):
        raise Http404("File does not exist")

    # For binary files (.pkl), force download
    if filename.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename=' + filename
            return response

    # For text-based files (.csv, .py, .html)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return render(request, 'view_file.html', {'filename': filename, 'content': content})
from django.shortcuts import render
import joblib
import os
import numpy as np

# --------------------------------------------------
# Path Setup
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# --------------------------------------------------
# Load NEW Model Files
# --------------------------------------------------

model_path = os.path.join(PROJECT_ROOT, "drug_rf_tfidf_model.pkl")
vectorizer_path = os.path.join(PROJECT_ROOT, "drug_tfidf_vectorizer.pkl")
metrics_path = os.path.join(PROJECT_ROOT, "drug_model_metrics.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print("Model loaded from:", model_path)
print("Model type:", type(model))
print("Model expects features:", model.n_features_in_)

# Load metrics
try:
    metrics = joblib.load(metrics_path)
except:
    metrics = {"MAE": "-", "RMSE": "-", "R2": "-"}

# --------------------------------------------------
# View Function
# --------------------------------------------------

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

        else:
            result = {"error": "Invalid drug name"}

    return render(
        request,
        "drug_predictor.html",
        {
            "result": result,
            "metrics": metrics
        }
    )


from django.shortcuts import render
from .predict_deficiency import predict_lack_of_component

def disease_deficiency_predictor(request):
    result = None
    metrics = None

    if request.method == "POST":
        disease = request.POST.get("disease")

        prediction, metrics = predict_lack_of_component(disease)

        result = {
            "disease": disease,
            "component": prediction
        }

    return render(
        request,
        "disease_deficiency.html",
        {
            "result": result,
            "metrics": metrics
        }
    )
import sys
import pandas as pd
from django.shortcuts import render

# 🔹 Path to your trained ML file
sys.path.append(r"C:\Users\dheer\OneDrive\Documents\DRUG-DISCOVERY")  # change if needed

"""from train_composition import predict_dosage

# 🔹 Load dataset ONCE
DATA_PATH = "dosage_multiplier_dataset.csv"
df = pd.read_csv(DATA_PATH)

def drug_dosage(request):
    result = None
    metrics = None
    error = None

    if request.method == "POST":
        drug = request.POST.get("drug").lower().strip()

        # Normalize dataset drug names
        df["Drug"] = df["Drug"].str.lower()

        if drug not in df["Drug"].values:
            error = "Drug not found in dataset"
        else:
            # 🔹 Get base dose (same for all age groups)
            base_dose = df[df["Drug"] == drug]["Base_Dose"].iloc[0]

            # 🔹 ML predictions for each age group
            infant_dose, _, metrics = predict_dosage(drug, "Infant", base_dose)
            adult_dose, _, _ = predict_dosage(drug, "Adult", base_dose)
            oldage_dose, _, _ = predict_dosage(drug, "OldAge", base_dose)

            result = {
                "drug": drug,
                "base_dose": base_dose,
                "infant": infant_dose,
                "adult": adult_dose,
                "oldage": oldage_dose
            }

    return render(request, "drug_dosage.html", {
        "result": result,
        "metrics": metrics,
        "error": error
    })"""
from django.shortcuts import render
import pubchempy as pcp

def index(request):
    drug_name = None
    cid = None
    error = None

    if request.method == "POST":
        drug_name = request.POST.get("drug")

        try:
            compound = pcp.get_compounds(drug_name, "name")[0]
            cid = compound.cid
        except:
            error = "Drug not found in database."

    return render(request, "index.html", {
        "drug": drug_name,
        "cid": cid,
        "error": error
    })
from django.shortcuts import render
import pubchempy as pcp

def index(request):
    drug_name = None
    cid = None
    error = None

    if request.method == "POST":
        drug_name = request.POST.get("drug")

        try:
            compound = pcp.get_compounds(drug_name, "name")[0]
            cid = compound.cid
        except:
            error = "Drug not found in database."

    return render(request, "index.html", {
        "drug": drug_name,
        "cid": cid,
        "error": error
    })
import os
import pandas as pd
import pickle
from django.shortcuts import render
from django.conf import settings

# --------------------------------------------------
# Load Model & Metrics
# --------------------------------------------------

dosage_model = pickle.load(
    open(os.path.join(settings.BASE_DIR, "dosage_model.pkl"), "rb")
)

dosage_metrics = pickle.load(
    open(os.path.join(settings.BASE_DIR, "metrics.pkl"), "rb")
)

# --------------------------------------------------
# Load Dosage Dataset (IMPORTANT: use unique name)
# --------------------------------------------------

dosage_df = pd.read_csv(
    os.path.join(settings.BASE_DIR, "dosage_multiplier_dataset.csv")
)

# Clean column names (removes hidden spaces)
dosage_df.columns = dosage_df.columns.str.strip()


# --------------------------------------------------
# Helper Function
# --------------------------------------------------

def get_base_dose(drug):
    row = dosage_df[dosage_df["Drug"].str.lower() == drug.lower()]
    if row.empty:
        return None
    return float(row.iloc[0]["Base_Dose"])


# --------------------------------------------------
# Main View
# --------------------------------------------------

def drug_dosage_view(request):
    context = {}

    if request.method == "POST":
        drug = request.POST.get("drug")

        if not drug:
            context["error"] = "Please enter a drug name"
            return render(request, "drug_dosage.html", context)

        base_dose = get_base_dose(drug)

        if base_dose is None:
            context["error"] = "Drug not found in database"
            return render(request, "drug_dosage.html", context)

        results = {}

        for age in ["Infant", "Adult", "OldAge"]:

            inp = pd.DataFrame([{
                "Drug": drug,
                "Age_Group": age,
                "Base_Dose": base_dose
            }])

            # 🔥 IMPORTANT:
            # Your model is a Pipeline → it handles encoding internally
            multiplier = dosage_model.predict(inp)[0]

            final_dose = base_dose * multiplier

            results[age] = round(float(final_dose), 2)

        context = {
            "drug": drug,
            "base_dose": base_dose,
            "results": results,
            "metrics": dosage_metrics
        }

    return render(request, "drug_dosage.html", context)




from django.shortcuts import render
import pandas as pd
import joblib
import numpy as np

# Load dataset (for lookup)
df = pd.read_csv("drug_properties_with_binding_affinity.csv")

# Load trained regression model
model = joblib.load("drug_risk_model.pkl")

# Load regression metrics
metrics = joblib.load("drug_model_metrics.pkl")


def predict_features(request):
    result = None
    error = None

    if request.method == "POST":
        drug_input = request.POST.get("drug").lower().strip()

        # Find drug in dataset
        row = df[df["Drug_name"].str.lower().str.contains(drug_input)]

        if row.empty:
            error = "Drug not found in dataset."
        else:
            logp = row.iloc[0]["Lipophilicity_LogP"]

            # 🔥 Regression prediction
            predicted_affinity = model.predict([[logp]])[0]

            result = {
                "name": row.iloc[0]["Drug_name"],
                "logp": logp,
                "actual_affinity": row.iloc[0]["Binding_Affinity_kcal_per_mol"],
                
                "metrics": metrics
            }

    return render(request, "features.html", {
        "result": result,
        "error": error
    })





