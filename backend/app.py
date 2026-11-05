
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =====================================
# 1. LOAD TRAINED MODEL
# =====================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_rf_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

FEATURES = list(model.feature_names_in_)

if len(FEATURES) != 31:
    raise ValueError(
        f"Expected 31 features, found {len(FEATURES)}"
    )

CLASSES = list(model.classes_)

DROPOUT_CLASS_INDEX = next(
    (
        i for i, label in enumerate(CLASSES)
        if int(label) == 1
    ),
    None
)

if DROPOUT_CLASS_INDEX is None:
    raise ValueError("Dropout class 1 not found")

print("Model loaded successfully!")
print("Number of features:", len(FEATURES))
print("Model classes:", CLASSES)


# =====================================
# 2. FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="Student Dropout Prediction API",
    description=(
        "Student Dropout Prediction API "
        "(SHAP temporarily disabled for diagnosis)"
    ),
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================
# 3. FEATURE LABELS
# =====================================

FEATURE_LABELS = {
    "Marital status": "Situation matrimoniale",
    "Application mode": "Mode de candidature",
    "Application order": "Ordre de candidature",
    "Course": "Filière",
    "Daytime/evening attendance\t": "Régime d'études",
    "Previous qualification": "Diplôme antérieur",
    "Previous qualification (grade)": "Note du diplôme antérieur",
    "Nacionality": "Nationalité",
    "Mother's qualification": "Niveau d'études de la mère",
    "Father's qualification": "Niveau d'études du père",
    "Mother's occupation": "Profession de la mère",
    "Father's occupation": "Profession du père",
    "Admission grade": "Note d'admission",
    "Displaced": "Étudiant déplacé",
    "Educational special needs": "Besoins éducatifs spécifiques",
    "Debtor": "Situation de dette",
    "Tuition fees up to date": "Frais de scolarité à jour",
    "Gender": "Genre",
    "Scholarship holder": "Bénéficiaire d'une bourse",
    "Age at enrollment": "Âge à l'inscription",
    "International": "Étudiant international",
    "Curricular units 1st sem (credited)": "Unités créditées",
    "Curricular units 1st sem (enrolled)": "Unités inscrites",
    "Curricular units 1st sem (evaluations)": "Nombre d'évaluations",
    "Curricular units 1st sem (approved)": "Unités validées",
    "Curricular units 1st sem (grade)": "Moyenne du semestre",
    "Curricular units 1st sem (without evaluations)": "Unités sans évaluation",
    "Unemployment rate": "Taux de chômage",
    "Inflation rate": "Taux d'inflation",
    "GDP": "PIB",
    "Success_Rate": "Taux de réussite"
}


# =====================================
# 4. HELPER FUNCTIONS
# =====================================

def get_dropout_probability(probabilities):

    return round(
        float(probabilities[DROPOUT_CLASS_INDEX]),
        4
    )


def validate_student_data(data):

    if not isinstance(data, dict):
        raise ValueError("Student data must be a dictionary")

    missing = [
        col for col in FEATURES
        if col not in data
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    extra = [
        col for col in data
        if col not in FEATURES
    ]

    if extra:
        raise ValueError(
            f"Unexpected columns: {extra}"
        )


# =====================================
# 5. TEMPORARY XAI PLACEHOLDER
# =====================================

def explain_students(X):

    explanations = []

    for _ in range(len(X)):

        explanations.append({

            "method": "SHAP temporarily disabled",

            "summary": (
                "L'explication SHAP est temporairement "
                "désactivée pour le diagnostic technique."
            ),

            "top_factors": [],

            "recommendations": [],

            "disclaimer": (
                "La prédiction est générée par le modèle "
                "Random Forest. Les explications SHAP "
                "seront réactivées après résolution "
                "du problème technique."
            )

        })

    return explanations


# =====================================
# 6. INDIVIDUAL PREDICTION LOGIC
# =====================================

def predict_student(data):

    validate_student_data(data)

    df = pd.DataFrame(
        [[data[col] for col in FEATURES]],
        columns=FEATURES
    )

    prediction = model.predict(df)[0]

    probabilities = model.predict_proba(df)[0]

    probability_dict = {
        str(label): round(float(prob), 4)
        for label, prob in zip(
            CLASSES,
            probabilities
        )
    }

    explanation = explain_students(df)[0]

    return {

        "prediction": str(prediction),

        "probabilities": probability_dict,

        "dropout_probability": get_dropout_probability(
            probabilities
        ),

        "explanation": explanation

    }


# =====================================
# 7. HEALTH CHECK
# =====================================

@app.get("/")
def home():

    return {
        "message": "Student Dropout Prediction API is running",
        "status": "OK"
    }


@app.get("/health")
def health():

    return {

        "model_loaded": True,

        "xai_loaded": False,

        "xai_method": "Temporarily disabled for diagnosis",

        "number_of_features": len(FEATURES),

        "classes": [
            str(c) for c in CLASSES
        ]

    }


# =====================================
# 8. REQUEST MODEL
# =====================================

class StudentData(BaseModel):

    data: dict[str, Any]


# =====================================
# 9. INDIVIDUAL PREDICTION ENDPOINT
# =====================================

@app.post("/predict")
def predict(data: StudentData):

    try:

        result = predict_student(
            data.data
        )

        return {

            "success": True,

            "result": result

        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =====================================
# 10. CSV BATCH PREDICTION
# =====================================

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):

    if (
        not file.filename
        or not file.filename.lower().endswith(".csv")
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file"
        )

    try:

        df = pd.read_csv(
            file.file,
            encoding="utf-8-sig",
            sep=None,
            engine="python"
        )

        # Remove BOM if present
        df.columns = [
            col.removeprefix("\ufeff")
            for col in df.columns
        ]

        missing = [
            col for col in FEATURES
            if col not in df.columns
        ]

        if missing:

            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Missing columns",
                    "columns": missing
                }
            )

        if df.empty:

            raise HTTPException(
                status_code=400,
                detail="The CSV file contains no student records"
            )

        # Keep the exact model feature order
        X = df[FEATURES].copy()

        # Predictions
        predictions = model.predict(X)

        # Class probabilities
        probabilities = model.predict_proba(X)

        # Temporary explanations without SHAP
        explanations = explain_students(X)

        results = []

        for i, (prediction, probs, explanation) in enumerate(
            zip(
                predictions,
                probabilities,
                explanations
            )
        ):

            probability_dict = {
                str(label): round(float(prob), 4)
                for label, prob in zip(
                    CLASSES,
                    probs
                )
            }

            results.append({

                "row": i + 1,

                "prediction": str(prediction),

                "probabilities": probability_dict,

                "dropout_probability": get_dropout_probability(
                    probs
                ),

                "explanation": explanation

            })

        return {

            "success": True,

            "total_students": len(results),

            "results": results

        }

    except HTTPException:

        raise

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        file.file.close()

