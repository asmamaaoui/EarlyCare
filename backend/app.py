from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

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

# SHAP explainer for Random Forest
# استخراج آخر عنصر داخل Pipeline
if hasattr(model, "steps"):
    final_estimator = model.steps[-1][1]
else:
    final_estimator = model

# إنشاء SHAP Explainer باستعمال الـ estimator النهائي
explainer = shap.TreeExplainer(final_estimator)
print("Model loaded successfully!")
print("Number of features:", len(FEATURES))
print("Model classes:", CLASSES)


# =====================================
# 2. FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="Student Dropout Prediction API",
    description="Student Dropout Prediction with SHAP Explainability",
    version="2.0.0"
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


def get_shap_vector(shap_values, row_index):

    if isinstance(shap_values, list):

        return np.asarray(
            shap_values[DROPOUT_CLASS_INDEX][row_index]
        ).reshape(-1)

    values = np.asarray(shap_values)

    if values.ndim == 3:

        if values.shape[-1] == len(CLASSES):

            return values[
                row_index,
                :,
                DROPOUT_CLASS_INDEX
            ].reshape(-1)

        if values.shape[0] == len(CLASSES):

            return values[
                DROPOUT_CLASS_INDEX,
                row_index,
                :
            ].reshape(-1)

    if values.ndim == 2:

        return values[row_index].reshape(-1)

    raise ValueError(
        f"Unsupported SHAP shape: {values.shape}"
    )


def get_direction(shap_value):

    if shap_value > 0:

        return {
            "direction": "increases",
            "direction_label": "Augmente le risque prédit"
        }

    elif shap_value < 0:

        return {
            "direction": "decreases",
            "direction_label": "Diminue le risque prédit"
        }

    return {
        "direction": "neutral",
        "direction_label": "Influence neutre"
    }


# =====================================
# 5. RECOMMENDATIONS
# =====================================

def generate_recommendations(top_factors):

    recommendations = []

    for factor in top_factors:

        if factor["direction"] != "increases":
            continue

        feature = factor["feature"]

        if feature in [
            "Success_Rate",
            "Curricular units 1st sem (approved)"
        ]:

            recommendation = (
                "Proposer un accompagnement pédagogique "
                "et du tutorat dans les matières difficiles."
            )

        elif feature == "Curricular units 1st sem (without evaluations)":

            recommendation = (
                "Contacter l'étudiant pour comprendre les absences "
                "aux évaluations et identifier les difficultés."
            )

        elif feature in [
            "Curricular units 1st sem (grade)",
            "Admission grade",
            "Previous qualification (grade)"
        ]:

            recommendation = (
                "Évaluer les besoins de remise à niveau "
                "et proposer un soutien académique adapté."
            )

        elif feature in [
            "Debtor",
            "Tuition fees up to date"
        ]:

            recommendation = (
                "Vérifier si l'étudiant rencontre des difficultés "
                "financières et l'orienter vers le service compétent."
            )

        elif feature == "Age at enrollment":

            recommendation = (
                "Organiser un échange individuel pour identifier "
                "les contraintes personnelles ou professionnelles."
            )

        else:

            recommendation = (
                "Examiner ce facteur avec l'étudiant afin de vérifier "
                "s'il existe un besoin particulier d'accompagnement."
            )

        if recommendation not in recommendations:

            recommendations.append(recommendation)

    if not recommendations:

        recommendations.append(
            "Maintenir le suivi pédagogique régulier "
            "et vérifier périodiquement la situation de l'étudiant."
        )

    return recommendations[:4]


# =====================================
# 6. SHAP EXPLANATION
# =====================================

def explain_students(X):

    shap_values = explainer.shap_values(
        X,
        check_additivity=False
    )

    explanations = []

    for row_index in range(len(X)):

        values = get_shap_vector(
            shap_values,
            row_index
        )

        if len(values) != len(FEATURES):

            raise ValueError(
                "SHAP feature count does not match model features"
            )

        factors = []

        for feature, shap_value in zip(FEATURES, values):

            raw_value = X.iloc[row_index][feature]

            if hasattr(raw_value, "item"):
                raw_value = raw_value.item()

            direction = get_direction(
                float(shap_value)
            )

            factors.append({

                "feature": feature,

                "label": FEATURE_LABELS.get(
                    feature,
                    feature
                ),

                "value": raw_value,

                "shap_value": round(
                    float(shap_value),
                    6
                ),

                "direction": direction["direction"],

                "direction_label": direction["direction_label"]

            })

        # Sort by absolute SHAP contribution
        factors.sort(
            key=lambda x: abs(x["shap_value"]),
            reverse=True
        )

        top_factors = factors[:5]

        summary = (
            "Les facteurs ayant le plus contribué "
            "à la prédiction sont : "
            + ", ".join(
                factor["label"]
                for factor in top_factors[:3]
            )
            + "."
        )

        recommendations = generate_recommendations(
            top_factors
        )

        explanations.append({

            "method": "SHAP TreeExplainer",

            "summary": summary,

            "top_factors": top_factors,

            "recommendations": recommendations,

            "disclaimer": (
                "Les valeurs SHAP expliquent la contribution "
                "des variables à la prédiction du modèle. "
                "Elles ne prouvent pas une relation de cause à effet. "
                "Les recommandations sont des pistes d'accompagnement."
            )

        })

    return explanations


# =====================================
# 7. STUDENT VALIDATION
# =====================================

def validate_student_data(data):

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
# 8. INDIVIDUAL PREDICTION
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
# 9. HEALTH CHECK
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

        "xai_loaded": True,

        "xai_method": "SHAP TreeExplainer",

        "number_of_features": len(FEATURES),

        "classes": [
            str(c) for c in CLASSES
        ]

    }


# =====================================
# 10. INDIVIDUAL PREDICTION ENDPOINT
# =====================================

class StudentData(BaseModel):

    data: dict[str, Any]


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
# 11. CSV BATCH PREDICTION
# =====================================

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".csv"):

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

        X = df[FEATURES].copy()

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)

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