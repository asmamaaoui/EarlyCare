
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


# Get the Pipeline components
if not hasattr(model, "named_steps"):
    raise ValueError("The loaded model is not a scikit-learn Pipeline")

if "preprocessor" not in model.named_steps:
    raise ValueError("Pipeline step 'preprocessor' not found")

preprocessor = model.named_steps["preprocessor"]

if "classifier" in model.named_steps:
    classifier = model.named_steps["classifier"]
else:
    classifier = model.steps[-1][1]


print("Model loaded successfully!")
print("Number of features:", len(FEATURES))
print("Model classes:", CLASSES)
print("Preprocessor:", type(preprocessor).__name__)
print("Classifier:", type(classifier).__name__)


# =====================================
# 2. FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="Student Dropout Prediction API",
    description="Student Dropout Prediction with SHAP Explainability",
    version="3.0.0"
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
# 4. SHAP CONFIGURATION
# =====================================

SHAP_EXPLAINER = None
SHAP_DISABLED = False
SHAP_ERROR = None
TRANSFORMED_FEATURE_GROUPS = None

SHAP_BATCH_SIZE = 16


def get_shap_explainer():

    global SHAP_EXPLAINER

    if SHAP_DISABLED:
        raise RuntimeError(
            "SHAP is disabled because a previous explanation failed"
        )

    if SHAP_EXPLAINER is None:

        SHAP_EXPLAINER = shap.TreeExplainer(
            classifier
        )

        print("SHAP TreeExplainer initialized successfully")

    return SHAP_EXPLAINER


# =====================================
# 5. MAP TRANSFORMED FEATURES
#    TO ORIGINAL FEATURES
# =====================================

def get_transformed_feature_groups():

    global TRANSFORMED_FEATURE_GROUPS

    if TRANSFORMED_FEATURE_GROUPS is not None:
        return TRANSFORMED_FEATURE_GROUPS

    transformed_names = list(
        preprocessor.get_feature_names_out()
    )

    groups = {
        feature: []
        for feature in FEATURES
    }

    for index, transformed_name in enumerate(
        transformed_names
    ):

        # Remove ColumnTransformer prefix
        clean_name = transformed_name.split(
            "__",
            1
        )[-1]

        # Find the original feature that generated this column
        matches = [
            feature
            for feature in FEATURES
            if (
                clean_name == feature
                or clean_name.startswith(feature + "_")
            )
        ]

        if not matches:

            raise ValueError(
                "Cannot map transformed feature "
                f"'{transformed_name}' to an original feature"
            )

        # Longest match avoids ambiguous feature names
        original_feature = max(
            matches,
            key=len
        )

        groups[original_feature].append(index)

    empty_groups = [
        feature
        for feature, indices in groups.items()
        if not indices
    ]

    if empty_groups:

        raise ValueError(
            f"Some original features have no transformed columns: "
            f"{empty_groups}"
        )

    TRANSFORMED_FEATURE_GROUPS = groups

    print(
        "Transformed features mapped:",
        len(transformed_names)
    )

    return groups


# =====================================
# 6. HELPER FUNCTIONS
# =====================================

def get_dropout_probability(probabilities):

    return round(
        float(probabilities[DROPOUT_CLASS_INDEX]),
        4
    )


def get_direction(shap_value):

    if shap_value > 0:

        return {
            "direction": "increases",
            "direction_label": "Augmente le risque prédit"
        }

    if shap_value < 0:

        return {
            "direction": "decreases",
            "direction_label": "Diminue le risque prédit"
        }

    return {
        "direction": "neutral",
        "direction_label": "Influence neutre"
    }


def json_safe_value(value):

    if hasattr(value, "item"):
        value = value.item()

    if pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    return value


# =====================================
# 7. EXTRACT DROPOUT CLASS SHAP VALUES
# =====================================

def extract_dropout_shap_values(shap_values, n_rows, n_features):

    if isinstance(shap_values, list):

        if DROPOUT_CLASS_INDEX >= len(shap_values):
            raise ValueError("Dropout SHAP class not found")

        result = np.asarray(
            shap_values[DROPOUT_CLASS_INDEX]
        )

    else:

        result = np.asarray(shap_values)

        if result.ndim == 3:

            # Newer SHAP format:
            # (samples, features, classes)
            if (
                result.shape[0] == n_rows
                and result.shape[1] == n_features
                and result.shape[2] == len(CLASSES)
            ):

                result = result[
                    :,
                    :,
                    DROPOUT_CLASS_INDEX
                ]

            # Alternative format:
            # (classes, samples, features)
            elif (
                result.shape[0] == len(CLASSES)
                and result.shape[1] == n_rows
                and result.shape[2] == n_features
            ):

                result = result[
                    DROPOUT_CLASS_INDEX,
                    :,
                    :
                ]

            else:

                raise ValueError(
                    f"Unsupported 3D SHAP shape: {result.shape}"
                )

    if result.ndim != 2:

        raise ValueError(
            f"Expected 2D SHAP values, got {result.shape}"
        )

    if result.shape != (n_rows, n_features):

        raise ValueError(
            f"Unexpected SHAP shape: {result.shape}; "
            f"expected {(n_rows, n_features)}"
        )

    return result


# =====================================
# 8. RECOMMENDATIONS
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
# 9. SHAP EXPLANATION
# =====================================

def explain_students(X):

    global SHAP_DISABLED
    global SHAP_ERROR

    explanations = []

    if SHAP_DISABLED:

        return [
            {
                "method": "SHAP unavailable",
                "summary": "L'explication SHAP est temporairement indisponible.",
                "top_factors": [],
                "recommendations": [],
                "disclaimer": (
                    "La prédiction reste disponible. "
                    "L'explication SHAP n'a pas pu être calculée."
                )
            }
            for _ in range(len(X))
        ]

    try:

        explainer = get_shap_explainer()

        groups = get_transformed_feature_groups()

        for start in range(
            0,
            len(X),
            SHAP_BATCH_SIZE
        ):

            end = min(
                start + SHAP_BATCH_SIZE,
                len(X)
            )

            batch = X.iloc[start:end]

            # Apply the exact preprocessing used by the Pipeline
            transformed = preprocessor.transform(batch)

            # SHAP TreeExplainer receives the transformed matrix
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            transformed = np.asarray(
                transformed,
                dtype=np.float64
            )

            raw_shap_values = explainer.shap_values(
                transformed,
                check_additivity=False
            )

            shap_matrix = extract_dropout_shap_values(
                raw_shap_values,
                len(batch),
                transformed.shape[1]
            )

            # Aggregate one-hot columns into original features
            for local_index in range(len(batch)):

                row_index = start + local_index

                factors = []

                for feature in FEATURES:

                    indices = groups[feature]

                    # Sum contributions of all encoded columns
                    aggregated_value = float(
                        np.sum(
                            shap_matrix[
                                local_index,
                                indices
                            ]
                        )
                    )

                    raw_value = json_safe_value(
                        X.iloc[row_index][feature]
                    )

                    direction = get_direction(
                        aggregated_value
                    )

                    factors.append({

                        "feature": feature,

                        "label": FEATURE_LABELS.get(
                            feature,
                            feature
                        ),

                        "value": raw_value,

                        "shap_value": round(
                            aggregated_value,
                            6
                        ),

                        "direction": direction["direction"],

                        "direction_label": direction["direction_label"]

                    })

                factors.sort(
                    key=lambda item: abs(item["shap_value"]),
                    reverse=True
                )

                top_factors = factors[:5]

                summary = (
                    "Les facteurs ayant le plus contribué "
                    "à la prédiction du risque de dropout sont : "
                    + ", ".join(
                        item["label"]
                        for item in top_factors[:3]
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

    except Exception as error:

        SHAP_DISABLED = True
        SHAP_ERROR = str(error)

        print("SHAP explanation failed:", repr(error))
        print("SHAP has been disabled for this process.")

        return [
            {
                "method": "SHAP unavailable",
                "summary": (
                    "L'explication SHAP est temporairement "
                    "indisponible en raison d'une erreur technique."
                ),
                "top_factors": [],
                "recommendations": [],
                "disclaimer": (
                    "La prédiction reste disponible, "
                    "mais les explications ne sont pas disponibles."
                )
            }
            for _ in range(len(X))
        ]


# =====================================
# 10. STUDENT VALIDATION
# =====================================

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
# 11. INDIVIDUAL PREDICTION
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
# 12. HEALTH CHECK
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

        "xai_loaded": (
            SHAP_EXPLAINER is not None
            and not SHAP_DISABLED
        ),

        "xai_method": "SHAP TreeExplainer",

        "xai_error": SHAP_ERROR,

        "number_of_features": len(FEATURES),

        "classes": [
            str(c) for c in CLASSES
        ]

    }


# =====================================
# 13. REQUEST MODEL
# =====================================

class StudentData(BaseModel):

    data: dict[str, Any]


# =====================================
# 14. INDIVIDUAL PREDICTION ENDPOINT
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

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print("Prediction error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs."
        )


# =====================================
# 15. CSV BATCH PREDICTION
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

        # Use the exact original feature order
        X = df[FEATURES].copy()

        # Predict all students
        predictions = model.predict(X)

        probabilities = model.predict_proba(X)

        # Explain in small batches
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

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print("CSV prediction error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="CSV prediction failed. Check server logs."
        )

    finally:

        await file.close()

