"""
API de inferencia para el clasificador de ingresos.
Permite a cualquier usuario enviar datos de una persona
y recibir la predicción de si sus ingresos superan los 50K anuales.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

# ──────────────────────────────────────────────
# Cargar artefactos al iniciar
# ──────────────────────────────────────────────
try:
    model    = joblib.load(MODEL_DIR / "best_model.pkl")
    scaler   = joblib.load(MODEL_DIR / "scaler.pkl")
    encoders = joblib.load(MODEL_DIR / "encoders.pkl")
except FileNotFoundError:
    model = scaler = encoders = None  # Se maneja en el endpoint

# ──────────────────────────────────────────────
# Schema de entrada
# ──────────────────────────────────────────────
class PersonaInput(BaseModel):
    edad: int = Field(..., ge=17, le=90, example=39, description="Edad en años")
    clase_trabajo: Literal[
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ] = Field(..., example="State-gov")
    fnlwgt: int = Field(..., ge=0, example=77516, description="Peso censal final")
    educacion: Literal[
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
        "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
        "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
    ] = Field(..., example="Bachelors")
    educacion_num: int = Field(..., ge=1, le=16, example=13)
    estado_civil: Literal[
        "Married-civ-spouse", "Divorced", "Never-married", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ] = Field(..., example="Never-married")
    ocupacion: Literal[
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
    ] = Field(..., example="Adm-clerical")
    relacion: Literal[
        "Wife", "Own-child", "Husband", "Not-in-family",
        "Other-relative", "Unmarried"
    ] = Field(..., example="Not-in-family")
    raza: Literal[
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
        "Other", "Black"
    ] = Field(..., example="White")
    genero: Literal["Male", "Female"] = Field(..., example="Male")
    ganancia_capital: float = Field(..., ge=0, example=2174.0)
    perdida_capital: float = Field(..., ge=0, example=0.0)
    horas_por_semana: int = Field(..., ge=1, le=99, example=40)
    pais_nativo: str = Field(..., example="United-States")


class PrediccionOutput(BaseModel):
    prediccion: int
    etiqueta: str
    probabilidad_mayor_50k: float
    mensaje: str


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(
    title="Clasificador de Ingresos",
    description=(
        "API que predice si los ingresos anuales de una persona "
        "superan los USD 50.000, basándose en sus características personales y laborales."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COLUMNAS_NUMERICAS = [
    "age", "fnlwgt", "educational-num",
    "capital-gain", "capital-loss", "hours-per-week"
]
COLUMNAS_CATEGORICAS = [
    "workclass", "education", "marital-status",
    "occupation", "relationship", "race", "gender", "native-country"
]


def preprocesar(persona: PersonaInput) -> pd.DataFrame:
    """Convierte el input del usuario al formato esperado por el modelo."""
    # Mapear nombres al formato original del dataset
    raw = {
        "age":             persona.edad,
        "fnlwgt":          persona.fnlwgt,
        "educational-num": persona.educacion_num,
        "capital-gain":    persona.ganancia_capital,
        "capital-loss":    persona.perdida_capital,
        "hours-per-week":  persona.horas_por_semana,
        "workclass":       persona.clase_trabajo,
        "education":       persona.educacion,
        "marital-status":  persona.estado_civil,
        "occupation":      persona.ocupacion,
        "relationship":    persona.relacion,
        "race":            persona.raza,
        "gender":          persona.genero,
        "native-country":  persona.pais_nativo,
    }
    df = pd.DataFrame([raw])

    # Escalar numéricas
    df[COLUMNAS_NUMERICAS] = scaler.transform(df[COLUMNAS_NUMERICAS])

    # Encodificar categóricas
    for col in COLUMNAS_CATEGORICAS:
        le = encoders[col]
        val = df[col].iloc[0]
        if val not in le.classes_:
            # Si el valor no fue visto en entrenamiento, usar la moda
            df[col] = le.transform([le.classes_[0]])
        else:
            df[col] = le.transform([val])

    return df[COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    return {
        "servicio": "Clasificador de Ingresos",
        "estado": "activo",
        "modelo_cargado": model is not None,
    }


@app.post("/predecir", response_model=PrediccionOutput, summary="Clasificar ingresos")
def predecir(persona: PersonaInput):
    """
    Recibe las características de una persona y devuelve:
    - **prediccion**: 0 (≤50K) o 1 (>50K)
    - **etiqueta**: descripción legible
    - **probabilidad_mayor_50k**: confianza del modelo
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Ejecuta el pipeline primero."
        )

    try:
        X = preprocesar(persona)
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error en preprocesamiento: {e}")

    etiqueta = ">50K anuales" if pred == 1 else "≤50K anuales"
    mensaje = (
        f"El modelo estima que esta persona {'SÍ' if pred == 1 else 'NO'} "
        f"supera los 50.000 USD anuales "
        f"(confianza: {prob * 100:.1f}%)."
    )

    return PrediccionOutput(
        prediccion=pred,
        etiqueta=etiqueta,
        probabilidad_mayor_50k=round(prob, 4),
        mensaje=mensaje,
    )
