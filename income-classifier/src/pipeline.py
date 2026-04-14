"""
Pipeline de ML para clasificación de ingresos de adultos.
Usa Prefect para orquestación y MLflow para tracking.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
from pathlib import Path

from prefect import task, flow
from prefect.logging import get_run_logger

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, recall_score, f1_score,
    precision_score, accuracy_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
DATA_PATH = Path(os.getenv("DATA_PATH", "data/adult.data.csv"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "income-classifier"
RANDOM_STATE = 42

# ──────────────────────────────────────────────
# TASK 1 – CARGA DE DATOS
# ──────────────────────────────────────────────
@task(name="cargar_datos", retries=2, retry_delay_seconds=5)
def cargar_datos(path: Path = DATA_PATH) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Cargando datos desde: {path}")

    column_names = [
        "age", "workclass", "fnlwgt", "education", "educational-num",
        "marital-status", "occupation", "relationship", "race", "gender",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    df = pd.read_csv(path, header=None, names=column_names, skiprows=1)
    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


# ──────────────────────────────────────────────
# TASK 2 – LIMPIEZA Y PREPROCESAMIENTO
# ──────────────────────────────────────────────
@task(name="limpiar_datos")
def limpiar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    logger = get_run_logger()
    logger.info("Iniciando limpieza de datos...")

    # Limpiar espacios en strings
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Reemplazar '?' por NaN y eliminar esos registros
    df.replace("?", np.nan, inplace=True)
    filas_antes = len(df)
    df.dropna(inplace=True)
    filas_despues = len(df)
    logger.info(f"Eliminados {filas_antes - filas_despues} registros con valores faltantes")

    # Eliminar duplicados
    df.drop_duplicates(inplace=True)
    logger.info(f"Registros finales tras limpieza: {len(df)}")

    # Codificar variable objetivo: '>50K' -> 1, '<=50K' -> 0
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

    # Columnas a usar (según EDA)
    columnas_numericas = [
        "age", "fnlwgt", "educational-num",
        "capital-gain", "capital-loss", "hours-per-week"
    ]
    columnas_categoricas = [
        "workclass", "education", "marital-status",
        "occupation", "relationship", "race", "gender", "native-country"
    ]

    # Convertir numéricas
    for col in columnas_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Codificar categóricas con LabelEncoder
    encoders = {}
    for col in columnas_categoricas:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Escalar numéricas
    scaler = MinMaxScaler()
    df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])

    # Guardar artefactos de preprocesamiento
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(encoders, MODEL_DIR / "encoders.pkl")

    feature_cols = columnas_numericas + columnas_categoricas
    metadata = {
        "feature_cols": feature_cols,
        "target_col": "income",
        "n_records": len(df),
        "class_distribution": df["income"].value_counts().to_dict(),
    }

    logger.info(f"Distribución de clases: {metadata['class_distribution']}")
    return df, metadata


# ──────────────────────────────────────────────
# TASK 3 – ENTRENAMIENTO DE MODELOS
# ──────────────────────────────────────────────
@task(name="entrenar_modelos")
def entrenar_modelos(df: pd.DataFrame, metadata: dict) -> list[dict]:
    logger = get_run_logger()

    feature_cols = metadata["feature_cols"]
    target_col = metadata["target_col"]

    X = df[feature_cols]
    y = df[target_col]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Balanceo con SMOTE (sólo en train)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    logger.info(f"Después de SMOTE – train size: {len(X_train_bal)}")

    # Definir modelos candidatos
    candidatos = [
        {
            "nombre": "RandomForest",
            "modelo": RandomForestClassifier(
                n_estimators=200, max_depth=15,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ),
            "params": {"n_estimators": 200, "max_depth": 15}
        },
        {
            "nombre": "GradientBoosting",
            "modelo": GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1,
                max_depth=5, random_state=RANDOM_STATE
            ),
            "params": {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 5}
        },
        {
            "nombre": "XGBoost",
            "modelo": xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                use_label_encoder=False, eval_metric="logloss",
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            "params": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6}
        },
    ]

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    resultados = []

    for candidato in candidatos:
        nombre = candidato["nombre"]
        logger.info(f"Entrenando: {nombre}")

        with mlflow.start_run(run_name=nombre):
            # Loggear parámetros
            mlflow.log_params(candidato["params"])

            # Entrenar
            modelo = candidato["modelo"]
            modelo.fit(X_train_bal, y_train_bal)

            # Evaluar en test
            y_pred = modelo.predict(X_test)

            metricas = {
                "accuracy":          accuracy_score(y_test, y_pred),
                "recall_class1":     recall_score(y_test, y_pred, pos_label=1),
                "precision_class1":  precision_score(y_test, y_pred, pos_label=1),
                "f1_class1":         f1_score(y_test, y_pred, pos_label=1),
            }

            # Loggear métricas en MLflow
            mlflow.log_metrics(metricas)

            # Loggear modelo en MLflow
            mlflow.sklearn.log_model(modelo, artifact_path="model")

            run_id = mlflow.active_run().info.run_id

        resultados.append({
            "nombre": nombre,
            "modelo": modelo,
            "metricas": metricas,
            "run_id": run_id,
        })

        logger.info(f"  {nombre} – recall clase 1: {metricas['recall_class1']:.4f}")

    return resultados


# ──────────────────────────────────────────────
# TASK 4 – SELECCIÓN DEL MEJOR MODELO
# ──────────────────────────────────────────────
@task(name="seleccionar_mejor_modelo")
def seleccionar_mejor_modelo(resultados: list[dict]) -> dict:
    logger = get_run_logger()

    # Ordenar por recall de clase 1 (el criterio del taller)
    mejor = max(resultados, key=lambda r: r["metricas"]["recall_class1"])

    logger.info(f"🏆 Mejor modelo: {mejor['nombre']}")
    logger.info(f"   Recall clase 1 : {mejor['metricas']['recall_class1']:.4f}")
    logger.info(f"   F1 clase 1     : {mejor['metricas']['f1_class1']:.4f}")
    logger.info(f"   Accuracy       : {mejor['metricas']['accuracy']:.4f}")
    logger.info(f"   MLflow run_id  : {mejor['run_id']}")

    return mejor


# ──────────────────────────────────────────────
# TASK 5 – GUARDAR EL MODELO
# ──────────────────────────────────────────────
@task(name="guardar_modelo")
def guardar_modelo(mejor: dict) -> Path:
    logger = get_run_logger()

    MODEL_DIR.mkdir(exist_ok=True)
    ruta = MODEL_DIR / "best_model.pkl"
    joblib.dump(mejor["modelo"], ruta)

    # También registrar en MLflow Model Registry
    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"runs:/{mejor['run_id']}/model"
    mlflow.register_model(model_uri, name="income-classifier-best")

    logger.info(f"Modelo guardado en: {ruta}")
    logger.info(f"Modelo registrado en MLflow como 'income-classifier-best'")
    return ruta


# ──────────────────────────────────────────────
# FLOW PRINCIPAL
# ──────────────────────────────────────────────
@flow(name="income-classifier-pipeline", log_prints=True)
def pipeline_principal(data_path: str = str(DATA_PATH)):
    """
    Pipeline completo:
    1. Carga de datos
    2. Limpieza y preprocesamiento
    3. Entrenamiento de múltiples modelos con MLflow
    4. Selección del mejor modelo por recall clase 1
    5. Guardado del modelo candidato a producción
    """
    # 1. Cargar
    df_raw = cargar_datos(Path(data_path))

    # 2. Limpiar
    df_clean, metadata = limpiar_datos(df_raw)

    # 3. Entrenar
    resultados = entrenar_modelos(df_clean, metadata)

    # 4. Seleccionar
    mejor = seleccionar_mejor_modelo(resultados)

    # 5. Guardar
    ruta_modelo = guardar_modelo(mejor)

    print(f"\n✅ Pipeline completado. Modelo en: {ruta_modelo}")
    print(f"   Mejor modelo : {mejor['nombre']}")
    print(f"   Recall cl. 1 : {mejor['metricas']['recall_class1']:.4f}")
    return mejor


if __name__ == "__main__":
    pipeline_principal()
