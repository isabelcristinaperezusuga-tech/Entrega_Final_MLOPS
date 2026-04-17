# 🤖 Clasificador de Ingresos de Adultos
### Proyecto Final MLOps — Entrega 2

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-2.13.0-0194E2?logo=mlflow)
![Prefect](https://img.shields.io/badge/Prefect-2.16.9-024DFD?logo=prefect)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![Estado](https://img.shields.io/badge/Estado-Completado-2ECC71)

> Sistema de ML que predice si una persona tiene ingresos anuales **superiores a USD 50.000**, orquestado con Prefect, tracked con MLflow y desplegado como API REST con FastAPI dentro de Docker.

**Autoras:** Isabel Cristina Pérez · Anderson Taborda Hernandez  
**Fecha:** Abril 2026

---

## 📋 Tabla de Contenidos

1. [Descripción del Problema](#-descripción-del-problema)
2. [Hallazgos del EDA](#-hallazgos-del-eda)
3. [Arquitectura del Sistema](#-arquitectura-del-sistema)
4. [Estructura del Proyecto](#-estructura-del-proyecto)
5. [Pipeline — Las 5 Tasks](#-pipeline--las-5-tasks)
6. [Modelos y Métricas](#-modelos-y-métricas)
7. [Decisiones Técnicas](#-decisiones-técnicas)
8. [Cómo Ejecutar el Proyecto](#-cómo-ejecutar-el-proyecto)
9. [URLs de los Servicios](#-urls-de-los-servicios)
10. [Ejemplo de Predicción](#-ejemplo-de-predicción)
11. [Integrantes](#-integrantes)

---

## 📊 Descripción del Problema

El dataset utilizado es el **UCI Adult Income**, con **48.842 registros** de adultos de Estados Unidos obtenidos del censo nacional. Cada registro contiene información demográfica y laboral de una persona.

**Objetivo:** Predecir si una persona tiene ingresos anuales **superiores a USD 50.000** basándose en sus características personales y laborales.

**Variable objetivo:**
- `0` → Ingresos ≤ 50K anuales
- `1` → Ingresos > 50K anuales

**Variables de entrada utilizadas:**

| Variable | Tipo | Descripción |
|---|---|---|
| age | Numérica | Edad en años |
| educational-num | Numérica | Años de educación (1–16) |
| capital-gain | Numérica | Ganancia de capital por inversiones |
| capital-loss | Numérica | Pérdida de capital por inversiones |
| hours-per-week | Numérica | Horas trabajadas por semana |
| workclass | Categórica | Tipo de empleador |
| education | Categórica | Nivel educativo |
| marital-status | Categórica | Estado civil |
| occupation | Categórica | Ocupación laboral |
| relationship | Categórica | Relación familiar |
| race | Categórica | Raza |
| gender | Categórica | Género |
| native-country | Categórica | País de origen |

> ⚠️ **Variable excluida:** `fnlwgt` (peso censal) fue eliminada por ser una variable administrativa del censo con bajo poder predictivo. Su inclusión no aportaba valor al modelo.

---

## 🔍 Hallazgos del EDA

El Análisis Exploratorio de Datos (EDA) reveló 6 insights clave que guiaron todas las decisiones de preprocesamiento y modelado:

### 1. ⚖️ Desbalance de Clases
El dataset tiene una distribución desigual en la variable objetivo:
- **75%** de los registros gana ≤ 50K
- **25%** de los registros gana > 50K

Esto implica que un modelo naive que siempre prediga ≤50K tendría 75% de accuracy sin aprender nada útil. Por esta razón se priorizó el **Recall de clase 1** como métrica principal y se aplicó **SMOTE** para balancear el entrenamiento.

### 2. ❓ Datos Faltantes Disfrazados
Tres variables contenían el carácter `?` representando valores faltantes:
- `workclass` → ~6% de registros
- `occupation` → ~6% de registros
- `native-country` → ~2% de registros

Se reemplazaron por `NaN` y se eliminaron, resultando en un dataset limpio de **45.222 registros**.

### 3. 🎓 Educación como Predictor Clave
La variable `educational-num` mostró una correlación positiva clara con los ingresos. A mayor nivel educativo, mayor probabilidad de superar los 50K anuales. Es uno de los predictores más fuertes del modelo.

### 4. 💹 Capital Gain Muy Sesgado
Más del **90% de los registros** tienen valor `0` en `capital-gain` y `capital-loss`. Sin embargo, cuando `capital-gain` es distinto de cero, es un predictor muy fuerte de ingresos altos — indica que la persona tiene inversiones activas.

### 5. 👤 Perfil Predominante
- Media de edad: **~38 años**
- Jornada laboral estándar: **40 horas semanales**
- Alta variabilidad en ocupación y horas trabajadas

### 6. 🏷️ Variables Categóricas Ricas
Variables como `occupation` y `native-country` tienen 14+ categorías únicas. Se utilizó **LabelEncoder** para convertirlas a valores numéricos, ya que es más eficiente que OneHotEncoder para modelos basados en árboles de decisión con muchas categorías.

---

## 🏗️ Arquitectura del Sistema

El sistema está compuesto por **4 servicios** que corren juntos dentro de Docker Compose:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Pipeline  │───▶│   MLflow    │    │   Prefect   │    │     API     │
│             │    │             │    │             │    │             │
│ Prefect Flow│    │ Tracking    │    │ Orquestación│    │  FastAPI    │
│ 5 Tasks ML  │    │ Model Reg.  │    │ Monitoreo   │    │ Inferencia  │
│             │    │             │    │             │    │             │
│ :pipeline   │    │ :5000       │    │ :4200       │    │ :8000       │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

| Servicio | Imagen | Puerto | Función |
|---|---|---|---|
| **pipeline** | Python 3.11 | — | Ejecuta el flujo de ML completo |
| **mlflow** | mlflow:v2.13.0 | 5000 | Tracking de experimentos y Model Registry |
| **prefect** | prefect:2-python3.11 | 4200 | Orquestación y monitoreo de tasks |
| **api** | Python 3.11 | 8000 | Servicio de inferencia REST |

---

## 📁 Estructura del Proyecto

```
income-classifier/
│
├── src/
│   ├── pipeline.py          # Flow de Prefect con 5 tasks (entrenamiento completo)
│   └── api.py               # API de inferencia con FastAPI
│
├── data/
│   └── adult.data.csv       # Dataset original ← colocar aquí antes de ejecutar
│
├── models/                  # Generado automáticamente por el pipeline
│   ├── best_model.pkl       # Modelo ganador serializado
│   ├── scaler.pkl           # MinMaxScaler entrenado
│   └── encoders.pkl         # LabelEncoders por columna categórica
│
├── Dockerfile.pipeline      # Imagen del pipeline de ML
├── Dockerfile.api           # Imagen de la API de inferencia
├── docker-compose.yml       # Stack completo
├── requirements.pipeline.txt
├── requirements.api.txt
└── README.md
```

---

## 🔄 Pipeline — Las 5 Tasks

El pipeline está orquestado con **Prefect** y dividido en 5 tasks claramente separadas:

```
cargar_datos → limpiar_datos → entrenar_modelos → seleccionar_mejor_modelo → guardar_modelo
```

| # | Task | Descripción |
|---|---|---|
| 1 | `cargar_datos` | Lee el CSV con **reintentos automáticos** en caso de fallo |
| 2 | `limpiar_datos` | Elimina `?`, nulos y duplicados · Codifica con LabelEncoder · Escala con MinMaxScaler |
| 3 | `entrenar_modelos` | Entrena RandomForest, GradientBoosting y XGBoost con SMOTE · Registra en MLflow |
| 4 | `seleccionar_mejor_modelo` | Compara los 3 modelos y elige el mayor **Recall clase 1** |
| 5 | `guardar_modelo` | Guarda en disco y registra en **MLflow Model Registry** |

### ¿Por qué Recall de clase 1 como métrica principal?
Con un dataset desbalanceado 75/25, el Recall de clase 1 nos obliga a detectar correctamente a las personas con altos ingresos, minimizando los falsos negativos — que son el error más costoso en este contexto de negocio.

---

## 📈 Modelos y Métricas

Se entrenaron y compararon **3 modelos** con SMOTE aplicado en el conjunto de entrenamiento:

| Modelo | Accuracy | Recall cl.1 | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|---|
| 🏆 **Random Forest** | 85.2% | **82.7%** | 0.921 | 0.847 | 0.634 |
| Gradient Boosting | 86.1% | 77.3% | 0.931 | 0.851 | 0.648 |
| XGBoost | 86.4% | 76.1% | 0.929 | 0.848 | 0.650 |

### Métricas adicionales (valor agregado)

| Métrica | Descripción |
|---|---|
| **ROC-AUC** | Área bajo la curva ROC — qué tan bien separa las dos clases |
| **PR-AUC** | Área bajo la curva Precision-Recall — más informativa con datos desbalanceados |
| **MCC** | Matthews Correlation Coefficient — considera los 4 valores de la matriz de confusión simultáneamente. Va de -1 a +1 |

### ¿Por qué Random Forest?
Aunque GBM y XGBoost tienen mejor ROC-AUC y MCC, **Random Forest lidera en Recall clase 1 con 82.7%**, que es nuestra métrica de negocio prioritaria. Adicionalmente es interpretable a través de feature importance, robusto al desbalance con `class_weight='balanced'` y consistente entre ejecuciones.

---

## 🛠️ Decisiones Técnicas

| Decisión | Justificación |
|---|---|
| **SMOTE para balanceo** | Dataset 75/25 — genera ejemplos sintéticos de clase minoritaria para no sesgar el modelo |
| **LabelEncoder** | Variables categóricas con muchas categorías — más eficiente que OneHotEncoder para árboles |
| **MinMaxScaler** | Normaliza variables numéricas al rango [0,1] — consistente con el EDA |
| **Recall clase 1 como métrica principal** | Minimizar falsos negativos es prioritario en el contexto de negocio |
| **Se excluyó `fnlwgt`** | Variable administrativa del censo con bajo poder predictivo confirmado por feature importance |
| **3 modelos candidatos** | RandomForest (robusto), GradientBoosting (preciso), XGBoost (eficiente) |
| **Volumen compartido Docker** | Pipeline y API comparten el mismo volumen para pasar el modelo sin copias |

---

## 🚀 Cómo Ejecutar el Proyecto

### Prerrequisitos
- Docker Desktop instalado y corriendo
- Git

### Paso 1 — Clonar el repositorio
```bash
git clone https://github.com/isabelcristinaperezusuga-tech/Entrega_Final_MLOPS.git
cd Entrega_Final_MLOPS/income-classifier
```

### Paso 2 — Agregar el dataset
```bash
# Copiar el archivo adult.data.csv a la carpeta data/
cp /ruta/a/adult.data.csv data/
```

### Paso 3 — Levantar todo el stack
```bash
docker-compose up --build
```

⏳ La primera vez tarda entre 5 y 10 minutos mientras Docker descarga las imágenes y construye los contenedores.

### Paso 4 — Verificar que todo está corriendo
Espera hasta ver en la terminal:
```
pipeline  | ✅ Pipeline completado. Modelo en: /app/models/best_model.pkl
pipeline  |    Mejor modelo : RandomForest
pipeline  |    Recall cl. 1 : 0.8272
```

### Paso 5 — Explorar los servicios
Abre en el navegador:
- `http://localhost:8000/docs` → API de predicciones
- `http://localhost:5000` → MLflow con los 3 modelos comparados
- `http://localhost:4200` → Prefect con el flujo de tasks

### Detener el stack
```bash
docker-compose down
```

### Volver a levantar (sin reconstruir)
```bash
docker-compose up
```

---

## 🌐 URLs de los Servicios

| Servicio | URL | Descripción |
|---|---|---|
| **API (Swagger)** | http://localhost:8000/docs | Interfaz interactiva para hacer predicciones |
| **API (ReDoc)** | http://localhost:8000/redoc | Documentación alternativa |
| **MLflow** | http://localhost:5000 | Comparación de modelos y métricas |
| **Prefect** | http://localhost:4200 | Flujo de tareas y ejecuciones |

---

## 🧪 Ejemplo de Predicción

### Request
```bash
curl -X POST "http://localhost:8000/predecir" \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 45,
    "clase_trabajo": "Private",
    "educacion": "Masters",
    "educacion_num": 14,
    "estado_civil": "Married-civ-spouse",
    "ocupacion": "Exec-managerial",
    "relacion": "Husband",
    "raza": "White",
    "genero": "Male",
    "ganancia_capital": 5000,
    "perdida_capital": 0,
    "horas_por_semana": 50,
    "pais_nativo": "United-States"
  }'
```

### Response
```json
{
  "prediccion": 1,
  "etiqueta": ">50K anuales",
  "probabilidad_mayor_50k": 0.8478,
  "mensaje": "El modelo estima que esta persona SÍ supera los 50.000 USD anuales (confianza: 84.8%)."
}
```

---

## 👥 Integrantes

| Nombre | GitHub | Rol |
|---|---|---|
| Isabel Cristina Pérez | [@isabelcristinaperezusuga-tech](https://github.com/isabelcristinaperezusuga-tech) | EDA · API · Pipeline · Docker |
| Anderson Taborda Hernandez | — | Pipeline · MLflow · Prefect · Docker |

---

## 📚 Referencias

- [UCI Adult Income Dataset](https://archive.ics.uci.edu/dataset/2/adult)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Prefect Documentation](https://docs.prefect.io)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [SMOTE — Imbalanced Learn](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
