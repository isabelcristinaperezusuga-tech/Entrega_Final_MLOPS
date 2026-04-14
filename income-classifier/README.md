# 🎓 Clasificador de Ingresos de Adultos
**Proyecto Final – Parte 2: Orquestación del Pipeline de ML**
*Isabel Cristina Pérez & Anderson Taborda Hernandez*

---

## 📋 Descripción del Proyecto

Sistema de ML que predice si una persona tiene ingresos anuales **superiores o iguales a USD 50.000**, basado en características demográficas y laborales del dataset UCI Adult Income.

El proyecto integra:
- **Prefect** → orquestación del pipeline (tasks y flows)
- **MLflow** → tracking de experimentos, métricas y Model Registry
- **FastAPI** → servicio web de inferencia con documentación automática
- **Docker** → contenedorización de todos los servicios

---

## 🏗️ Estructura del Proyecto

```
income-classifier/
│
├── src/
│   ├── pipeline.py       # Flow de Prefect con 5 tasks (entrenamiento completo)
│   └── api.py            # API de inferencia con FastAPI
│
├── data/
│   └── adult.data.csv    # Dataset original ← COLÓCALO AQUÍ
│
├── models/               # Generado automáticamente por el pipeline
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── encoders.pkl
│
├── Dockerfile.pipeline   # Imagen del pipeline de ML
├── Dockerfile.api        # Imagen de la API de inferencia
├── docker-compose.yml    # Stack completo (MLflow + Prefect + Pipeline + API)
├── requirements.pipeline.txt
├── requirements.api.txt
└── README.md
```

---

## 🔄 Pipeline de ML (Prefect)

El pipeline está dividido en **5 tasks** claramente separadas:

| # | Task | Descripción |
|---|------|-------------|
| 1 | `cargar_datos` | Lee el CSV y lo carga como DataFrame |
| 2 | `limpiar_datos` | Elimina nulos, duplicados, codifica variables y escala |
| 3 | `entrenar_modelos` | Entrena RandomForest, GradientBoosting y XGBoost con SMOTE |
| 4 | `seleccionar_mejor_modelo` | Elige el modelo con mayor **recall de clase 1** |
| 5 | `guardar_modelo` | Guarda el modelo y lo registra en MLflow Model Registry |

### ¿Por qué recall de clase 1?
La clase 1 (>50K) es la clase minoritaria. Un recall alto en esta clase significa que el modelo **detecta la mayoría de las personas de altos ingresos**, lo cual es el objetivo del negocio.

---

## 🚀 Cómo Ejecutar el Proyecto

### Prerrequisitos
- Docker Desktop instalado y corriendo
- Git (opcional)

### Paso 1 – Preparar los datos
Copia tu archivo `adult.data.csv` a la carpeta `data/`:
```bash
cp /ruta/a/tu/adult.data.csv data/
```

### Paso 2 – Levantar todo con Docker Compose
```bash
docker-compose up --build
```

Esto levanta 4 servicios:
1. **MLflow** en `http://localhost:5000` → ver experimentos y métricas
2. **Prefect** en `http://localhost:4200` → ver ejecuciones del pipeline
3. **Pipeline** → entrena los modelos automáticamente
4. **API** en `http://localhost:8000` → hacer predicciones

### Paso 3 – Hacer una predicción
Una vez el pipeline termine, abre:
```
http://localhost:8000/docs
```
Ahí encontrarás la interfaz interactiva de la API. Puedes hacer clic en `/predecir → Try it out` y enviar los datos de una persona.

### Ejemplo con curl:
```bash
curl -X POST "http://localhost:8000/predecir" \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 45,
    "clase_trabajo": "Private",
    "fnlwgt": 120000,
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

Respuesta esperada:
```json
{
  "prediccion": 1,
  "etiqueta": ">50K anuales",
  "probabilidad_mayor_50k": 0.87,
  "mensaje": "El modelo estima que esta persona SÍ supera los 50.000 USD anuales (confianza: 87.0%)."
}
```

---

## 📊 Ver Resultados en MLflow

1. Abre `http://localhost:5000`
2. Clic en el experimento `income-classifier`
3. Compara los 3 modelos entrenados por sus métricas
4. El modelo registrado como mejor estará en **Models → income-classifier-best**

---

## 🧠 Decisiones Técnicas

| Decisión | Justificación |
|----------|--------------|
| **SMOTE para balanceo** | El dataset tiene ~75% clase 0 y ~25% clase 1. SMOTE genera ejemplos sintéticos de la clase minoritaria para evitar que el modelo ignore los ingresos >50K |
| **3 modelos candidatos** | RandomForest (robusto, interpretable), GradientBoosting (alta precisión), XGBoost (rápido y eficiente) |
| **Métrica de selección: Recall clase 1** | Prioriza detectar correctamente a las personas con altos ingresos, minimizando falsos negativos |
| **LabelEncoder + MinMaxScaler** | Consistente con el EDA de la Parte 1 |
| **Volumen compartido Docker** | El pipeline y la API comparten el mismo volumen para pasar el modelo sin copias |

---

## 🔧 Re-ejecutar el Pipeline

Para volver a entrenar sin perder el historial de MLflow:
```bash
docker-compose restart pipeline
```

Para detener todo:
```bash
docker-compose down
```

Para borrar también los volúmenes (datos de MLflow, Prefect y modelos):
```bash
docker-compose down -v
```

---

## 📡 Endpoints de la API

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predecir` | Clasificar ingresos de una persona |
| `GET` | `/docs` | Documentación interactiva (Swagger UI) |
| `GET` | `/redoc` | Documentación alternativa (ReDoc) |
