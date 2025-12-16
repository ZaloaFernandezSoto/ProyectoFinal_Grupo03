# 🏭 Sistema de Detección de Fallos en el Proceso Tennessee Eastman (TEP)

**Asignatura**: Analítica de Datos para la Industria  
**Curso**: 2025/2026  
**Grupo**: 03

---

## Descripción del Proyecto

Este proyecto desarrolla un sistema integral de detección y clasificación de fallos industriales utilizando el dataset Tennessee Eastman Process (TEP). El sistema combina análisis exploratorio de datos, modelos de machine learning (Random Forest y LSTM) y una aplicación web interactiva con capacidades de predicción en tiempo real.

**Objetivo**: Crear una aplicación de analítica industrial que integre visualización, modelado y despliegue de inteligencia artificial para detectar y clasificar 22 modos de operación diferentes (1 normal + 21 tipos de fallos) en un proceso químico industrial.

---

## Estructura del Proyecto

```
ProyectoFinal_Grupo03/
│
├── 1 PreparacionDatos/              # Fase 1: Análisis y Entrenamiento
│   ├── eda.ipynb                    # Notebook principal con EDA completo
│   ├── generate_mappings.py         # Script de generación de datos
│   ├── tep_rf_model_optimized.pkl   # Modelo Random Forest entrenado
│   ├── tep_lstm_model.keras         # Modelo LSTM entrenado
│   └── tep_scaler.pkl              # Normalizador StandardScaler
│
├── 2 API/                           # Fase 2: Despliegue con BentoML
│   ├── service.py                   # Servicio API con endpoints
│   └── bentofile.yaml              # Configuración BentoML
│
├── 3 App/                           # Fase 3: Aplicación Streamlit
│   ├── app.py                       # Aplicación web interactiva
│   └── generar_datos_eda.py        # Utilidades de generación
│
├── data/                            # Datos del proyecto
│   ├── TEP_csv/                    # Dataset en formato CSV
│   │   ├── d00.csv - d21.csv      # Datos de entrenamiento
│   │   ├── d00_te.csv - d21_te.csv # Datos de test
│   │   └── variable_info.csv       # Información de variables
│   ├── processed/                  # Datos preprocesados
│   │   ├── X_train_scaled.csv
│   │   ├── X_test_scaled.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── images/                     # Recursos visuales
│
├── models/                          # Modelos personalizados (generados en runtime)
│
├── README.md                        
└── requirements.txt                 # Dependencias del proyecto
```

---

## Dataset: Tennessee Eastman Process (TEP)

El Tennessee Eastman Process es un benchmark ampliamente utilizado en la industria y la academia para evaluar sistemas de detección de fallos. 

### Características del Dataset:
- **52 variables de proceso**: 12 variables manipuladas, 22 variables de medición continua, 19 variables de composición
- **22 modos de operación**: 1 normal (d00) + 21 tipos de fallos (d01-d21)
- **Conjunto de entrenamiento**: 25h de simulación
  - Operación normal: 500 observaciones
  - Operación con fallo: 480 observaciones 

- **Conjunto de prueba**: 48h de simulación
  - Total: 960 observaciones por modo
  - Fallo introducido en t=8h (primeras 160 observaciones normales)

> **Para una explicación detallada del dataset**, estructura de variables, análisis exploratorio completo y decisiones de preprocesamiento, consultar el notebook **[`1 PreparacionDatos/eda.ipynb`](1%20PreparacionDatos/eda.ipynb)**

---

## Instalación y Configuración

### Prerrequisitos
- Python 3.10 o superior
- pip (gestor de paquetes de Python)

### 1. Clonar o descargar el proyecto

```bash
cd ProyectoFinal_Grupo03
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Verificar estructura de datos

Asegúrate de que la carpeta `data/TEP_csv/` contenga todos los archivos CSV del dataset (d00.csv hasta d21_te.csv).

---

## Ejecución del Proyecto

El proyecto se ejecuta en dos terminales simultáneas:

### Terminal 1: Servidor BentoML (API de Predicción)

```bash
cd "2 API"
bentoml serve service:DetectorFallos --port 3000 --reload
```

El servidor estará disponible en: `http://localhost:3000`

**Endpoints disponibles:**
- `POST /predecir_fallo_rapido` - Predicción con Random Forest
- `POST /predecir_fallo_inteligente` - Predicción con LSTM
- `GET /verificar_servidor` - Health check

### Terminal 2: Aplicación Streamlit (Interfaz Web)

```bash
cd "3 App"
streamlit run app.py
```

La aplicación se abrirá automáticamente en: `http://localhost:8501`

---

## Funcionalidades de la Aplicación

### 1. Análisis Exploratorio de Datos (EDA)
- Visualización de estadísticas descriptivas
- Histogramas y boxplots interactivos por variable
- Comparación visual: Operación Normal vs. Fallos
- Matriz de correlación interactiva

### 2. Entrenamiento y Validación
- **Evaluación de modelos**: Métricas de rendimiento (Accuracy, F1-Score, Precision, Recall)
- **Interpretabilidad**: Feature importance y variables críticas
- **Playground interactivo**: 
  - Entrenar modelos personalizados (Random Forest o Decision Tree)
  - Configurar hiperparámetros en tiempo real
  - Guardar modelos en carpeta `models/`

### 3. Monitorización en Tiempo Real
- Simulador de lecturas de sensores
- **Dos modos de predicción**:
  - API BentoML (modelos pre-entrenados)
  - Modelos locales (entrenados en el playground)
- Visualización de resultados con:
  - Tarjetas de diagnóstico (Normal/Fallo)
  - Nivel de confianza del modelo
  - Análisis de probabilidades por tipo de fallo

---

## Modelos Implementados

### Random Forest Optimizado
- **Algoritmo**: Ensemble de árboles de decisión
- **Optimización**: Grid Search con validación cruzada (162 configuraciones)
- **Accuracy**: ~58-59%
- **Tiempo de inferencia**: ~10ms
- **Uso**: Detección rápida en tiempo real

### LSTM (Long Short-Term Memory)
- **Arquitectura**: 2 capas LSTM (128→64) + Dense
- **Input**: Secuencias de 10 pasos temporales
- **Accuracy**: ~58%
- **Tiempo de inferencia**: ~200ms
- **Uso**: Análisis profundo con contexto temporal

**Estrategia híbrida**: Random Forest para screening inicial, LSTM para confirmación detallada.

---

## Tecnologías Utilizadas

### Data Science & ML
- **pandas**, **numpy**: Manipulación de datos
- **scikit-learn**: Random Forest, métricas, preprocesamiento
- **tensorflow/keras**: LSTM, deep learning
- **joblib**: Serialización de modelos

### Visualización
- **plotly**: Gráficos interactivos
- **matplotlib**, **seaborn**: Visualizaciones estáticas

### Aplicación Web
- **Streamlit**: Framework de aplicación web
- **BentoML**: Despliegue y servicio de modelos ML

---

## Documentación Técnica

### Notebook Principal
**[`1 PreparacionDatos/eda.ipynb`](1%20PreparacionDatos/eda.ipynb)** contiene:
- Descripción detallada del dataset TEP
- Análisis exploratorio completo (distribuciones, correlaciones, temporal)
- Preprocesamiento y normalización
- Feature engineering y selección de variables
- Entrenamiento de modelos con optimización
- Evaluación y comparativa de resultados
- Conclusiones técnicas de cada etapa

### Scripts de Producción
- **[`2 API/service.py`](2%20API/service.py)**: Implementación de la API BentoML
- **[`3 App/app.py`](3%20App/app.py)**: Aplicación Streamlit completa

---

## Metodología Aplicada

1. **Preparación de datos**: Normalización, limpieza, estructuración temporal
2. **Análisis exploratorio**: Estadística descriptiva, visualización, correlaciones
3. **Ingeniería de características**: Feature importance, selección de variables críticas
4. **Modelado**: Entrenamiento de Random Forest y LSTM con optimización
5. **Evaluación**: Métricas de clasificación, matriz de confusión, validación cruzada
6. **Despliegue**: API REST con BentoML para inferencia en tiempo real
7. **Interfaz**: Dashboard interactivo con Streamlit para exploración y predicción

---

## Referencias

- Tennessee Eastman Process: Downs, J. J., & Vogel, E. F. (1993). "A plant-wide industrial process control problem"
- Dataset público utilizado para investigación académica en detección de fallos industriales

