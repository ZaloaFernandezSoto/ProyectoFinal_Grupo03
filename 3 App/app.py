import os
import time
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import base64

# =====================================================
# CONFIGURACIÓN GLOBAL
# =====================================================
st.set_page_config(
    page_title="TEP - Detección de Fallos",
    page_icon="../data/images/eastman.png",
    layout="wide",
)

st.markdown("""
    <style>

        /* ---- Tipografía y layout ---- */
        h1, h2, h3 {
            font-family: 'Segoe UI', sans-serif;
            font-weight: 700;
        }

        p, li {
            font-family: 'Segoe UI', sans-serif;
            font-size: 16px;
        }

        /* ---- Tarjetas KPI ---- */
        .metric-container {
            background: #f7f9fc;
            padding: 18px 25px;
            border-radius: 12px;
            border: 1px solid #e3e6ef;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        /* ---- Separadores suaves ---- */
        hr {
            border: none;
            border-top: 1px solid #ddd;
            margin: 25px 0;
        }

        /* ---- Tablas ---- */
        .dataframe {
            border-radius: 10px !important;
            overflow: hidden !important;
        }

    </style>
    """, unsafe_allow_html=True)

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
img_base64 = get_base64_image("../data/images/eastman.png")

def build_fault_color_map(fault_types):
    """
    Normal -> verde
    Fallos -> escala de rojos (más fallos = rojo más claro)
    """
    color_map = {
        "Normal": "#2E7D32"  
    }

    faults = [f for f in fault_types if f != "Normal"]

    red_scale = [
        "#C62828", 
        "#D32F2F",
        "#E53935",
        "#EF5350",
        "#E57373",
        "#EF9A9A"
    ]

    for i, fault in enumerate(faults):
        color_map[fault] = red_scale[min(i, len(red_scale) - 1)]

    return color_map

st.sidebar.title("Navegación")
page = st.sidebar.radio(
    "Ir a:",
    ("Análisis de Datos", "Entrenamiento", "Monitorización en Tiempo Real"),
)

# =====================================================
# RUTAS IMPORTANTES
# =====================================================
EDA_DATA_PATH = "../data/TEP_csv/datos_eda.csv"
TEST_SCALED_PATH = "../data/processed/X_test_scaled.csv"
SCALER_PATH = "../1 PreparacionDatos/tep_scaler.pkl"

BENTOML_BASE_URL = "http://localhost:3000"
ENDPOINT_RF = f"{BENTOML_BASE_URL}/predecir_fallo_rapido"
ENDPOINT_LSTM = f"{BENTOML_BASE_URL}/predecir_fallo_inteligente"
ENDPOINT_HEALTH = f"{BENTOML_BASE_URL}/verificar_servidor"

# =====================================================
# LISTA DE VARIABLES (52 FEATURES)
# =====================================================
FEATURE_COLS: List[str] = [
    "A_Feed_stream1",
    "D_Feed_stream2",
    "E_Feed_stream3",
    "AC_Feed_stream4",
    "Recycle_Flow_stream8",
    "Reactor_Feed_Rate_stream6",
    "Reactor_Pressure",
    "Reactor_Level",
    "Reactor_Temperature",
    "Purge_Rate_stream9",
    "Product_Sep_Temp",
    "Product_Sep_Level",
    "Prod_Sep_Pressure",
    "Prod_Sep_Underflow_stream10",
    "Stripper_Level",
    "Stripper_Pressure",
    "Stripper_Underflow_stream11",
    "Stripper_Temperature",
    "Stripper_Steam_Flow",
    "Compressor_Work",
    "Reactor_CW_Outlet_Temp",
    "Separator_CW_Outlet_Temp",
    "Reactor_Feed_CompA",
    "Reactor_Feed_CompB",
    "Reactor_Feed_CompC",
    "Reactor_Feed_CompD",
    "Reactor_Feed_CompE",
    "Reactor_Feed_CompF",
    "Purge_Gas_CompA",
    "Purge_Gas_CompB",
    "Purge_Gas_CompC",
    "Purge_Gas_CompD",
    "Purge_Gas_CompE",
    "Purge_Gas_CompF",
    "Purge_Gas_CompG",
    "Purge_Gas_CompH",
    "Product_CompD",
    "Product_CompE",
    "Product_CompF",
    "Product_CompG",
    "Product_CompH",
    "D_Feed_Flow",
    "E_Feed_Flow",
    "A_Feed_Flow",
    "AC_Feed_Flow",
    "Compressor_Recycle_Valve",
    "Purge_Valve",
    "Separator_Pot_Liquid_Flow",
    "Stripper_Product_Flow",
    "Stripper_Steam_Valve",
    "Reactor_CW_Flow",
    "Condenser_CW_Flow",
]

# =====================================================
# CARGA DE DATOS Y SCALER 
# =====================================================

@st.cache_data
def load_eda_data() -> pd.DataFrame | None:
    if not os.path.exists(EDA_DATA_PATH):
        st.error(f"No se ha encontrado el fichero de EDA: {EDA_DATA_PATH}")
        return None
    return pd.read_csv(EDA_DATA_PATH)


@st.cache_data
def load_test_scaled() -> pd.DataFrame | None:
    if not os.path.exists(TEST_SCALED_PATH):
        st.error(f"No se ha encontrado X_test_scaled: {TEST_SCALED_PATH}")
        return None
    return pd.read_csv(TEST_SCALED_PATH)


@st.cache_resource
def load_scaler():
    if not os.path.exists(SCALER_PATH):
        st.warning(
            f"No se ha encontrado el scaler en {SCALER_PATH}. "
            "Se usarán los valores de X_test_scaled tal cual."
        )
        return None
    try:
        return joblib.load(SCALER_PATH)
    except Exception as e:
        st.warning(f"No se ha podido cargar el scaler: {e}")
        return None

# =====================================================
# FUNCIONES AUXILIARES DE ESCALADO
# =====================================================

def inverse_scale_row(row: pd.Series, scaler) -> np.ndarray:
    """Des-normaliza una fila de 52 variables si hay scaler."""
    x = row[FEATURE_COLS].values.reshape(1, -1)
    if scaler is not None:
        try:
            x_raw = scaler.inverse_transform(x)
            return x_raw.flatten()
        except Exception:
            return x.flatten()
    return x.flatten()


def inverse_scale_sequence(seq_df: pd.DataFrame, scaler) -> np.ndarray:
    """Des-normaliza una secuencia (10 x 52)."""
    x = seq_df[FEATURE_COLS].values
    if scaler is not None:
        try:
            x_raw = scaler.inverse_transform(x)
            return x_raw
        except Exception:
            return x
    return x

# =====================================================
# FUNCIONES AUXILIARES DE LA API BENTOML
# =====================================================

def _parse_error_response(resp: requests.Response) -> Dict[str, Any]:
    try:
        return {"error": str(resp.status_code), "detalle": resp.text}
    except Exception:
        return {"error": str(resp.status_code)}


def call_api_rapido(sensores_52: np.ndarray) -> Dict[str, Any]:
    """
    Llama al endpoint predecir_fallo_rapido de BentoML.

    Primero prueba con el formato actual del service.py:
        {"sensores": [...]}

    Si el servidor devuelve 400 y menciona que falta otra clave,
    intenta ser compatible con versiones antiguas (por ejemplo "ddatos").
    """
    valores = [float(x) for x in sensores_52]

    payload = {
        "datos": {
            "sensores": valores
        }
    }

    try:
        resp = requests.post(ENDPOINT_RF, json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 400 and "ddatos" in resp.text:
            payload_alt = {"ddatos": valores}
            resp2 = requests.post(ENDPOINT_RF, json=payload_alt, timeout=10)
            if resp2.status_code == 200:
                return resp2.json()
            return _parse_error_response(resp2)

        return _parse_error_response(resp)

    except Exception as e:
        return {"error": str(e)}


def call_api_inteligente(sensores_52: np.ndarray,
                         secuencia_10x52: np.ndarray) -> Dict[str, Any]:
    """Llama al endpoint predecir_fallo_inteligente (LSTM)."""
    valores = [float(x) for x in sensores_52]
    secuencia_list = secuencia_10x52.astype(float).tolist()

    payload = {
        "datos": {
            "sensores": valores,
            "secuencia_temporal": secuencia_list
        }
    }

    try:
        resp = requests.post(ENDPOINT_LSTM, json=payload, timeout=15)
        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 400 and "ddatos" in resp.text:
            payload_alt = {
                "ddatos": valores,
                "secuencia_temporal": secuencia_list,
            }
            resp2 = requests.post(ENDPOINT_LSTM, json=payload_alt, timeout=15)
            if resp2.status_code == 200:
                return resp2.json()
            return _parse_error_response(resp2)

        return _parse_error_response(resp)

    except Exception as e:
        return {"error": str(e)}


def call_healthcheck() -> Dict[str, Any] | None:
    """Llama al endpoint verificar_servidor (POST)."""
    try:
        resp = requests.post(ENDPOINT_HEALTH, timeout=5)
        if resp.status_code != 200:
            return _parse_error_response(resp)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

# =====================================================
# PRESENTACIÓN DEL RESULTADO
# =====================================================

def interpretar_respuesta(respuesta_api: Dict[str, Any]) -> str:
    """Construye un mensaje bonito a partir de RespuestaAPI o error."""
    if not respuesta_api:
        return "No se ha recibido respuesta del servidor."

    if "error" in respuesta_api:
        detalle = respuesta_api.get("detalle", "")
        return f"❌ **Error al llamar a la API**\n\n- Código: `{respuesta_api['error']}`\n- Detalle: `{detalle}`"

    nombre = respuesta_api.get("nombre_fallo", "Desconocido")
    id_fallo = respuesta_api.get("id_fallo", "?")
    confianza = float(respuesta_api.get("confianza", 0.0))
    modelo = respuesta_api.get("modelo_usado", "N/A")

    if str(nombre).lower().startswith("operación normal"):
        icono = "🟢"
        estado = "Sistema estable"
    else:
        icono = "🔴"
        estado = "¡ALERTA: fallo detectado!"

    return (
        f"{icono} **{estado}**\n\n"
        f"- Id fallo: `{id_fallo}`\n"
        f"- Nombre: **{nombre}**\n"
        f"- Confianza: **{confianza:.1%}**\n"
        f"- Modelo usado: `{modelo}`"
    )

# =====================================================
# ANÁLISIS DE DATOS (EDA)
# =====================================================

def page_eda():
    st.title("Análisis Exploratorio de Datos (EDA)")

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:20px; padding:15px 0;">
            <img src="data:image/png;base64,{img_base64}" style="height:60px;">
            <h2 style="margin:0;">Tennessee Eastman Process</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
        <div style="
            background:#eef3ff;
            padding:18px 25px;
            border-left: 5px solid #3E66F2;
            border-radius:8px;
            margin-bottom:20px;">

        <b>Bienvenido al módulo de Análisis Exploratorio de Datos (EDA) del Tennessee Eastman Process (TEP).</b><br>
        En esta sección se analiza el comportamiento de los <b>52 sensores del proceso industrial</b>, con el objetivo
        de comprender la dinámica del sistema y evaluar cómo los distintos tipos de fallo afectan a las variables del proceso.

        <ul>
        <li><b>Distribución estadística</b> de cada variable para identificar rangos normales y valores atípicos</li>
        <li><b>Comparación visual</b> entre operación normal y distintos modos de fallo</li>
        <li><b>Matriz de correlación interactiva</b> para detectar relaciones y sensores redundantes</li>
        </ul>

        Este análisis sirve como base para la selección de variables y el desarrollo de los modelos de detección de fallos.
        </div>
        """, unsafe_allow_html=True)

    df = load_eda_data()
    if df is None or df.empty:
        st.error("No se pudo cargar el dataset de EDA.")
        return

    # =====================================================
    # 1. KPI CARDS
    # =====================================================
    st.markdown("### Resumen general del dataset")
    n_faults = df["fault"].nunique() if "fault" in df.columns else 0
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Observaciones totales</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Variables del proceso</h3>
            <h2>{len(FEATURE_COLS)}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Modos de operación</h3>
            <h2>{n_faults}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # =====================================================
    # 2. VISTA GENERAL
    # =====================================================
    st.markdown("### Vista general del dataset")

    st.dataframe(df.head(), use_container_width=True)

    st.download_button(
        "Descargar dataset completo",
        df.to_csv(index=False).encode("utf-8"),
        "EDA_dataset.csv",
        "text/csv"
    )

    st.markdown("---")

    # =====================================================
    # 3. ESTADÍSTICOS DESCRIPTIVOS
    # =====================================================
    st.markdown("### Estadísticos descriptivos de los sensores")

    desc = df[FEATURE_COLS].describe().T
    st.dataframe(desc, use_container_width=True)

    st.info("**Interpretación**: Busca sensores con alta desviación estándar → suelen ser clave para diagnóstico.")

    st.markdown("---")
    
    # =====================================================
    # 4. FILTRO POR TIPO DE FALLO
    # =====================================================

    if "fault" in df.columns:
        st.markdown("### Filtro por tipo de fallo")
        # Mapeo bonito para el usuario
        mapa_fallos = {
            f: "Operación Normal" if f == 0 else f"Fallo Tipo {f}"
            for f in sorted(df["fault"].unique())
        }

        fallos_seleccionados_labels = st.multiselect(
            "Selecciona modos de operación:",
            options=list(mapa_fallos.values()),
            default=["Operación Normal"] if "Operación Normal" in mapa_fallos.values() else None,
        )

        fallos_seleccionados = [
            k for k, v in mapa_fallos.items()
            if v in fallos_seleccionados_labels
        ]

        df_filtro = df[df["fault"].isin(fallos_seleccionados)] if fallos_seleccionados else df.copy()

        if fallos_seleccionados:
            df_filtro = df[df["fault"].isin(fallos_seleccionados)]
        else:
            df_filtro = df.copy()
    else:
        df_filtro = df.copy()
        
    st.info(
            "Este filtro permite seleccionar los modos de operación a analizar. "
            "Todos los gráficos siguientes se actualizan dinámicamente en función "
            "de la operación normal y/o los tipos de fallo seleccionados."
        )
    
    # =====================================================
    # 5. HISTOGRAMA INTERACTIVO
    # =====================================================
    st.markdown("### Histograma interactivo")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        sensor_hist = st.selectbox("Selecciona una variable:", FEATURE_COLS, index=0)

    fault_types = df_filtro["fault_type"].unique()
    color_map = build_fault_color_map(fault_types)

    fig_hist = px.histogram(
        df_filtro,
        x=sensor_hist,
        color="fault_type",
        nbins=40,
        opacity=0.75,
        barmode="overlay",
        color_discrete_map=color_map,
    )
    fig_hist.update_layout(yaxis_title="Frecuencia"
    st.plotly_chart(fig_hist, use_container_width=True)

    st.info(
        "**Interpretación:** El histograma muestra la distribución de valores del sensor seleccionado "
        "para los modos de operación elegidos. Permite analizar cambios en la forma, "
        "dispersión o aparición de valores extremos al comparar operación normal y fallos."
    )

    st.markdown("---")

    # =====================================================
    # 6. BOXPLOT POR TIPO DE FALLO
    # =====================================================
    if "fault_type" in df.columns:
        st.markdown("### Boxplot por Tipo de Fallo")

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            sensor_box = st.selectbox("Variable para boxplot:", FEATURE_COLS)

        df_plot = df_filtro.copy()

        fig_box = px.box(df_plot, x="fault_type", y=sensor_box,
                         title=f"{sensor_box} por tipo de fallo",
                         color="fault_type")
        st.plotly_chart(fig_box, use_container_width=True)

        st.info(
            "**Interpretación:** Este boxplot permite comparar directamente el comportamiento del sensor entre "
            "operación normal y distintos tipos de fallo. Un desplazamiento de la mediana "
            "o un aumento de la dispersión indica que el sensor es sensible al fallo."
        )
    else:
        st.warning("El dataset no tiene columna 'fault_type', por lo que no se puede generar el boxplot por fallo.")

    st.markdown("---")

    # =====================================================
    # 7. MATRIZ DE CORRELACIÓN
    # =====================================================
    st.markdown("### Matriz de correlación")

    num_vars = st.slider(
        "Número de sensores a mostrar:",
        min_value=5,
        max_value=25,
        value=12,
    )

    vars_corr = FEATURE_COLS[:num_vars]
    corr = df_filtro[vars_corr].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Matriz de correlación"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.info(
        "**Interpretación:** La matriz de correlación muestra las relaciones entre los sensores del proceso "
        "para los modos de operación seleccionados. Correlaciones altas indican sensores "
        "redundantes, mientras que correlaciones negativas fuertes reflejan relaciones "
        "inversas propias de la dinámica del proceso."
    )

# =====================================================
# ENTRENAMIENTO (RESUMEN)
# =====================================================

def page_training():
    st.title("Resultados de Entrenamiento")

    st.markdown("""
        <div style="
            background:#eef3ff;
            padding:18px 25px;
            border-left: 5px solid #3E66F2;
            border-radius:8px;
            margin-bottom:20px;">

        En esta sección se evalúa el rendimiento de los modelos entrenados durante la <b>Fase 1</b> 
        (Random Forest y LSTM) utilizando el conjunto de test.  
        Las métricas mostradas permiten comparar un modelo rápido basado en variables instantáneas
        con un modelo secuencial que captura la dinámica temporal del proceso.

        </div>
        """, unsafe_allow_html=True)

    # Rutas
    RF_MODEL_PATH = "../1 PreparacionDatos/tep_rf_model_optimized.pkl"
    LSTM_MODEL_PATH = "../1 PreparacionDatos/tep_lstm_model.keras"
    X_TEST_PATH = "../data/processed/X_test_scaled.csv"
    Y_TEST_PATH = "../data/processed/y_test.csv"

    # Cargar datos
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

    # ---------- RANDOM FOREST ----------
    rf_model = joblib.load(RF_MODEL_PATH)
    y_pred_rf = rf_model.predict(X_test)

    rf_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf, average="weighted"),
        "recall": recall_score(y_test, y_pred_rf, average="weighted"),
        "f1": f1_score(y_test, y_pred_rf, average="weighted"),
    }

    # ---------- LSTM ----------
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)

    X_seq = []
    y_seq = []
    for i in range(9, len(X_test)):
        X_seq.append(X_test.iloc[i-9:i+1].values)
        y_seq.append(y_test[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    y_pred_lstm = np.argmax(lstm_model.predict(X_seq), axis=1)

    lstm_metrics = {
        "accuracy": accuracy_score(y_seq, y_pred_lstm),
        "precision": precision_score(y_seq, y_pred_lstm, average="weighted"),
        "recall": recall_score(y_seq, y_pred_lstm, average="weighted"),
        "f1": f1_score(y_seq, y_pred_lstm, average="weighted"),
    }

    # ---------- VISUALIZACIÓN ----------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Random Forest")
        st.caption("Modelo basado en variables instantáneas, optimizado para detección rápida de fallos.")
        st.metric("Accuracy", f"{rf_metrics['accuracy']:.3f}")
        st.metric("Precision", f"{rf_metrics['precision']:.3f}")
        st.metric("Recall", f"{rf_metrics['recall']:.3f}")
        st.metric("F1-Score", f"{rf_metrics['f1']:.3f}")

    with col2:
        st.markdown("### LSTM")
        st.caption("Modelo secuencial que utiliza ventanas temporales para capturar la evolución del proceso.")
        st.metric("Accuracy", f"{lstm_metrics['accuracy']:.3f}")
        st.metric("Precision", f"{lstm_metrics['precision']:.3f}")
        st.metric("Recall", f"{lstm_metrics['recall']:.3f}")
        st.metric("F1-Score", f"{lstm_metrics['f1']:.3f}")


# =====================================================
# MONITORIZACIÓN EN TIEMPO REAL
# =====================================================

def init_session_state():
    if "current_index" not in st.session_state:
        st.session_state["current_index"] = 0
    if "prediction_log" not in st.session_state:
        st.session_state["prediction_log"] = []  
    if "auto_analyze" not in st.session_state:
        st.session_state["auto_analyze"] = False


def page_monitoring():
    st.title("Monitorización en Tiempo Real")

    df_test = load_test_scaled()
    # scaler = load_scaler()
    if df_test is None or df_test.empty:
        return

    init_session_state()

    st.markdown("""
        <div style="
            background:#eef3ff;
            padding:18px 25px;
            border-left: 5px solid #3E66F2;
            border-radius:8px;
            margin-bottom:20px;">
            
        En esta sección se simula un sistema de <b>detección de fallos en tiempo real</b> para el 
        Tennessee Eastman Process (TEP).  
        Cada observación representa una nueva lectura del proceso y se envía a un servicio 
        desplegado con BentoML para evaluar el estado del sistema.

        El usuario puede seleccionar el modelo de predicción y analizar si el proceso se encuentra 
        en operación normal o si se detecta un fallo específico.
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Estado del servicio de predicción")
    st.caption(
        "Se comprueba la disponibilidad del servicio BentoML encargado de realizar las predicciones."
    )
    if st.button("Comprobar estado del servidor"):
        try:
            estado = call_healthcheck()
            if isinstance(estado, dict) and "status" in estado:
                if estado.get("status", "").upper() in ["OK", "OPERATIVO"]:
                    st.success("✅ Servidor activo y funcionando correctamente.")
                else:
                    st.warning("⚠️ Servidor activo, pero la respuesta no es la esperada.")
                    st.json(estado)
            else:
                st.warning("⚠️ Servidor activo, pero formato de respuesta inesperado.")
                st.json(estado)
        except Exception as e:
            st.error("❌ No se pudo conectar con el servicio BentoML.")
            st.code(str(e))

    st.markdown("---")
    st.subheader("Selección del modelo de predicción")
    modo = st.radio(
        "Elige el modelo que realizará la predicción:",
        (
            "Predicción rápida (Random Forest)",
            "Predicción inteligente (LSTM)"
        )
    )

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    # -------------------- LECTURA ACTUAL --------------------
    with col_left:
        st.subheader("Lectura actual")

        idx = st.session_state["current_index"]
        if idx >= len(df_test):
            idx = 0
            st.session_state["current_index"] = 0

        fila_actual = df_test.iloc[idx]
        st.dataframe(fila_actual.to_frame().T)

        if st.button("🔁 Simular nueva lectura (3 minutos después)"):
            STEP = 50 
            st.session_state["current_index"] = (idx + STEP) % len(df_test)
            st.session_state["auto_analyze"] = True
            st.rerun()

    # -------------------- PREDICCIÓN --------------------
    with col_right:
        st.subheader("Predicción del estado")

        analyze_clicked = st.button(" Analizar estado actual")
        if analyze_clicked or st.session_state.get("auto_analyze", False):
            st.session_state["auto_analyze"] = False

            idx = st.session_state["current_index"]
            fila_actual = df_test.iloc[idx]

            sensores_scaled = fila_actual[FEATURE_COLS].values.astype(float)

            if modo.startswith("Predicción rápida"):
                with st.spinner("Llamando a predecir_fallo_rapido..."):
                    respuesta = call_api_rapido(sensores_scaled)

            else:
                inicio = max(0, idx - 9)
                fin = idx + 1
                seq_df = df_test.iloc[inicio:fin]

                if len(seq_df) < 10:
                    primera = seq_df.iloc[0]
                    faltan = 10 - len(seq_df)
                    seq_df = pd.concat([pd.DataFrame([primera] * faltan), seq_df], ignore_index=True)

                secuencia_scaled = seq_df[FEATURE_COLS].values.astype(float)

                with st.spinner("Llamando a predecir_fallo_inteligente (LSTM)..."):
                    respuesta = call_api_inteligente(sensores_scaled, secuencia_scaled)

            st.markdown("### Resultado")
            st.markdown(interpretar_respuesta(respuesta))

            if respuesta and "error" not in respuesta:
                st.session_state["prediction_log"].append(
                    {
                        "t": time.strftime("%H:%M:%S"),
                        "idx": idx,
                        "id_fallo": respuesta.get("id_fallo"),
                        "nombre": respuesta.get("nombre_fallo"),
                        "confianza": float(respuesta.get("confianza", 0.0)),
                        "modelo": respuesta.get("modelo_usado", ""),
                    }
                )

            st.markdown("### Respuesta completa de la API (debug)")
            st.json(respuesta)

    # -------------------- HISTORIAL DE PREDICCIONES --------------------
    st.markdown("---")
    st.subheader("Historial de predicciones")

    if st.session_state["prediction_log"]:
        log_df = pd.DataFrame(st.session_state["prediction_log"])
        log_df["confianza"] = log_df["confianza"].round(3)
        st.dataframe(log_df.tail(20), use_container_width=True)
    else:
        st.write("Todavía no se ha realizado ninguna predicción en esta sesión.")

# =====================================================
# MAIN
# =====================================================

def main():
    if page == "Análisis de Datos":
        page_eda()
    elif page == "Entrenamiento":
        page_training()
    elif page == "Monitorización en Tiempo Real":
        page_monitoring()


if __name__ == "__main__":
    main()