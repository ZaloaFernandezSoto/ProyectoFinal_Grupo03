import os
import time
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# =====================================================
# CONFIGURACIÓN GLOBAL
# =====================================================
st.set_page_config(
    page_title="TEP - Detección de Fallos",
    page_icon="🧪",
    layout="wide",
)

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
ENDPOINT_RAPIDO = f"{BENTOML_BASE_URL}/predecir_fallo_rapido"
ENDPOINT_INTELIGENTE = f"{BENTOML_BASE_URL}/predecir_fallo_inteligente"
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
# CARGA DE DATOS Y SCALER (CACHEADOS)
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
# FUNCIONES AUXILIARES (ESCALADO)
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
# FUNCIONES AUXILIARES (API BENTOML)
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

    # Intento 1: formato correcto para tu service.py actual
    payload = {
        "datos": {
            "sensores": valores
        }
    }

    try:
        resp = requests.post(ENDPOINT_RAPIDO, json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json()

        # Si hay error de validación, intentamos compatibilidad
        if resp.status_code == 400 and "ddatos" in resp.text:
            payload_alt = {"ddatos": valores}
            resp2 = requests.post(ENDPOINT_RAPIDO, json=payload_alt, timeout=10)
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
        resp = requests.post(ENDPOINT_INTELIGENTE, json=payload, timeout=15)
        if resp.status_code == 200:
            return resp.json()

        # Compatibilidad por si en algún momento cambia el nombre de campo
        if resp.status_code == 400 and "ddatos" in resp.text:
            payload_alt = {
                "ddatos": valores,
                "secuencia_temporal": secuencia_list,
            }
            resp2 = requests.post(ENDPOINT_INTELIGENTE, json=payload_alt, timeout=15)
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
# PÁGINA 1: ANÁLISIS DE DATOS (EDA)
# =====================================================

def page_eda():
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.markdown(
        """
        Esta sección permite explorar el dataset del **Tennessee Eastman Process**  
        de forma interactiva para comprender:
        - cómo se distribuyen los sensores
        - cómo cambian bajo distintos fallos
        - qué sensores están correlacionados

        ---
        """
    )

    df = load_eda_data()
    if df is None or df.empty:
        st.error("No se pudo cargar el dataset de EDA.")
        return

    # =====================================================
    # 🔹 1. KPI CARDS
    # =====================================================
    st.markdown("### Resumen general del dataset")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Observaciones totales", f"{len(df):,}")
    with col2:
        st.metric("Variables del proceso", len(FEATURE_COLS))
    with col3:
        n_faults = df["fault"].nunique() if "fault" in df.columns else 0
        st.metric("Modos de operación", n_faults)

    st.markdown("---")

    # =====================================================
    # 🔹 2. VISTA GENERAL
    # =====================================================
    st.markdown("### Vista general del dataset")

    st.dataframe(df.head(), use_container_width=True)

    st.download_button(
        "📥 Descargar dataset completo",
        df.to_csv(index=False).encode("utf-8"),
        "EDA_dataset.csv",
        "text/csv"
    )

    st.markdown("---")

    # =====================================================
    # 🔹 3. ESTADÍSTICOS DESCRIPTIVOS
    # =====================================================
    st.markdown("### Estadísticos descriptivos de los sensores")

    desc = df[FEATURE_COLS].describe().T
    st.dataframe(desc, use_container_width=True)

    st.info("**Interpretación**: Busca sensores con alta desviación estándar → suelen ser clave para diagnóstico.")

    st.markdown("---")

    # =====================================================
    # 🔹 4. HISTOGRAMA INTERACTIVO
    # =====================================================
    st.markdown("### Histograma interactivo")

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        sensor_hist = st.selectbox("Selecciona una variable:", FEATURE_COLS, index=0)

    fig_hist = px.histogram(df, x=sensor_hist, nbins=50,
                            title=f"Distribución de {sensor_hist}",
                            color_discrete_sequence=["#3E66F2"])

    st.plotly_chart(fig_hist, use_container_width=True)

    st.info("**Interpretación**: Mira si la distribución es normal, sesgada o tiene varios picos → puede indicar distintos modos de operación.")

    st.markdown("---")

    # =====================================================
    # 🔹 5. BOXPLOT POR TIPO DE FALLO
    # =====================================================
    if "fault_type" in df.columns:
        st.markdown("### Boxplot por Tipo de Fallo")

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            sensor_box = st.selectbox("Variable para boxplot:", FEATURE_COLS)

        df_plot = df.copy()

        fig_box = px.box(df_plot, x="fault_type", y=sensor_box,
                         title=f"{sensor_box} por tipo de fallo",
                         color="fault_type")
        st.plotly_chart(fig_box, use_container_width=True)

        st.info("**Interpretación**: Si un fallo desplaza la mediana o aumenta la dispersión, ese sensor es un buen indicador del fallo.")
    else:
        st.warning("El dataset no tiene columna 'fault_type', por lo que no se puede generar el boxplot por fallo.")

    st.markdown("---")

    # =====================================================
    # 🔹 6. MATRIZ DE CORRELACIÓN
    # =====================================================
    st.markdown("### Matriz de correlación")

    num_vars = st.slider(
        "Número de sensores a mostrar:",
        min_value=5,
        max_value=25,
        value=12,
    )

    vars_corr = FEATURE_COLS[:num_vars]
    corr = df[vars_corr].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Matriz de correlación"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.info("**Interpretación:** Correlaciones altas (>0.8) indican sensores redundantes. Valores negativos fuertes indican relación inversa real del proceso.")

# =====================================================
# PÁGINA 2: ENTRENAMIENTO (RESUMEN)
# =====================================================

def page_training():
    import json

    st.title("Resultados de Entrenamiento")

    st.markdown("""
    Esta sección resume el rendimiento de los modelos entrenados durante la **Fase 1**  
    (Random Forest y LSTM). Las métricas se cargan automáticamente desde los archivos JSON
    generados durante el entrenamiento.
    """)

    # Rutas de los JSON
    RF_METRICS = "../1 PreparacionDatos/metrics_rf.json"
    LSTM_METRICS = "../1 PreparacionDatos/metrics_lstm.json"

    # Cargar métricas
    def load_metrics(path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    metrics_rf = load_metrics(RF_METRICS)
    metrics_lstm = load_metrics(LSTM_METRICS)

    st.markdown("## Métricas de los Modelos")

    col1, col2 = st.columns(2)

    # ---------------- RANDOM FOREST ----------------
    with col1:
        st.markdown("### Modelo Rápido: Random Forest")

        if metrics_rf:
            st.success("Métricas cargadas correctamente.")

            st.metric("Accuracy", f"{metrics_rf['accuracy']:.3f}")
            st.metric("Precision", f"{metrics_rf['precision']:.3f}")
            st.metric("Recall", f"{metrics_rf['recall']:.3f}")
            st.metric("F1-Score", f"{metrics_rf['f1_score']:.3f}")

        else:
            st.error("No se encontró `metrics_rf.json`")
            st.info("Ejecuta el script de generación de métricas para crearlo.")

    # ---------------- LSTM ----------------
    with col2:
        st.markdown("### Modelo Secuencial: LSTM")

        if metrics_lstm:
            st.success("Métricas cargadas correctamente.")

            st.metric("Accuracy", f"{metrics_lstm['accuracy']:.3f}")
            st.metric("Precision", f"{metrics_lstm['precision']:.3f}")
            st.metric("Recall", f"{metrics_lstm['recall']:.3f}")
            st.metric("F1-Score", f"{metrics_lstm['f1_score']:.3f}")

        else:
            st.error("No se encontró `metrics_lstm.json`")
            st.info("Ejecuta el script de generación de métricas para crearlo.")

    # ---------------- Health Check ----------------

    st.markdown("---")
    st.subheader("Health Check de la API BentoML")

    if st.button("Verificar estado del servidor"):
        estado = call_healthcheck()
        st.json(estado)


# =====================================================
# PÁGINA 3: MONITORIZACIÓN EN TIEMPO REAL
# =====================================================

def init_session_state():
    if "current_index" not in st.session_state:
        st.session_state["current_index"] = 0
    if "prediction_log" not in st.session_state:
        st.session_state["prediction_log"] = []  # lista de dicts


def page_monitoring():
    st.title("Monitorización en Tiempo Real")

    df_test = load_test_scaled()
    scaler = load_scaler()
    if df_test is None or df_test.empty:
        return

    init_session_state()

    st.markdown(
        """
        Esta pestaña simula la llegada de datos del **Tennessee Eastman Process** cada **3 minutos** 
        usando las filas de `X_test_scaled`:

        1. Se toma una fila de `X_test_scaled` como si fuera una nueva lectura del proceso.  
        2. Si hay *scaler*, se des-normalizan los 52 sensores para mandarlos a la API.  
        3. Se envían al servicio BentoML:  
           - `predecir_fallo_rapido` → Random Forest (rápido).  
           - `predecir_fallo_inteligente` → LSTM (usa una ventana de 10 pasos).
        """
    )

    modo = st.radio(
        "Modo de predicción:",
        ("Predicción rápida (Random Forest)", "Predicción inteligente (LSTM)"),
        horizontal=True,
    )

    col_left, col_right = st.columns([2, 1])

    # -------------------- LADO IZQUIERDO: LECTURA ACTUAL --------------------
    with col_left:
        st.subheader("Lectura actual (normalizada de X_test_scaled)")

        idx = st.session_state["current_index"]
        if idx >= len(df_test):
            idx = 0
            st.session_state["current_index"] = 0

        fila_actual = df_test.iloc[idx]
        st.dataframe(fila_actual.to_frame().T)

        if st.button("🔁 Simular nueva lectura (3 minutos después)"):
            st.session_state["current_index"] = (idx + 1) % len(df_test)
            # en un sistema real serían 180s; aquí solo recargamos la página
            st.rerun()

    # -------------------- LADO DERECHO: PREDICCIÓN --------------------
    with col_right:
        st.subheader("Predicción del estado")

        if st.button(" Analizar estado actual"):
            idx = st.session_state["current_index"]
            fila_actual = df_test.iloc[idx]

            # 1) Des-normalizar la fila
            sensores_raw = inverse_scale_row(fila_actual, scaler)

            # 2) Llamar al modelo escogido
            if modo.startswith("Predicción rápida"):
                with st.spinner("Llamando a predecir_fallo_rapido..."):
                    respuesta = call_api_rapido(sensores_raw)
            else:
                # Construimos secuencia de 10 pasos
                inicio = max(0, idx - 9)
                fin = idx + 1
                seq_df = df_test.iloc[inicio:fin]

                if len(seq_df) < 10:
                    primera = seq_df.iloc[0:1]
                    faltan = 10 - len(seq_df)
                    seq_df = pd.concat([primera] * faltan + [seq_df], ignore_index=True)

                secuencia_raw = inverse_scale_sequence(seq_df, scaler)

                with st.spinner("Llamando a predecir_fallo_inteligente (LSTM)..."):
                    respuesta = call_api_inteligente(sensores_raw, secuencia_raw)

            # 3) Mostrar resultado en formato "semáforo"
            st.markdown("### Resultado")
            st.markdown(interpretar_respuesta(respuesta))

            # 4) Guardar en log si no es error
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
    st.subheader("Historial de predicciones (sesión actual)")

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