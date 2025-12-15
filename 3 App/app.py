import os
import time
import json
import base64
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =====================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =====================================================
st.set_page_config(
    page_title="TEP - Panel Industrial",
    page_icon="🏭",
    layout="wide",
)

# Estilos CSS corregidos para funcionar con el config.toml
st.markdown("""
    <style>
        /* Ajuste de contenedores de métricas para que parezcan tarjetas industriales */
        div[data-testid="metric-container"] {
            background-color: #262730;
            border: 1px solid #3E66F2;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }
        
        /* Títulos destacados */
        h1, h2, h3 {
            color: #ffffff !important;
        }

        /* Alertas personalizadas */
        .stAlert {
            background-color: #262730;
            color: #ffffff;
        }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# 2. CONSTANTES Y RUTAS
# =====================================================
# Ajusta estas rutas si tus carpetas se llaman diferente
DATA_PATH = "../data/TEP_csv/datos_eda.csv"
X_TEST_PATH = "../data/processed/X_test_scaled.csv"
Y_TEST_PATH = "../data/processed/y_test.csv"
RF_MODEL_PATH = "../1 PreparacionDatos/tep_rf_model_optimized.pkl"

# API BentoML (Asegúrate de que el puerto coincida con el comando 'bentoml serve')
API_URL = "http://localhost:3000"

FEATURE_COLS = [
    f"XMEAS_{i}" if i < 42 else f"XMV_{i-41}" for i in range(1, 53)
]
# Si tus columnas tienen nombres reales (Reactor_Pressure, etc), el script de carga
# de datos debería manejarlos. Aquí usaremos los nombres del CSV cargado.

# =====================================================
# 3. FUNCIONES DE CARGA
# =====================================================
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

@st.cache_data
def load_test_data():
    if os.path.exists(X_TEST_PATH) and os.path.exists(Y_TEST_PATH):
        X = pd.read_csv(X_TEST_PATH)
        y = pd.read_csv(Y_TEST_PATH)
        return X, y
    return None, None

# =====================================================
# 4. LÓGICA DE API (BENTOML)
# =====================================================
def consultar_api(endpoint, datos_json):
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}",
            headers={"content-type": "application/json"},
            data=json.dumps(datos_json),
            timeout=2
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"Error {response.status_code}", "detalle": response.text}
    except requests.exceptions.ConnectionError:
        return {"error": "Conexión rechazada", "detalle": "Asegúrate de que 'bentoml serve' está corriendo."}
    except Exception as e:
        return {"error": "Excepción", "detalle": str(e)}

# =====================================================
# 5. ESTRUCTURA DE LA APP
# =====================================================

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3043/3043187.png", width=80)
st.sidebar.title("Panel de Control TEP")
st.sidebar.markdown("Sistema de detección de fallos en procesos químicos.")
menu = st.sidebar.radio("Navegación", ["1. Análisis de Datos (EDA)", "2. Entrenamiento y Validación", "3. Monitorización Real-Time"])

# -----------------------------------------------------
# PESTAÑA 1: EDA
# -----------------------------------------------------
if menu == "1. Análisis de Datos (EDA)":
    st.title("📊 Análisis Exploratorio del Proceso")
    
    df = load_data()
    if df is not None:
        # Tarjetas de resumen
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Muestras", df.shape[0])
        c2.metric("Variables", df.shape[1])
        c3.metric("Tipos de Fallo", df['fault'].nunique() if 'fault' in df.columns else "N/A")
        
        st.markdown("---")
        
        # Filtros
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("Configuración")
            if 'fault' in df.columns:
                fallos_sel = st.multiselect("Filtrar por Fallo:", sorted(df['fault'].unique()), default=[0, 1])
                df_filtered = df[df['fault'].isin(fallos_sel)]
            else:
                df_filtered = df
                
            var_sel = st.selectbox("Variable a Analizar:", [c for c in df.columns if c not in ['fault', 'fault_type', 'sample']])

        with col2:
            # Gráfico de líneas o histograma
            tab1, tab2 = st.tabs(["📉 Evolución Temporal", "📊 Distribución"])
            
            with tab1:
                # Tomamos una muestra para no saturar el gráfico
                fig_line = px.line(df_filtered.iloc[:2000], y=var_sel, color='fault_type' if 'fault_type' in df.columns else None,
                                  title=f"Comportamiento de {var_sel} (Muestra)", template="plotly_dark")
                st.plotly_chart(fig_line, use_container_width=True)
            
            with tab2:
                fig_hist = px.histogram(df_filtered, x=var_sel, color='fault_type' if 'fault_type' in df.columns else None,
                                       barmode="overlay", template="plotly_dark", title=f"Histograma de {var_sel}")
                st.plotly_chart(fig_hist, use_container_width=True)
                
        # Matriz de Correlación
        st.subheader("Matriz de Correlación (Top 10 Variables)")
        numeric_df = df_filtered.select_dtypes(include=[np.number]).iloc[:, :10] # Solo las primeras 10 para demo
        corr = numeric_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)
        
    else:
        st.warning("⚠️ No se encontró 'datos_eda.csv'. Ejecuta primero los scripts de preparación.")

# -----------------------------------------------------
# PESTAÑA 2: ENTRENAMIENTO
# -----------------------------------------------------
elif menu == "2. Entrenamiento y Validación":
    st.title("⚙️ Rendimiento de los Modelos")
    
    X_test, y_test = load_test_data()
    
    if X_test is not None and os.path.exists(RF_MODEL_PATH):
        
        # Cargar modelo RF para mostrar métricas reales
        rf_model = joblib.load(RF_MODEL_PATH)
        
        st.markdown("### Evaluación del modelo Random Forest")
        
        if st.button("Recalcular Métricas en Test"):
            with st.spinner("Realizando inferencia en conjunto de test..."):
                y_pred = rf_model.predict(X_test)
                y_true = y_test.values.ravel()
                
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted')
                
                c1, c2 = st.columns(2)
                c1.metric("Exactitud (Accuracy)", f"{acc:.2%}", delta="Objetivo > 90%")
                c2.metric("F1-Score (Ponderado)", f"{f1:.2%}")
                
                st.markdown("#### Matriz de Confusión")
                cm = confusion_matrix(y_true, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Viridis', template="plotly_dark",
                                   labels=dict(x="Predicción", y="Real"), title="Matriz de Confusión")
                st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Pulsa el botón para ejecutar la validación sobre los datos de prueba procesados.")
            
        st.markdown("### Comparativa Teórica (RF vs LSTM)")
        st.markdown("""
        | Métrica | Random Forest (Rápido) | LSTM (Secuencial) |
        |---------|------------------------|-------------------|
        | **Input** | Vector instantáneo (t) | Ventana temporal (t-10 a t) |
        | **Velocidad** | < 50ms | ~200ms |
        | **Uso** | Detección inmediata | Diagnóstico complejo |
        """)
        
    else:
        st.error("No se encuentran los archivos de test o el modelo .pkl. Revisa la carpeta 'data/processed'.")

# -----------------------------------------------------
# PESTAÑA 3: MONITORIZACIÓN REAL-TIME
# -----------------------------------------------------
elif menu == "3. Monitorización Real-Time":
    st.title("📡 Sala de Control")
    st.markdown("Simulación de conexión con sensores IoT y predicción mediante API BentoML.")
    
    col_control, col_display = st.columns([1, 2])
    
    with col_control:
        st.subheader("Simulador")
        
        # Cargar datos para simular
        X_test, _ = load_test_data()
        
        if X_test is not None:
            # Seleccionar un índice aleatorio para simular
            if 'sensor_idx' not in st.session_state:
                st.session_state.sensor_idx = 0
                
            if st.button("🎲 Generar nueva lectura"):
                st.session_state.sensor_idx = np.random.randint(0, len(X_test))
            
            # Obtener datos
            current_data = X_test.iloc[st.session_state.sensor_idx].values
            
            st.write("---")
            st.write("**Datos del sensor (Normalizados):**")
            st.dataframe(pd.DataFrame(current_data.reshape(1, -1), columns=X_test.columns).T.head(10), height=300)
            
            st.write("---")
            modelo_elegido = st.radio("Modelo:", ["Random Forest", "LSTM (Secuencia)"])
            
            btn_predict = st.button("🔍 Analizar Estado")
            
        else:
            st.error("No hay datos de test para simular.")

    with col_display:
        st.subheader("Diagnóstico del Sistema")
        
        # Contenedor para resultados
        result_container = st.empty()
        
        # Verificar estado API al inicio
        health = consultar_api("verificar_servidor", {})
        if "error" in health:
             st.error(f"⚠️ La API no responde: {health['detalle']}")
             st.info("Ejecuta en terminal: `bentoml serve service.py:svc --reload`")
        else:
             st.success(f"🟢 API Conectada: {health.get('status', 'OK')}")

        if 'btn_predict' in locals() and btn_predict:
            
            # 1. Preparar los datos internos (lo que va dentro de DatosDeEntrada)
            datos_internos = {
                "sensores": current_data.tolist()
            }
            
            if modelo_elegido == "Random Forest":
                endpoint = "predecir_fallo_rapido"
            else:
                # Para LSTM simulamos la secuencia cogiendo 10 filas anteriores
                idx = st.session_state.sensor_idx
                if idx < 10: idx = 10
                secuencia = X_test.iloc[idx-10:idx].values.tolist()
                
                # Añadimos la secuencia a los datos internos
                datos_internos["secuencia_temporal"] = secuencia
                endpoint = "predecir_fallo_inteligente"

            # 2. IMPORTANTE: Envolver todo en la clave "datos"
            # Esto es necesario porque tu función en service.py se define como:
            # def predecir...(self, datos: DatosDeEntrada)
            payload_final = {
                "datos": datos_internos
            }

            # 3. Llamada a la API con el payload correcto
            with st.spinner(f"Consultando modelo {modelo_elegido}..."):
                respuesta = consultar_api(endpoint, payload_final)

            # Mostrar resultados visuales
            if "error" in respuesta:
                st.error(f"Error en predicción: {respuesta['detalle']}")
            else:
                # (El resto del código de visualización se mantiene igual)
                fallo_id = respuesta.get("id_fallo", -1)
                nombre_fallo = respuesta.get("nombre_fallo", "Desconocido")
                confianza = respuesta.get("confianza", 0)
                
                # Diseño de la tarjeta de resultado
                color = "#28a745" if fallo_id == 0 else "#dc3545" # Verde o Rojo
                icono = "✅" if fallo_id == 0 else "🚨"
                
                st.markdown(f"""
                <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h1 style="margin:0; color:white;">{icono} {nombre_fallo}</h1>
                    <h3 style="color:white; opacity: 0.8;">ID Fallo: {fallo_id}</h3>
                    <p style="color:white;">Confianza del modelo: <strong>{confianza:.2%}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Mostrar probabilidades si es fallo
                if fallo_id != 0:
                    st.warning("⚠️ Se recomienda inspección inmediata.")
                    if "probabilidades" in respuesta and respuesta["probabilidades"]:
                        probs = pd.DataFrame(list(respuesta["probabilidades"].items()), columns=["Fallo", "Probabilidad"])
                        # Filtrar solo las altas
                        probs = probs[probs["Probabilidad"] > 0.05].sort_values("Probabilidad", ascending=False)
                        
                        fig_bar = px.bar(probs, x="Probabilidad", y="Fallo", orientation='h', 
                                         template="plotly_dark", title="Análisis de causas probables")
                        st.plotly_chart(fig_bar, use_container_width=True)