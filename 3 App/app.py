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
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 1. CONFIGURACIÓN DE LA PÁGINA
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

# 2. CONSTANTES Y RUTAS
DATA_PATH = "../data/TEP_csv/datos_eda.csv"
X_TEST_PATH = "../data/processed/X_test_scaled.csv"
Y_TEST_PATH = "../data/processed/y_test.csv"
RF_MODEL_PATH = "../1 PreparacionDatos/tep_rf_model_optimized.pkl"

# API BentoML 
API_URL = "http://localhost:3000"

FEATURE_COLS = [
    f"XMEAS_{i}" if i < 42 else f"XMV_{i-41}" for i in range(1, 53)
]

# 3. FUNCIONES DE CARGA
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


# 4. LÓGICA DE API (BENTOML)
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


# 5. ESTRUCTURA DE LA APP
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3043/3043187.png", width=80)
st.sidebar.title("Panel de Control TEP")
st.sidebar.markdown("Sistema de detección de fallos en procesos químicos.")
menu = st.sidebar.radio("Navegación", ["1. Análisis de Datos (EDA)", "2. Entrenamiento y Validación", "3. Monitorización Real-Time"])

# PESTAÑA 1: EDA
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
                # Coger una muestra para no saturar el gráfico
                fig_line = px.line(df_filtered.iloc[:2000], y=var_sel, color='fault_type' if 'fault_type' in df.columns else None,
                                  title=f"Comportamiento de {var_sel} (Muestra)", template="plotly_dark")
                st.plotly_chart(fig_line, width='stretch')
            
            with tab2:
                fig_hist = px.histogram(df_filtered, x=var_sel, color='fault_type' if 'fault_type' in df.columns else None,
                                       barmode="overlay", template="plotly_dark", title=f"Histograma de {var_sel}")
                st.plotly_chart(fig_hist, width='stretch')
                
        # Matriz de Correlación
        st.subheader("Matriz de Correlación (Top 10 Variables)")
        numeric_df = df_filtered.select_dtypes(include=[np.number]).iloc[:, :10] # Solo las primeras 10 para demo
        corr = numeric_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark")
        st.plotly_chart(fig_corr, width='stretch')


        # Análisis de Sensores Críticos por Fallo
        st.markdown("---")
        st.subheader("🔬 Sensores Críticos por Tipo de Fallo")
        st.info("Identifica qué sensores se alteran más en cada tipo de fallo específico")
        
        if 'fault' in df.columns and 'fault_type' in df.columns:
            # Selector de fallo a analizar
            fallos_disponibles = sorted([f for f in df['fault'].unique() if f != 0])
            fallo_analizar = st.selectbox(
                "Selecciona un tipo de fallo para analizar:",
                fallos_disponibles,
                format_func=lambda x: f"Fallo {x}" if x > 0 else "Normal"
            )
            
            if fallo_analizar:
                # Separar datos normales vs. fallo seleccionado
                df_normal = df[df['fault'] == 0]
                df_fallo = df[df['fault'] == fallo_analizar]
                
                # Calcular estadísticas solo para columnas numéricas de sensores
                # Excluir columnas no numéricas y metadatos
                sensor_cols = [c for c in df.columns if c not in ['fault', 'fault_type', 'sample'] and df[c].dtype in ['float64', 'int64']]
                
                if not sensor_cols:
                    st.error("No se encontraron columnas de sensores en los datos.")
                else:
                    # Calcular desviación absoluta media por sensor
                    cambios = {}
                    for col in sensor_cols:
                        media_normal = df_normal[col].mean()
                        media_fallo = df_fallo[col].mean()
                        cambio_abs = abs(media_fallo - media_normal)
                        cambio_pct = ((media_fallo - media_normal) / (media_normal + 1e-10)) * 100
                        
                        cambios[col] = {
                            'cambio_absoluto': cambio_abs,
                            'cambio_porcentual': cambio_pct,
                            'media_normal': media_normal,
                            'media_fallo': media_fallo
                        }
                    
                    # Ordenar por cambio absoluto
                    cambios_ordenados = sorted(cambios.items(), key=lambda x: x[1]['cambio_absoluto'], reverse=True)
                    top_10_sensores = cambios_ordenados[:10]
                
                    # VISUALIZACIÓN 1: Top 10 sensores más alterados
                    col_grafico1, col_grafico2 = st.columns([1, 1])
                    
                    with col_grafico1:
                        st.markdown("#### 📊 Top 10 Sensores Más Alterados")
                        df_top10 = pd.DataFrame([
                            {
                                'Sensor': sensor,
                                'Cambio Absoluto': datos['cambio_absoluto'],
                                'Cambio %': datos['cambio_porcentual']
                            }
                            for sensor, datos in top_10_sensores
                        ])
                        
                        fig_top10 = px.bar(
                            df_top10, 
                            y='Sensor', 
                            x='Cambio Absoluto',
                            orientation='h',
                            color='Cambio Absoluto',
                            color_continuous_scale='Reds',
                            template="plotly_dark",
                            title=f"Sensores críticos para detectar Fallo {fallo_analizar}"
                        )
                        fig_top10.update_layout(yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_top10, use_container_width=True)
                    
                    # VISUALIZACIÓN 2: Comparación Heatmap
                    with col_grafico2:
                        st.markdown("#### 🌡️ Comparación: Normal vs. Fallo")
                        # Tomar los top 10 sensores y comparar
                        top_10_nombres = [s[0] for s in top_10_sensores]
                        
                        comparacion = pd.DataFrame({
                            'Normal': [cambios[s]['media_normal'] for s in top_10_nombres],
                            f'Fallo {fallo_analizar}': [cambios[s]['media_fallo'] for s in top_10_nombres]
                        }, index=top_10_nombres)
                        
                        fig_heat = px.imshow(
                            comparacion.T,
                            text_auto='.2f',
                            color_continuous_scale='RdYlBu_r',
                            template="plotly_dark",
                            aspect='auto',
                            title="Valores promedio de sensores críticos"
                        )
                        st.plotly_chart(fig_heat, use_container_width=True)
                    
                    # TABLA DETALLADA
                    st.markdown("#### 📋 Tabla Detallada de Cambios")
                    df_detalle = pd.DataFrame([
                        {
                            'Sensor': sensor,
                            'Valor Normal': f"{datos['media_normal']:.3f}",
                            'Valor con Fallo': f"{datos['media_fallo']:.3f}",
                            'Cambio Absoluto': f"{datos['cambio_absoluto']:.3f}",
                            'Cambio (%)': f"{datos['cambio_porcentual']:.1f}%"
                        }
                        for sensor, datos in top_10_sensores
                    ])
                    st.dataframe(df_detalle, use_container_width=True)
                    
                    # INSIGHTS AUTOMÁTICOS
                    sensor_mas_critico = top_10_sensores[0][0]
                    cambio_mayor = top_10_sensores[0][1]['cambio_porcentual']
                    
                    st.success(f"""
                    💡 **Insight Clave**: El sensor **{sensor_mas_critico}** es el indicador más fuerte del Fallo {fallo_analizar}, 
                    mostrando un cambio de **{cambio_mayor:.1f}%** respecto a operación normal.
                    """)
        
        else:
            st.warning("⚠️ Los datos no contienen información de fallos para este análisis.")
        
    else:
        st.warning("⚠️ No se encontró 'datos_eda.csv'. Ejecuta primero los scripts de preparación.")


# PESTAÑA 2: ENTRENAMIENTO Y VALIDACIÓN AVANZADA
elif menu == "2. Entrenamiento y Validación":
    st.title("⚙️ Laboratorio de Modelos y Diagnóstico")
    
    # Sub-pestañas para organizar la información
    tab_eval, tab_interp, tab_train = st.tabs(["📉 Evaluación del Modelo", "🔍 Interpretabilidad (XAI)", "🧪 Playground de Entrenamiento"])
    
    X_test, y_test = load_test_data()
    
    # --- TAB 1: EVALUACIÓN 
    with tab_eval:
        st.markdown("### Selección de Modelo a Evaluar")
        
        # Recopilar todos los modelos disponibles (solo modelos .pkl)
        modelos_disponibles = {}
        
        # Modelos pre-entrenados (solo Random Forest)
        preentrenados_dir = "../1 PreparacionDatos/"
        if os.path.exists(os.path.join(preentrenados_dir, "tep_rf_model_optimized.pkl")):
            modelos_disponibles["Random Forest Pre-entrenado"] = os.path.join(preentrenados_dir, "tep_rf_model_optimized.pkl")
        
        # Modelos del playground (solo .pkl)
        models_dir = "../models/"
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.pkl'):
                    nombre_display = file.replace('.pkl', '').replace('modelo_', 'Modelo ').replace('_', ' ')
                    modelos_disponibles[nombre_display] = os.path.join(models_dir, file)
        
        if not modelos_disponibles:
            st.warning("⚠️ No se encontraron modelos para evaluar. Entrena un modelo en el Playground o verifica la carpeta de modelos pre-entrenados.")
        else:
            modelo_seleccionado = st.selectbox(
                "Elige un modelo:",
                options=list(modelos_disponibles.keys()),
                help="Selecciona entre modelos pre-entrenados o los que hayas guardado en el Playground"
            )
            
            modelo_path = modelos_disponibles[modelo_seleccionado]
            
            st.info(f"📂 **Ruta del modelo**: `{modelo_path}`")
            
            if X_test is not None:
                if st.button("Ejecutar Validación en Test", key="btn_validar"):
                    with st.spinner(f"Calculando métricas de {modelo_seleccionado}..."):
                        try:
                            # Cargar modelo (RF, DT)
                            modelo = joblib.load(modelo_path)
                            y_pred = modelo.predict(X_test)
                            y_true = y_test.values.ravel()
                            
                            # Métricas Globales
                            acc = accuracy_score(y_true, y_pred)
                            f1 = f1_score(y_true, y_pred, average='weighted')
                            prec = precision_score(y_true, y_pred, average='weighted')
                            
                            # Mostrar Tarjetas
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Exactitud (Accuracy)", f"{acc:.2%}", delta="Objetivo > 90%")
                            c2.metric("F1-Score Ponderado", f"{f1:.2%}")
                            c3.metric("Precisión Global", f"{prec:.2%}")
                            
                            st.success(f"✅ Evaluación completada con {len(y_true)} muestras")
                            
                            st.divider()
                            
                            col_izq, col_der = st.columns([1, 1])
                            
                            # Matriz de Confusión
                            with col_izq:
                                st.subheader("Matriz de Confusión")
                                cm = confusion_matrix(y_true, y_pred)
                                # Normalizar opción
                                norm_cm = st.checkbox("Normalizar valores", value=True)
                                if norm_cm:
                                    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                                    fmt = '.2f'
                                else:
                                    fmt = 'd'
                                    
                                fig_cm = px.imshow(cm, text_auto=fmt, color_continuous_scale='Viridis', 
                                                   template="plotly_dark", labels=dict(x="Predicción", y="Real"))
                                st.plotly_chart(fig_cm, width='stretch')
                            
                            # Reporte de Clasificación por Clase (Heatmap)
                            with col_der:
                                st.subheader("Rendimiento por Tipo de Fallo")
                                report = classification_report(y_true, y_pred, output_dict=True)
                                df_report = pd.DataFrame(report).transpose()
                                # Quitamos las filas de promedio para ver solo las clases
                                df_classes = df_report.drop(['accuracy', 'macro avg', 'weighted avg'])
                                
                                fig_rep = px.bar(df_classes, y=df_classes.index, x='f1-score', 
                                                 color='f1-score', orientation='h', range_x=[0, 1],
                                                 template="plotly_dark", title="F1-Score por cada Fallo")
                                st.plotly_chart(fig_rep, width='stretch')
                        
                        except Exception as e:
                            st.error(f"❌ Error al cargar o evaluar el modelo: {str(e)}")
                            st.info("Si es un modelo LSTM, verifica que TensorFlow esté instalado correctamente.")
                
                else:
                    st.info("👈 Pulsa el botón para cargar las métricas del modelo seleccionado.")
            else:
                st.error("⚠️ No se encontraron los datos de test. Verifica la carpeta `data/processed/`.")

    # TAB 2: INTERPRETABILIDAD
    with tab_interp:
        st.markdown("### ¿Qué variables está mirando el modelo?")
        st.info("El análisis de **Feature Importance** permite a los ingenieros entender qué sensores son críticos para detectar fallos.")
        
        if os.path.exists(RF_MODEL_PATH):
            rf_model = joblib.load(RF_MODEL_PATH)
            
            # Obtener importancia
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                feature_names = X_test.columns if X_test is not None else [f"Var {i}" for i in range(len(importances))]
                
                df_imp = pd.DataFrame({'Variable': feature_names, 'Importancia': importances})
                df_imp = df_imp.sort_values('Importancia', ascending=False).head(15) # Top 15
                
                fig_imp = px.bar(df_imp, x='Importancia', y='Variable', orientation='h',
                                 template="plotly_dark", color='Importancia', color_continuous_scale='Plasma',
                                 title="Top 15 Variables más Influyentes en la Detección")
                fig_imp.update_layout(yaxis=dict(autorange="reversed")) # De mayor a menor
                st.plotly_chart(fig_imp, width='stretch')
                
                st.markdown("**Conclusión:** Las variables en la parte superior son los primeros indicadores de que algo va mal en el proceso.")
            else:
                st.warning("El modelo cargado no soporta 'feature_importances_' (quizás no es un árbol).")

    # TAB 3: PLAYGROUND (Entrenar y Guardar Modelos) ---
    with tab_train:
        st.markdown("### 🧪 Entrenamiento de Modelos")
        st.markdown("Entrena nuevos modelos Random Forest con diferentes configuraciones y guárdalos en la carpeta `models/`.")
        
        # Cargar datos de entrenamiento
        X_train_path = "../data/processed/X_train_scaled.csv"
        y_train_path = "../data/processed/y_train.csv"
        
        if os.path.exists(X_train_path) and os.path.exists(y_train_path):
            col_config, col_res = st.columns([1, 1])
            
            with col_config:
                st.subheader("⚙️ Configuración del Modelo")
                
                # Seleccionar tipo de modelo
                modelo_tipo = st.selectbox("Tipo de Modelo", ["Random Forest", "Decision Tree"])
                
                # Hiperparámetros
                if modelo_tipo == "Random Forest":
                    n_estimators = st.slider("Número de Árboles", 10, 200, 100, step=10)
                    max_depth = st.slider("Profundidad Máxima", 5, 50, 10)
                    min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
                else:
                    max_depth = st.slider("Profundidad Máxima", 2, 20, 5)
                    criterion = st.selectbox("Criterio", ["gini", "entropy"])
                
                # Nombre del modelo
                st.markdown("---")
                nombre_modelo = st.text_input("Nombre del modelo", f"modelo_{modelo_tipo.lower().replace(' ', '_')}")
                
                btn_entrenar = st.button("🚀 Entrenar y Guardar Modelo", type="primary")
                
            with col_res:
                if btn_entrenar:
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.tree import DecisionTreeClassifier
                    from sklearn.model_selection import train_test_split
                    from datetime import datetime
                    
                    with st.spinner("🔄 Entrenando modelo... Esto puede tardar unos segundos."):
                        # Cargar datos
                        X_train = pd.read_csv(X_train_path)
                        y_train = pd.read_csv(y_train_path).values.ravel()
                        
                        # Dividir en train/validation
                        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
                        
                        # Entrenar modelo
                        if modelo_tipo == "Random Forest":
                            modelo = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                random_state=42,
                                n_jobs=-1
                            )
                        else:
                            modelo = DecisionTreeClassifier(
                                max_depth=max_depth,
                                criterion=criterion,
                                random_state=42
                            )
                        
                        modelo.fit(X_tr, y_tr)
                        
                        # Evaluar
                        acc_train = modelo.score(X_tr, y_tr)
                        acc_val = modelo.score(X_val, y_val)
                        
                        # Crear carpeta models si no existe
                        models_dir = "../models"
                        os.makedirs(models_dir, exist_ok=True)
                        
                        # Guardar modelo con timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        nombre_archivo = f"{nombre_modelo}_{timestamp}.pkl"
                        ruta_completa = os.path.join(models_dir, nombre_archivo)
                        
                        joblib.dump(modelo, ruta_completa)
                        
                        st.success("✅ ¡Modelo entrenado y guardado exitosamente!")
                        
                        # Mostrar resultados
                        st.markdown("### 📊 Resultados")
                        c1, c2 = st.columns(2)
                        c1.metric("Accuracy Entrenamiento", f"{acc_train:.2%}")
                        c2.metric("Accuracy Validación", f"{acc_val:.2%}")
                        
                        # Mostrar información del archivo
                        st.info(f"📁 Modelo guardado en: `{ruta_completa}`")
                        
                        # Feature importance
                        if hasattr(modelo, 'feature_importances_'):
                            st.markdown("### 🔍 Variables Más Importantes")
                            imp_df = pd.DataFrame({
                                'Variable': X_tr.columns,
                                'Importancia': modelo.feature_importances_
                            }).sort_values('Importancia', ascending=False).head(10)
                            
                            fig_imp = px.bar(imp_df, x='Importancia', y='Variable', 
                                           orientation='h', template="plotly_dark",
                                           title="Top 10 Variables")
                            st.plotly_chart(fig_imp, width='stretch')
                else:
                    st.info("👈 Configura los parámetros y pulsa **Entrenar y Guardar Modelo**")
                    
                    # Mostrar modelos guardados
                    models_dir = "../models"
                    if os.path.exists(models_dir):
                        modelos_guardados = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
                        if modelos_guardados:
                            st.markdown("### 📦 Modelos Guardados")
                            for modelo_file in sorted(modelos_guardados, reverse=True)[:5]:
                                st.text(f"• {modelo_file}")
        else:
            st.error("No se encuentran los archivos de datos de entrenamiento.")

# PESTAÑA 3: MONITORIZACIÓN REAL-TIME
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
            st.subheader("🤖 Selección de Modelo")
            
            # Opción de fuente del modelo
            fuente_modelo = st.radio("Fuente del modelo:", 
                                    ["API BentoML (Modelos pre-entrenados)", 
                                     "Modelos Locales (Carpeta models/)"])
            
            if fuente_modelo == "API BentoML (Modelos pre-entrenados)":
                modelo_elegido = st.selectbox("Modelo:", ["Random Forest", "LSTM (Secuencia)"])
                usar_api = True
            else:
                # Listar modelos disponibles en carpeta models/
                models_dir = "../models"
                if os.path.exists(models_dir):
                    modelos_disponibles = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
                    if modelos_disponibles:
                        modelo_elegido = st.selectbox("Selecciona un modelo:", 
                                                     sorted(modelos_disponibles, reverse=True))
                        usar_api = False
                    else:
                        st.warning("No hay modelos guardados. Entrena uno primero en la pestaña de Entrenamiento.")
                        modelo_elegido = None
                        usar_api = False
                else:
                    st.warning("La carpeta 'models/' no existe. Entrena un modelo primero.")
                    modelo_elegido = None
                    usar_api = False
            
            btn_predict = st.button("🔍 Analizar Estado", disabled=(modelo_elegido is None))
            
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

        if 'btn_predict' in locals() and btn_predict and modelo_elegido:
            
            if usar_api:
                # PREDICCIÓN CON API BENTOML
                # 1. Preparar los datos internos
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
                payload_final = {
                    "datos": datos_internos
                }

                # 3. Llamada a la API con el payload correcto
                with st.spinner(f"Consultando modelo {modelo_elegido}..."):
                    respuesta = consultar_api(endpoint, payload_final)
            
            else:
                # PREDICCIÓN CON MODELO LOCAL
                with st.spinner(f"Cargando y ejecutando modelo {modelo_elegido}..."):
                    try:
                        # Cargar modelo desde carpeta models/
                        models_dir = "../models"
                        ruta_modelo = os.path.join(models_dir, modelo_elegido)
                        modelo_local = joblib.load(ruta_modelo)
                        
                        # Hacer predicción
                        prediccion = modelo_local.predict(current_data.reshape(1, -1))[0]
                        
                        # Obtener probabilidades si están disponibles
                        if hasattr(modelo_local, 'predict_proba'):
                            probabilidades = modelo_local.predict_proba(current_data.reshape(1, -1))[0]
                            confianza = float(np.max(probabilidades))
                            dict_probs = {f"fallo_{i}": float(p) for i, p in enumerate(probabilidades)}
                        else:
                            confianza = 1.0
                            dict_probs = {}
                        
                        # Mapeo de nombres de fallos
                        nombres_fallos = {
                            0: "Operación Normal",
                            1: "Fallo Tipo 1", 2: "Fallo Tipo 2", 3: "Fallo Tipo 3",
                            4: "Fallo Tipo 4", 5: "Fallo Tipo 5", 6: "Fallo Tipo 6",
                            7: "Fallo Tipo 7", 8: "Fallo Tipo 8", 9: "Fallo Tipo 9",
                            10: "Fallo Tipo 10", 11: "Fallo Tipo 11", 12: "Fallo Tipo 12",
                            13: "Fallo Tipo 13", 14: "Fallo Tipo 14", 15: "Fallo Tipo 15",
                            16: "Fallo Tipo 16", 17: "Fallo Tipo 17", 18: "Fallo Tipo 18",
                            19: "Fallo Tipo 19", 20: "Fallo Tipo 20", 21: "Fallo Tipo 21"
                        }
                        
                        # Crear respuesta similar a la API
                        respuesta = {
                            "id_fallo": int(prediccion),
                            "nombre_fallo": nombres_fallos.get(int(prediccion), "Desconocido"),
                            "confianza": confianza,
                            "probabilidades": dict_probs,
                            "modelo_usado": modelo_elegido
                        }
                        
                    except Exception as e:
                        respuesta = {
                            "error": True,
                            "detalle": f"Error al cargar o ejecutar el modelo: {str(e)}"
                        }

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
                
                # Mostrar información del modelo usado
                modelo_info = ""
                if not usar_api:
                    modelo_info = f"<p style='color:white; font-size: 0.9em; opacity: 0.7;'>📦 Modelo Local: {respuesta.get('modelo_usado', 'N/A')}</p>"
                
                st.markdown(f"""
                <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h1 style="margin:0; color:white;">{icono} {nombre_fallo}</h1>
                    <h3 style="color:white; opacity: 0.8;">ID Fallo: {fallo_id}</h3>
                    <p style="color:white;">Confianza del modelo: <strong>{confianza:.2%}</strong></p>
                    {modelo_info}
                </div>
                """, unsafe_allow_html=True)
                
                # Mostrar probabilidades si es fallo
                if fallo_id != 0:
                    st.warning("⚠️ Se recomienda inspección inmediata.")
                    
                    # === RECOMENDACIÓN DE COMPONENTES A REVISAR ===
                    df = load_data()
                    if df is not None and 'fault' in df.columns:
                        # Calcular qué sensores cambian más para este fallo específico
                        df_normal = df[df['fault'] == 0]
                        df_fallo_detectado = df[df['fault'] == fallo_id]
                        
                        if len(df_fallo_detectado) > 0:
                            sensor_cols = [c for c in df.columns if c not in ['fault', 'fault_type', 'sample'] and df[c].dtype in ['float64', 'int64']]
                            
                            cambios_componentes = {}
                            for col in sensor_cols:
                                media_normal = df_normal[col].mean()
                                media_fallo = df_fallo_detectado[col].mean()
                                cambio_abs = abs(media_fallo - media_normal)
                                cambios_componentes[col] = cambio_abs
                            
                            # Obtener los 3 sensores más críticos
                            top_3_criticos = sorted(cambios_componentes.items(), key=lambda x: x[1], reverse=True)[:3]
                            
                            if top_3_criticos:
                                st.info("### 🔧 Componentes Prioritarios para Revisión")
                                
                                componentes_texto = ""
                                for idx, (sensor, cambio) in enumerate(top_3_criticos, 1):
                                    componentes_texto += f"{idx}. **{sensor}** (desviación: {cambio:.2f})\n"
                                
                                st.markdown(componentes_texto)
                                st.caption("💡 Estos son los sensores que muestran mayor alteración en este tipo de fallo según datos históricos.")
                    
                    if "probabilidades" in respuesta and respuesta["probabilidades"]:
                        probs = pd.DataFrame(list(respuesta["probabilidades"].items()), columns=["Fallo", "Probabilidad"])
                        # Filtrar solo las altas
                        probs = probs[probs["Probabilidad"] > 0.05].sort_values("Probabilidad", ascending=False)
                        
                        fig_bar = px.bar(probs, x="Probabilidad", y="Fallo", orientation='h', 
                                         template="plotly_dark", title="Análisis de causas probables")
                        st.plotly_chart(fig_bar, width='stretch')