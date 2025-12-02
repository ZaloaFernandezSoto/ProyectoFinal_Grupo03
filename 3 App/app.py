#Fase 3: Aplicación (Streamlit)
#Estructura de la App (app.py):
#Configurar el menú lateral para navegar entre "Análisis de Datos", "Entrenamiento" y "Monitorización en Tiempo Real"
#Pestaña de Exploración (EDA Interactivo):
#Migrar los gráficos del notebook a Streamlit. Permitir filtrar por "Tipo de Fallo" (ej. Fallo 1 vs. Normal) para ver cómo reaccionan las variables.
#Pestaña de Monitorización (Conexión con BentoML):
#Crear una interfaz que simule la llegada de datos cada 3 minutos.
#Conectar el botón de "Analizar Estado Actual" a la API local de BentoML creada en el paso 7.
import streamlit as st

# --- Definición de las PÁGINAS ---

def analisis_datos_page():
    """Contenido para la pestaña de Análisis de Datos (EDA)."""
    st.header("📊 Análisis de Datos Interactivo (EDA)")
    st.write("Aquí irán los gráficos de series temporales, histogramas y correlaciones.")
    # TODO: Implementar la carga de datos y los gráficos interactivos (Plotly/Altair)
    st.markdown("---")
    st.subheader("Filtros")
    fallo_seleccionado = st.selectbox("Seleccionar Tipo de Fallo:", 
                                     ["Normal", "Fallo 1", "Fallo 2", "Todos"])
    st.info(f"Mostrando datos para el estado: **{fallo_seleccionado}**")
    
    # 

def entrenamiento_page():
    """Contenido para la pestaña de Entrenamiento de Modelos."""
    st.header("🧠 Entrenamiento y Comparativa de Modelos")
    st.write("Mostrar un resumen de los modelos entrenados (Random Forest, LSTM), sus métricas (F1-Score) y la matriz de confusión.")
    st.markdown("---")
    st.subheader("Resultados del Mejor Modelo")
    
    # Ejemplo de visualización de métricas
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="F1-Score", value="0.92")
    with col2:
        st.metric(label="Precisión General", value="95%")
    
    # TODO: Mostrar matriz de confusión y gráficos de ROC (si aplica)

def monitorizacion_page():
    """Contenido para la pestaña de Monitorización en Tiempo Real."""
    st.header("🚨 Monitorización en Tiempo Real")
    st.write("Simulación de la llegada de datos de sensor y conexión con la API de BentoML.")
    st.warning("⚠️ **¡IMPORTANTE!** Esta pestaña requiere que el servicio BentoML esté corriendo localmente.")
    st.markdown("---")
    
    # Simulación del estado del sistema
    st.subheader("Simulador de Sensor")
    st.markdown("Presiona el botón para simular una nueva medición y analizar el estado.")
    
    if st.button("Analizar Estado Actual"):
        # TODO: Implementar la lógica para:
        # 1. Generar/Cargar un vector de datos simulado (las 52 variables).
        # 2. Llamar a la API de BentoML (usando 'requests').
        # 3. Mostrar el resultado.
        
        # Placeholder del resultado
        st.success("✅ **Sistema estable.** Predicción de la API: 'Normal'")
        # o st.error("❌ **¡ALERTA!** Fallo X detectado.")


# --- CONFIGURACIÓN PRINCIPAL DE STREAMLIT ---

# 1. Configuración de la página (título y layout)
st.set_page_config(
    page_title="Analítica Predictiva Industrial",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏭 Proyecto: Analítica para la Industria")

# 2. Creación del Menú Lateral
st.sidebar.title("Menú de Navegación")
opcion = st.sidebar.radio(
    "Selecciona una Sección",
    ("Análisis de Datos", "Entrenamiento", "Monitorización en Tiempo Real")
)

# 3. Lógica para renderizar la página seleccionada
if opcion == "Análisis de Datos":
    analisis_datos_page()
elif opcion == "Entrenamiento":
    entrenamiento_page()
elif opcion == "Monitorización en Tiempo Real":
    monitorizacion_page()

# 4. Información adicional en el sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Asignatura:** Analítica para la Industria")