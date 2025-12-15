# 1 IMPORTS: Importamos la librerías necesarias

import bentoml
import numpy as np
import joblib
import os
import logging
from typing import Optional, List
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model


# Configurmos los mensajes que apareceran en la terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 1 DATOS: Definimos las estructuras de datos para el API --> datos que recibe y devuelve 
# 1.1: Datos de entrada (lo que alguien envía al API/ datos que recibe)
class DatosDeEntrada(BaseModel):
    
    # Campo 1: Array OBLIGATORIO de exactamente 52 números
    sensores: List[float] = Field(
        ...,
        description="Array con 52 valores de sensores normalizados",
        min_length=52,
        max_length=52
    )
    
    # Campo 2: Array OPCIONAL para LSTM con datos temporales 
    secuencia_temporal: Optional[List[List[float]]] = Field(
        None, #Es opcional
        description="Secuencia de 10 pasos x 52 sensores (solo para LSTM)"
    )


# 1.2: Datos de salida (lo que devuelve el API)
class RespuestaAPI(BaseModel):
    id_fallo: int                          # Id del fallo (0-21)
    nombre_fallo: str                      # Nombre descriptivo
    confianza: float                       # Porcentaje (0.0 a 1.0)
    modelo_usado: str                      # Modelo empleado
    probabilidades: Optional[dict] = None  # Detalles de todas las opciones



# 2: CREAMOS EL SERVIDOR 
# Crear el servidor API de BentoML
@bentoml.service(name="detector_Fallos")
class DetectorFallos:
    
    # 3: VARIABLES PARA GUARDAR LOS MODELOS EN MEMORIA
    modelo_random_forest = None    
    modelo_lstm = None             
    scaler = None
    
    # Diccionario que traduce números de fallos a mensajes para el usuario
    NOMBRES_FALLOS = {
        0: "Operación Normal",
        1: "Fallo Tipo 1", 2: "Fallo Tipo 2", 3: "Fallo Tipo 3", 
        4: "Fallo Tipo 4", 5: "Fallo Tipo 5", 6: "Fallo Tipo 6",
        7: "Fallo Tipo 7", 8: "Fallo Tipo 8", 9: "Fallo Tipo 9",
        10: "Fallo Tipo 10", 11: "Fallo Tipo 11", 12: "Fallo Tipo 12",
        13: "Fallo Tipo 13", 14: "Fallo Tipo 14", 15: "Fallo Tipo 15",
        16: "Fallo Tipo 16", 17: "Fallo Tipo 17", 18: "Fallo Tipo 18",
        19: "Fallo Tipo 19", 20: "Fallo Tipo 20", 21: "Fallo Tipo 21"
    }
    
    def __init__(self):
        # 4: CARGAR LOS MODELOS DE LA FASE 1
        self.cargar_modelos()
        logger.info("SERVIDOR LISTO PARA RECIBIR PETICIONES")
    
    def cargar_modelos(self):
        """Cargamos los modelos desde la carpeta de la fase 1"""
        
        # 4.1 Cargamos Random Forest
        try:
            logger.info("Cargando Random Forest...")
            ruta = "../1 PreparacionDatos/tep_rf_model_optimized.pkl"
            
            if os.path.exists(ruta):
                self.modelo_random_forest = joblib.load(ruta)
                logger.info(f"Random Forest cargado desde: {ruta}")
            else:
                logger.error(f"Archivo no encontrado en: {ruta}")

        except Exception as error:
            logger.error(f"Error cargando Random Forest: {error}")    
        
        # 4.2: Cargamos LSTM
        try:
            logger.info("Cargando LSTM...")
            ruta = "../1 PreparacionDatos/tep_lstm_model.keras"
            
            if os.path.exists(ruta):
                self.modelo_lstm = load_model(ruta)
                logger.info(f"LSTM cargado desde: {ruta}")
            else:
                logger.error(f"Archivo no encontrado en: {ruta}")
        
        except Exception as error:
            logger.error(f"Error cargando LSTM: {error}")
        
      
        # 4.3: Por último cargamos el normalizador (scaler)
        try:
            logger.info("Cargando normalizador...")
            ruta = "../1 PreparacionDatos/tep_scaler.pkl"
            
            if os.path.exists(ruta):
                self.scaler = joblib.load(ruta)
                logger.info(f"Normalizador cargado desde: {ruta}")
            else:
                logger.error(f"Archivo no encontrado en: {ruta}")
        
        except Exception as error:
            logger.error(f"Error cargando normalizador: {error}")

    
    # 6: DETECCIÓN DE FALLOS CON RANDOM FOREST (ENDPOINT 1)
    @bentoml.api
    def predecir_fallo_rapido(self, datos: DatosDeEntrada) -> RespuestaAPI:
        """
        Endpoint 1: Predicción RÁPIDA usando Random Forest
        Entrada: JSON con 52 sensores
        Salida: JSON con ID del fallo, confianza y probabilidades
        """
        # Comprobamos si el modelo se ha cargado realmente 
        if self.modelo_random_forest is None:
            raise Exception("Random Forest no se ha cargado")
        
        #Si se ha cargado hacemos la predicción
        try:
            #Pasamos los datos de entrada (sensores) a un array de numpy
            sensores = np.array(datos.sensores).reshape(1, -1)  # Aseguramos que es 2D: (1,52)  
            logger.info(f"Sensores recibidos: {sensores.shape}")
            
            #Normalizamos los datos 
            #Debemos comprobar si existe el scaler encargado de normalizarlos
            if self.scaler is not None:
                sensores = self.scaler.transform(sensores) #normalizamos
                logger.info("Datos normalizados")
            else:
                logger.warning("Normalizador no disponible, usando datos sin normalizar")
            

            # Finalmente hacemos la predicción
            id_fallo = int(self.modelo_random_forest.predict(sensores)[0])
            logger.info(f"Predicción: Fallo {id_fallo}")
            
      
            # Obtenemos la probabilidad estimada por el modelo para la clase escogida -->confianza
            # Utilizamos el método del modelo para obtener las probabilidades 
            if hasattr(self.modelo_random_forest, 'predict_proba'):
                probabilidades = self.modelo_random_forest.predict_proba(sensores)[0]
                confianza = float(np.max(probabilidades))  # La maxima probabilidad es la confianza
                
                # Creamos un diccionario con todas las probabilidades
                dict_probabilidades = {
                    f"fallo_{i}": float(p) 
                    for i, p in enumerate(probabilidades)
                }
            #En caso de no haber probabilidades usamos una confianza del 100% --> 1.0
            else:
                confianza = 1.0
                dict_probabilidades = None #No tendremos diccionario de probabilidades
        
            logger.info(f"Confianza: {confianza:.2%}")
            
            #Obtener el nombre del fallo
            nombre_fallo = self.NOMBRES_FALLOS.get(
                id_fallo,
                f"Fallo desconocido {id_fallo}"
            )
            
            # Paso 6: DEVOLVEMOS LA RESPUESTA DEL API
            respuesta = RespuestaAPI(
                id_fallo=id_fallo,
                nombre_fallo=nombre_fallo,
                confianza=confianza,
                modelo_usado="Random Forest",
                probabilidades=dict_probabilidades
            )
            return respuesta
        
        except Exception as error:
            logger.error(f"Error en la predicción del modelo: {error}")
            raise



    # 7: PREDICCIÓN CON LSTM (ENDPOINT 2)
    @bentoml.api
    def predecir_fallo_inteligente(self, datos: DatosDeEntrada) -> RespuestaAPI:
        """
        Endpoint 2: Predicción INTELIGENTE usando LSTM
        URL: POST http://localhost:3000/predecir_fallo_inteligente
        
        Entrada: 52 sensores + secuencia temporal de 10 pasos
        Salida: ID del fallo + confianza
        Tiempo: ~500ms (más lento pero más preciso)
        """
        #Comprobamos si el modelo se ha cargado realmente 
        if self.modelo_lstm is None:
            raise Exception("LSTM no cargado. Instala TensorFlow o revisa los logs.")
        
        try:
           #Comprobamos que tenemos la lista temporal necesaria para LSM 
            if datos.secuencia_temporal is None:
                raise ValueError(
                    "Se necesita enviar la 'secuencia_temporal' para LSTM "
                )
            logger.info("Secuencia temporal recibida")
            
            
            # Convertimos los datos de entrada a un array de numpy y verificamos tambien su tamaño
           
            secuencia = np.array(datos.secuencia_temporal)
            if secuencia.shape != (10, 52):
                raise ValueError(
                    f"Tamaño incorrecto. Debe ser (10, 52) "
                    f"en cambio es {secuencia.shape}"
                )
            logger.info(f"Secuencia validada: {secuencia.shape}")
            
            
            # Normalizamos cada paso de la secuencia
           #Comrpobamos que tenemos el scaler para normalizar
            if self.scaler is not None:
                for paso in range(len(secuencia)):
                    # Normalizamos fila por fila
                    secuencia[paso] = self.scaler.transform(
                        secuencia[paso].reshape(1, -1)
                    )[0]
                logger.info("Secuencia normalizada")
            
           
            # Cambiamos forma para LSTM       
            # LSTM espera: (batch_size, time_steps, features) --> (1,10,52)
            secuencia = secuencia.reshape(1, 10, 52)
            logger.info(f"Secuencia cambaida correctamente para LSTM: {secuencia.shape}")

            #Predecimos con el modelo LSTM --> obtenemos probabilidades y confianza
            probabilidades = self.modelo_lstm.predict(secuencia, verbose=0)[0]
            id_fallo = int(np.argmax(probabilidades))
            confianza = float(np.max(probabilidades))
            
            logger.info(f"Predicción LSTM: Fallo {id_fallo} ({confianza:.2%})")
            
            #Creamos el diccionario de probabilidades
            dict_probabilidades = {
                f"fallo_{i}": float(p)
                for i, p in enumerate(probabilidades)
            }
            #Obtenemos el nombre del fallo
            nombre_fallo = self.NOMBRES_FALLOS.get(
                id_fallo,
                f"Fallo desconocido {id_fallo}"
            )
            # Devolvemos la respuesta del API
            return RespuestaAPI(
                id_fallo=id_fallo,
                nombre_fallo=nombre_fallo,
                confianza=confianza,
                modelo_usado="LSTM ",
                probabilidades=dict_probabilidades
            )
        
        except Exception as error:
            logger.error(f"Error en predicción LSTM: {error}")
            raise


    # 8: VERIFICAMOS EL ESTADO DEL SERVIDOR (ENDPOINT 3)
    @bentoml.api
    def verificar_servidor(self) -> dict:
        """
        Endpoint 3: Verificar que el servidor está funcionando
        URL: GET http://localhost:3000/verificar_servidor
        
        Respuesta: Estado de los modelos
        """
        #definimos el estado del servidor y los modelos
        estado = {
            "status": "OPERATIVO",
            "random_forest_cargado": self.modelo_random_forest is not None,
            "lstm_cargado": self.modelo_lstm is not None,
            "normalizador_cargado": self.scaler is not None
        }
        # lo devolvemos
        logger.info(f"Health Check: {estado}")
        
        return estado
    

svc = DetectorFallos()