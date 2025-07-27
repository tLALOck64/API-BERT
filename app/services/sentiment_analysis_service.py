import pandas as pd
import re
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()

class SentimentAnalysisService:
    
    DATASET_PATH = "datasets/comentarios.csv"
    ABREVIACIONES_PATH = "app/tools/diccionario_abrevaciones.csv"
    RESULTADO_PATH = "datasets/intenciones_sentimientos.csv"
    HF_TOKEN = os.getenv('HUGGIN_FACE_TOKEN')
    
    # Configuración de modelos
    MODELO_SENTIMIENTO_ID = "nlptown/bert-base-multilingual-uncased-sentiment"
    MODELO_INTENCION_ID = "facebook/bart-large-mnli"
    INTENCIONES = ["appreciation", "complaint", "request", "bug", "confusion"]
    
    def __init__(self):
        self.clasificador_sentimiento = None
        self.clasificador_intencion = None
        self._init_models()
    
    def _init_models(self):
        try:
            self.clasificador_sentimiento = pipeline(
                "sentiment-analysis", 
                model=self.MODELO_SENTIMIENTO_ID, 
                token=self.HF_TOKEN
            )
            self.clasificador_intencion = pipeline(
                "zero-shot-classification", 
                model=self.MODELO_INTENCION_ID, 
                token=self.HF_TOKEN
            )
        except Exception as e:
            print(f"Error al inicializar modelos: {e}")
            # Intentar sin token si falla
            try:
                print("Intentando cargar modelos sin token...")
                self.clasificador_sentimiento = pipeline(
                    "sentiment-analysis", 
                    model=self.MODELO_SENTIMIENTO_ID
                )
                self.clasificador_intencion = pipeline(
                    "zero-shot-classification", 
                    model=self.MODELO_INTENCION_ID
                )
                print("Modelos cargados exitosamente sin token")
            except Exception as e2:
                print(f"Error al cargar modelos sin token: {e2}")
                raise

    def cargar_abreviaciones(self) -> dict:
        try:
            # Buscar el archivo en varias ubicaciones posibles
            posibles_rutas = [
                self.ABREVIACIONES_PATH,
                f"../{self.ABREVIACIONES_PATH}",
                f"tools/{self.ABREVIACIONES_PATH}",
                f"../tools/{self.ABREVIACIONES_PATH}"
            ]
            
            for ruta in posibles_rutas:
                if os.path.exists(ruta):
                    print(f"Archivo de abreviaciones encontrado en: {ruta}")
                    df = pd.read_csv(ruta)
                    return dict(zip(df['abreviacion'], df['expansion']))
            
            print(f"Archivo de abreviaciones no encontrado en ninguna ubicación")
            print(f"Ubicaciones buscadas: {posibles_rutas}")
            print("Continuando sin diccionario de abreviaciones...")
            return {}
            
        except Exception as e:
            print(f"Error al cargar abreviaciones: {e}")
            return {}

    def limpiar_comentario(self, texto: str, contracciones: dict) -> str:
        if not isinstance(texto, str) or not texto.strip():
            return ""
        
        # Convertir a minúsculas
        texto = texto.lower()
        
        # Remover URLs
        texto = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', texto)
        
        # Remover menciones (@usuario)
        texto = re.sub(r'@\w+', '', texto)
        
        # Limpiar hashtags (mantener solo el texto)
        texto = re.sub(r'#(\w+)', r'\1', texto)
        
        # Mantener solo caracteres válidos
        texto = re.sub(r'[^\w\sáéíóúüñ¿¡]', ' ', texto)
        
        # Normalizar espacios
        texto = re.sub(r'\s+', ' ', texto)
        
        # Reducir repeticiones excesivas de caracteres
        texto = re.sub(r'(.)\1{2,}', r'\1\1', texto)
        
        # Remover números
        texto = re.sub(r'\b\d+\b', '', texto)
        
        # Expandir abreviaciones
        palabras = texto.split()
        palabras_limpias = []
        
        for palabra in palabras:
            if palabra in contracciones:
                palabras_limpias.append(contracciones[palabra])
            else:
                palabras_limpias.append(palabra)
        
        texto = ' '.join(palabras_limpias)
        
        # Remover palabras de una sola letra
        palabras = [palabra for palabra in texto.split() if len(palabra) > 1]
        texto = ' '.join(palabras)
        
        return texto.strip()

    def identificar_intencion(self, texto: str) -> dict:
        try:
            resultado = self.clasificador_intencion(texto, self.INTENCIONES)
            return {
                "intencion": resultado['labels'][0],
                "confianza_intencion": round(resultado['scores'][0], 2)
            }
        except Exception as e:
            print(f"Error al identificar intención: {e}")
            return {
                "intencion": "unknown",
                "confianza_intencion": 0.0
            }

    def analizar_sentimiento(self, texto: str) -> dict:
        try:
            resultado = self.clasificador_sentimiento(texto)[0]
            estrellas = int(resultado['label'][0])
            score = resultado['score']
            
            if estrellas <= 2:
                sentimiento = "negativo"
            elif estrellas == 3:
                sentimiento = "neutro"
            else:
                sentimiento = "positivo"
            
            return {
                "estrellas": estrellas,
                "confianza_sentimiento": round(score, 2),
                "sentimiento": sentimiento
            }
        except Exception as e:
            print(f"Error al analizar sentimiento: {e}")
            return {
                "estrellas": 3,
                "confianza_sentimiento": 0.0,
                "sentimiento": "neutro"
            }

    def analizar_comentario(self, texto: str, contracciones: dict) -> dict:
        if not texto or not texto.strip():
            return {
                "comentario_original": texto,
                "comentario_limpio": "",
                "estrellas": 3,
                "confianza_sentimiento": 0.0,
                "sentimiento": "neutro",
                "intencion": "unknown",
                "confianza_intencion": 0.0
            }
        
        texto_limpio = self.limpiar_comentario(texto, contracciones)
        
        # Si el texto limpio es muy corto, usar el original
        if len(texto_limpio.strip()) < 5:
            texto_limpio = texto
        
        # Análisis de sentimiento
        resultado_sentimiento = self.analizar_sentimiento(texto_limpio)
        
        # Análisis de intención
        resultado_intencion = self.identificar_intencion(texto_limpio)
        
        return {
            "comentario_original": texto,
            "comentario_limpio": texto_limpio,
            **resultado_sentimiento,
            **resultado_intencion
        }

    def generar_dataset_resumen(self, resultados: list) -> pd.DataFrame:
        df = pd.DataFrame(resultados)
        
        if df.empty:
            return pd.DataFrame(columns=['intencion', 'sentimiento_promedio', 'estrella_promedio'])
        
        df_agrupado = df.groupby('intencion').agg({
            'sentimiento': lambda x: x.mode()[0] if not x.mode().empty else 'neutro',
            'estrellas': 'mean'
        }).reset_index()
        
        df_agrupado.columns = ['intencion', 'sentimiento_promedio', 'estrella_promedio']
        df_agrupado['estrella_promedio'] = df_agrupado['estrella_promedio'].round(1)
        
        return df_agrupado
    
    def subir_archivo_s3(self, bucket_name: str, file_path: str, object_key: str) -> bool:
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                aws_session_token = os.getenv('AWS_SESSION_TOKEN'),
                region_name = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            )
            s3_client.upload_file(file_path, bucket_name, object_key)
            print(f"Archivo {file_path} subido a s3://{bucket_name}/{object_key}")
            return True
        except ClientError as e:
            print(f"Error al subir archivo a S3: {e}")
            return False

    def procesar_dataset(self) -> dict:
        try:
            # Verificar que existe el dataset
            if not os.path.exists(self.DATASET_PATH):
                return {
                    "success": False,
                    "message": f"Dataset no encontrado: {self.DATASET_PATH}"
                }
            
            # Cargar dataset
            df = pd.read_csv(self.DATASET_PATH)
            
            if df.empty or 'comentario' not in df.columns:
                return {
                    "success": False,
                    "message": "Dataset vacío o sin columna 'comentario'"
                }
            
            # Cargar abreviaciones
            contracciones = self.cargar_abreviaciones()
            
            # Procesar comentarios
            resultados = []
            total_comentarios = len(df)
            
            for idx, row in df.iterrows():
                comentario = str(row['comentario']).strip()
                if comentario and comentario != 'nan':
                    resultado = self.analizar_comentario(comentario, contracciones)
                    resultado['usuario_id'] = row.get('usuario_id', '')
                    resultados.append(resultado)
                    
                    # Log de progreso
                    if (idx + 1) % 10 == 0 or (idx + 1) == total_comentarios:
                        print(f"Procesados {idx + 1}/{total_comentarios} comentarios")
            
            if not resultados:
                return {
                    "success": False,
                    "message": "No se encontraron comentarios válidos para procesar"
                }
            
            # Generar dataset resumen
            df_resumen = self.generar_dataset_resumen(resultados)
            
            # Crear directorio de resultados si no existe
            os.makedirs(os.path.dirname(self.RESULTADO_PATH), exist_ok=True)
            
            # Guardar resultados
            df_resumen.to_csv(self.RESULTADO_PATH, index=False)

            bucket_name = "dataleak-nativox-integrador"  # Cambia por el bucket real
            object_key_resumen = os.path.basename(self.RESULTADO_PATH)
            object_key_detallado = os.path.basename(self.RESULTADO_PATH.replace('.csv', '_detallado.csv'))

            self.subir_archivo_s3(bucket_name, self.RESULTADO_PATH, object_key_resumen)
            self.subir_archivo_s3(bucket_name, self.RESULTADO_PATH.replace('.csv', '_detallado.csv'), object_key_detallado)
            
            # Guardar resultados detallados
            df_detallado = pd.DataFrame(resultados)
            df_detallado.to_csv(self.RESULTADO_PATH.replace('.csv', '_detallado.csv'), index=False)
            
            return {
                "success": True,
                "message": "Análisis completado exitosamente",
                "total_comentarios": len(resultados),
                "archivo_resumen": self.RESULTADO_PATH,
                "archivo_detallado": self.RESULTADO_PATH.replace('.csv', '_detallado.csv'),
                "resumen": df_resumen.to_dict('records')
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error durante el procesamiento: {str(e)}"
            }