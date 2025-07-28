import pandas as pd
import re
import boto3
from app.services.s3_service import S3Service
from botocore.exceptions import NoCredentialsError
from transformers import pipeline
import os
from dotenv import load_dotenv
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import lru_cache
import torch

load_dotenv()

class SentimentAnalysisService:
    
    DATASET_PATH = "datasets/comentarios.csv"
    ABREVIACIONES_PATH = "app/tools/diccionario_abrevaciones.csv"
    RESULTADO_PATH = "datasets/intenciones_sentimientos.csv"
    HF_TOKEN = os.getenv('HUGGIN_FACE_TOKEN')
    
    MODELO_SENTIMIENTO_ID = "nlptown/bert-base-multilingual-uncased-sentiment"
    MODELO_INTENCION_ID = "facebook/bart-large-mnli"
    INTENCIONES = ["appreciation", "complaint", "request", "bug", "confusion"]
    

    BATCH_SIZE = 32  
    MAX_WORKERS = min(4, multiprocessing.cpu_count()) 
    
    def __init__(self):
        self.clasificador_sentimiento = None
        self.clasificador_intencion = None
        self.contracciones = {}
        self._init_models()
        self._load_contracciones()
    
    def _init_models(self):
        try:
            # Configuración optimizada para GPU si está disponible
            device = 0 if torch.cuda.is_available() else -1
            
            self.clasificador_sentimiento = pipeline(
                "sentiment-analysis", 
                model=self.MODELO_SENTIMIENTO_ID, 
                token=self.HF_TOKEN,
                device=device,
                batch_size=self.BATCH_SIZE,
                return_all_scores=False  # Solo el mejor resultado
            )
            
            self.clasificador_intencion = pipeline(
                "zero-shot-classification", 
                model=self.MODELO_INTENCION_ID, 
                token=self.HF_TOKEN,
                device=device,
                batch_size=self.BATCH_SIZE
            )
            
            print(f"Modelos cargados en dispositivo: {'GPU' if device == 0 else 'CPU'}")
            
        except Exception as e:
            print(f"Error al inicializar modelos: {e}")
            try:
                print("Intentando cargar modelos sin token...")
                device = 0 if torch.cuda.is_available() else -1
                
                self.clasificador_sentimiento = pipeline(
                    "sentiment-analysis", 
                    model=self.MODELO_SENTIMIENTO_ID,
                    device=device,
                    batch_size=self.BATCH_SIZE,
                    return_all_scores=False
                )
                
                self.clasificador_intencion = pipeline(
                    "zero-shot-classification", 
                    model=self.MODELO_INTENCION_ID,
                    device=device,
                    batch_size=self.BATCH_SIZE
                )
                
                print("Modelos cargados exitosamente sin token")
            except Exception as e2:
                print(f"Error al cargar modelos sin token: {e2}")
                raise

    def _load_contracciones(self):
        self.contracciones = self.cargar_abreviaciones()

    def cargar_abreviaciones(self) -> dict:

        try:
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
            
            print("Continuando sin diccionario de abreviaciones...")
            return {}
            
        except Exception as e:
            print(f"Error al cargar abreviaciones: {e}")
            return {}

    @lru_cache(maxsize=1000)
    def limpiar_comentario_cached(self, texto: str) -> str:
        return self._limpiar_comentario_interno(texto)
    
    def _limpiar_comentario_interno(self, texto: str) -> str:
        if not isinstance(texto, str) or not texto.strip():
            return ""
        
        # Operaciones en orden de eficiencia
        texto = texto.lower()
        
        # Compilar regex una sola vez (esto debería estar en __init__ idealmente)
        if not hasattr(self, '_compiled_regexes'):
            self._compiled_regexes = {
                'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
                'mentions': re.compile(r'@\w+'),
                'hashtags': re.compile(r'#(\w+)'),
                'chars': re.compile(r'[^\w\sáéíóúüñ¿¡]'),
                'spaces': re.compile(r'\s+'),
                'repeats': re.compile(r'(.)\1{2,}'),
                'numbers': re.compile(r'\b\d+\b')
            }
        
        # Aplicar regex compiladas
        texto = self._compiled_regexes['urls'].sub('', texto)
        texto = self._compiled_regexes['mentions'].sub('', texto)
        texto = self._compiled_regexes['hashtags'].sub(r'\1', texto)
        texto = self._compiled_regexes['chars'].sub(' ', texto)
        texto = self._compiled_regexes['spaces'].sub(' ', texto)
        texto = self._compiled_regexes['repeats'].sub(r'\1\1', texto)
        texto = self._compiled_regexes['numbers'].sub('', texto)
        
        # Expandir abreviaciones de manera eficiente
        if self.contracciones:
            palabras = texto.split()
            palabras_limpias = [self.contracciones.get(palabra, palabra) for palabra in palabras]
            texto = ' '.join(palabra for palabra in palabras_limpias if len(palabra) > 1)
        else:
            palabras = texto.split()
            texto = ' '.join(palabra for palabra in palabras if len(palabra) > 1)
        
        return texto.strip()

    def limpiar_comentario(self, texto: str, contracciones: dict = None) -> str:
        return self.limpiar_comentario_cached(texto)

    def analizar_lote_sentimientos(self, textos: list) -> list:
        try:
            if not textos:
                return []
            
            # Filtrar textos válidos
            textos_validos = [t for t in textos if t and t.strip()]
            if not textos_validos:
                return [{"estrellas": 3, "confianza_sentimiento": 0.0, "sentimiento": "neutro"} for _ in textos]
            
            # Análisis en lote
            resultados = self.clasificador_sentimiento(textos_validos)
            
            # Procesar resultados
            resultados_procesados = []
            idx_valido = 0
            
            for texto_original in textos:
                if texto_original and texto_original.strip():
                    resultado = resultados[idx_valido]
                    estrellas = int(resultado['label'][0])
                    score = resultado['score']
                    
                    if estrellas <= 2:
                        sentimiento = "negativo"
                    elif estrellas == 3:
                        sentimiento = "neutro"
                    else:
                        sentimiento = "positivo"
                    
                    resultados_procesados.append({
                        "estrellas": estrellas,
                        "confianza_sentimiento": round(score, 2),
                        "sentimiento": sentimiento
                    })
                    idx_valido += 1
                else:
                    resultados_procesados.append({
                        "estrellas": 3,
                        "confianza_sentimiento": 0.0,
                        "sentimiento": "neutro"
                    })
            
            return resultados_procesados
            
        except Exception as e:
            print(f"Error en análisis de lote de sentimientos: {e}")
            return [{"estrellas": 3, "confianza_sentimiento": 0.0, "sentimiento": "neutro"} for _ in textos]

    def analizar_lote_intenciones(self, textos: list) -> list:
        try:
            if not textos:
                return []
            
            textos_validos = [t for t in textos if t and t.strip()]
            if not textos_validos:
                return [{"intencion": "unknown", "confianza_intencion": 0.0} for _ in textos]
            
            # Análisis en lote
            resultados = []
            for texto in textos_validos:
                resultado = self.clasificador_intencion(texto, self.INTENCIONES)
                resultados.append(resultado)
            
            # Procesar resultados
            resultados_procesados = []
            idx_valido = 0
            
            for texto_original in textos:
                if texto_original and texto_original.strip():
                    resultado = resultados[idx_valido]
                    resultados_procesados.append({
                        "intencion": resultado['labels'][0],
                        "confianza_intencion": round(resultado['scores'][0], 2)
                    })
                    idx_valido += 1
                else:
                    resultados_procesados.append({
                        "intencion": "unknown",
                        "confianza_intencion": 0.0
                    })
            
            return resultados_procesados
            
        except Exception as e:
            print(f"Error en análisis de lote de intenciones: {e}")
            return [{"intencion": "unknown", "confianza_intencion": 0.0} for _ in textos]

    def procesar_lote_comentarios(self, comentarios_data: list) -> list:
        comentarios_originales = []
        comentarios_limpios = []
        usuarios_ids = []
        
        for item in comentarios_data:
            comentario_original = str(item['comentario']).strip()
            comentarios_originales.append(comentario_original)
            usuarios_ids.append(item.get('usuario_id', ''))
            
            if comentario_original and comentario_original != 'nan':
                texto_limpio = self.limpiar_comentario(comentario_original)
                if len(texto_limpio.strip()) < 5:
                    texto_limpio = comentario_original
                comentarios_limpios.append(texto_limpio)
            else:
                comentarios_limpios.append("")
        
        # Análisis en lotes
        resultados_sentimiento = self.analizar_lote_sentimientos(comentarios_limpios)
        resultados_intencion = self.analizar_lote_intenciones(comentarios_limpios)
        
        # Combinar resultados
        resultados_finales = []
        for i, (original, limpio, usuario_id) in enumerate(zip(comentarios_originales, comentarios_limpios, usuarios_ids)):
            resultado = {
                "comentario_original": original,
                "comentario_limpio": limpio,
                "usuario_id": usuario_id,
                **resultados_sentimiento[i],
                **resultados_intencion[i]
            }
            resultados_finales.append(resultado)
        
        return resultados_finales

    def generar_dataset_resumen(self, resultados: list) -> pd.DataFrame:
        if not resultados:
            return pd.DataFrame(columns=['intencion', 'sentimiento_promedio', 'estrella_promedio'])
        
        df = pd.DataFrame(resultados)
        
        # Usar groupby más eficientemente
        df_agrupado = df.groupby('intencion').agg({
            'sentimiento': lambda x: x.mode().iloc[0] if not x.mode().empty else 'neutro',
            'estrellas': 'mean'
        }).reset_index()
        
        df_agrupado.columns = ['intencion', 'sentimiento_promedio', 'estrella_promedio']
        df_agrupado['estrella_promedio'] = df_agrupado['estrella_promedio'].round(1)
        
        return df_agrupado

    def procesar_dataset(self) -> dict:
        try:
            # Verificaciones iniciales
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
            
            # Filtrar comentarios válidos
            df_valido = df[df['comentario'].notna() & (df['comentario'].astype(str).str.strip() != '')]
            
            if df_valido.empty:
                return {
                    "success": False,
                    "message": "No se encontraron comentarios válidos para procesar"
                }
            
            print(f"Procesando {len(df_valido)} comentarios válidos en lotes de {self.BATCH_SIZE}")
            
            # Procesar en lotes
            resultados = []
            total_comentarios = len(df_valido)
            
            for i in range(0, total_comentarios, self.BATCH_SIZE):
                lote = df_valido.iloc[i:i+self.BATCH_SIZE]
                comentarios_data = lote.to_dict('records')
                
                resultados_lote = self.procesar_lote_comentarios(comentarios_data)
                resultados.extend(resultados_lote)
                
                # Log de progreso
                procesados = min(i + self.BATCH_SIZE, total_comentarios)
                print(f"Procesados {procesados}/{total_comentarios} comentarios")
            
            # Generar dataset resumen
            df_resumen = self.generar_dataset_resumen(resultados)

            # Crear directorio de resultados
            os.makedirs(os.path.dirname(self.RESULTADO_PATH), exist_ok=True)

            # Guardar resultados
            df_resumen.to_csv(self.RESULTADO_PATH, index=False)

            df_detallado = pd.DataFrame(resultados)
            path_detallado = self.RESULTADO_PATH.replace('.csv', '_detallado.csv')
            df_detallado.to_csv(path_detallado, index=False)

            # Subir a S3
            BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
            if BUCKET_NAME:
                object_key_resumen = os.path.basename(self.RESULTADO_PATH)
                object_key_detallado = os.path.basename(path_detallado)

                S3Service.subir_archivo(BUCKET_NAME, self.RESULTADO_PATH, object_key_resumen)
                S3Service.subir_archivo(BUCKET_NAME, path_detallado, object_key_detallado)

            return {
                "success": True,
                "message": "Análisis completado exitosamente",
                "total_comentarios": len(resultados),
                "archivo_resumen": self.RESULTADO_PATH,
                "archivo_detallado": path_detallado,
                "resumen": df_resumen.to_dict('records')
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error durante el procesamiento: {str(e)}"
            }

    # Métodos de compatibilidad para mantener la API existente
    def analizar_comentario(self, texto: str, contracciones: dict = None) -> dict:
        resultado = self.procesar_lote_comentarios([{'comentario': texto, 'usuario_id': ''}])
        return resultado[0] if resultado else {
            "comentario_original": texto,
            "comentario_limpio": "",
            "estrellas": 3,
            "confianza_sentimiento": 0.0,
            "sentimiento": "neutro",
            "intencion": "unknown",
            "confianza_intencion": 0.0
        }

    def identificar_intencion(self, texto: str) -> dict:
        resultado = self.analizar_lote_intenciones([texto])
        return resultado[0] if resultado else {"intencion": "unknown", "confianza_intencion": 0.0}

    def analizar_sentimiento(self, texto: str) -> dict:
        resultado = self.analizar_lote_sentimientos([texto])
        return resultado[0] if resultado else {"estrellas": 3, "confianza_sentimiento": 0.0, "sentimiento": "neutro"}