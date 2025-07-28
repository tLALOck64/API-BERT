import csv
import re
import os
import io
import pandas as pd
from datetime import datetime, timezone
from app.repositories.external_api import ExternalAPI
from app.services.sentiment_analysis_service import SentimentAnalysisService
from app.services.s3_service import S3Service

class DataService:
    
    DATASET_PATH = "datasets/comentarios.csv"
    BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        cleaned = re.sub(r'[¡!¿?@#$%^&*()_+={}\[\]|\\:";\'<>,./?~`]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned

    @staticmethod
    def get_data_as_csv() -> None:
        response = ExternalAPI.fetch_data("micro-user/api_user/usuarios/comentarios")
        data = response.get("comentarios", [])

        os.makedirs(os.path.dirname(DataService.DATASET_PATH), exist_ok=True)

        if os.path.exists(DataService.DATASET_PATH):
            os.remove(DataService.DATASET_PATH)

        if not data:
            with open(DataService.DATASET_PATH, "w", encoding="utf-8-sig", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["usuario_id", "comentario"])
                writer.writeheader()
            return
        
        rows = [
            {
                "usuario_id": DataService.clean_text(str(item["_usuarioId"])),
                "comentario": DataService.clean_text(item["_mensaje"])
            }
            for item in data
        ]
        
        with open(DataService.DATASET_PATH, "w", encoding="utf-8-sig", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["usuario_id", "comentario"])
            writer.writeheader()
            writer.writerows(rows)
        
        print("Iniciano análisis")
        sentiment_service = SentimentAnalysisService()
        result_analisis = sentiment_service.procesar_dataset()

        if result_analisis["success"]:
            print(f"Análisis completado: {result_analisis['message']}")
            print(f"Total comentarios procesados: {result_analisis['total_comentarios']}")
            print(f"Archivo resumen: {result_analisis['archivo_resumen']}")
            print(f"Archivo detallado: {result_analisis['archivo_detallado']}")
        else: 
            print(f"Error en análisis:{result_analisis['message']}")

        return result_analisis
    
    @staticmethod
    def get_data_users_data_csv() -> dict:
        try:
            response = ExternalAPI.fetch_data("micro-learning/api_learning/userResponse/analiticas_llm")
        
            df = pd.DataFrame(response)

            
            hoy = datetime.now(timezone.utc)
            def calcular_abandono(ultima_fecha):
                try:
                    fecha = pd.to_datetime(ultima_fecha)
                    dias = (hoy - fecha).days
                    return 1 if dias > 14 else 0
                except Exception:
                    return 0
            df['abandono'] = df['ultima_fecha_de_actividad'].apply(calcular_abandono)

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding="utf-8")
            csv_content = csv_buffer.getvalue().encode('utf-8-sig')
            csv_buffer.close()

            object_key = "usuarios.csv"
            success = S3Service.subir_archivo(
                DataService.BUCKET_NAME, 
                csv_content, 
                object_key
            )

            if success:
                return {
                    "success": True,
                    "message": "Archivo CSV subido exitosamente a S3",
                    "bucket": DataService.BUCKET_NAME,
                    "key": object_key,
                    "url": f"s3://{DataService.BUCKET_NAME}/{object_key}"
                }
            else:
                return {
                    "success": False,
                    "message": "Error al subir archivo CSV a S3"
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error al procesar datos: {str(e)}"
            }