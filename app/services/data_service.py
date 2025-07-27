import csv
import re
import os
from app.repositories.external_api import ExternalAPI
from app.services.sentiment_analysis_service import SentimentAnalysisService

class DataService:
    
    DATASET_PATH = "datasets/comentarios.csv"

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
        response = ExternalAPI.fetch_data()
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