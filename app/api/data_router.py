from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
from app.services.data_service import DataService
from io import StringIO

router = APIRouter()

@router.get("/export")
def export_data():
    try:
        result = DataService.get_data_as_csv()

        if result["success"]:
            return {
                "status": "success",
                "message": "Dataset creado y análisis completado",
                "data": result
            }
        else:
            return {
                "status": "error",
                "message": result["message"]
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error interno: {str(e)}"
        }
    
@router.get("/users")
def export_users_csv():
    try:
        result = DataService.get_data_users_data_csv()

        if result["success"]:
            return {
                "status": "success",
                "message": "Dataset creado",
                "data": result
            }
        else:
            return{
                "status": "error",
                "message": result["message"],
                "data": result
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error interno: {str(e)}"
        }