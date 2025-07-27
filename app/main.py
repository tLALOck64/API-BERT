from fastapi import FastAPI
from app.api.data_router import router as data_router

app = FastAPI()

app.include_router(data_router, prefix="/data", tags=["Data"])