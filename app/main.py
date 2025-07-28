from fastapi import FastAPI
from app.api.data_router import router as data_router

app = FastAPI()

app.include_router(data_router, prefix="/data", tags=["Data"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)