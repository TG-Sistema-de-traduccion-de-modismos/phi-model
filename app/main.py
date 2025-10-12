from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.infrastructure.routes import router
from app.core.logging_config import setup_logging
from app.core.config import settings

setup_logging()

app = FastAPI(title="Phi Model Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
async def root():
    return {
        "service": "Phi Model",
        "model": settings.model_name,
        "status": "running"
    }