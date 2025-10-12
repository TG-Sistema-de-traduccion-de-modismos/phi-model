from fastapi import APIRouter, HTTPException
from app.domain.models import CorrectionRequest, CorrectionResponse
from app.application.phi_processor import PhiProcessor
from app.core.logging_config import logger

router = APIRouter()
processor = PhiProcessor()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(processor.model.device)
    }


@router.post("/correct", response_model=CorrectionResponse)
async def corregir_frase(request: CorrectionRequest):
    try:
        logger.info(f"Recibida solicitud de corrección: {request.frase}")
        frase_corregida = processor.corregir_gramatica(request.frase)
        logger.info(f"Frase corregida: {frase_corregida}")
        return CorrectionResponse(frase_corregida=frase_corregida)
    except Exception as e:
        logger.error(f"Error al corregir frase: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))