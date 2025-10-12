from pydantic import BaseModel

class CorrectionRequest(BaseModel):
    frase: str

class CorrectionResponse(BaseModel):
    frase_corregida: str