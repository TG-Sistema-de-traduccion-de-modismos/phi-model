from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    port: int = 8006
    host: str = "0.0.0.0"
    
    class Config:
        env_file = ".env"

settings = Settings()