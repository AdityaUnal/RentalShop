from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/bike_rental")
    
    # LLM settings
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "google/flan-t5-base")
    
    # Application settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    API_V1_STR: str = "/api/v1"
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    
    class Config:
        case_sensitive = True

settings = Settings() 