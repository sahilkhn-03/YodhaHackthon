from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """
    Application configuration settings.
    Reads from environment variables or .env file.
    """
    
    # Supabase Database Configuration
    # Format: postgresql://postgres.[ref]:[password]@[host].supabase.com:5432/postgres
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/neurobalance_db"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Application Settings
    APP_NAME: str = "NeuroBalance AI Backend"
    DEBUG: bool = True
    
    # CORS Origins (comma-separated in .env)
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8501"
    
    class Config:
        # Tell Pydantic to read from .env file in the backend directory
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Create cached settings instance.
    Using lru_cache ensures we only create one Settings object.
    """
    return Settings()


# Export a ready-to-use settings object
settings = get_settings()
