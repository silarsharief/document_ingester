import os
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # --- Project Info ---
    PROJECT_NAME: str = "QuickSight AI"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # --- API Keys (Required) ---
    # It will automatically look for these in your .env file
    GOOGLE_API_KEY: str 
    
    # --- Paths ---
    # This automatically finds the root folder, no matter where you run it from
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TEMP_DIR: str = os.path.join(BASE_DIR, "data", "temp_crops")
    
    # --- M1 Optimization Limits ---
    MAX_CONCURRENT_PAGES: int = 3  # Keeps RAM usage safe
    YOLO_MODEL_PATH: str = "keremberke/yolov8m-doclaynet" 
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()

# Global settings instance
settings = get_settings()

# Ensure temp directory exists at startup
os.makedirs(settings.TEMP_DIR, exist_ok=True)