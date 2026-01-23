"""
Application configuration management
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "text-embedding-3-large"
    openai_chat_model: str = "gpt-4o-mini"
    
    # Oracle Database Configuration
    oracle_username: str = os.getenv("ORACLE_USERNAME", "")
    oracle_password: str = os.getenv("ORACLE_PASSWORD", "")
    oracle_dsn: str = os.getenv("ORACLE_DSN", "")
    oracle_wallet_path: str = os.getenv("ORACLE_WALLET_PATH", "")
    
    # Application Settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_upload_entries: int = 5000
    embedding_batch_size: int = 1000
    top_k_default: int = 3
    
    # Database Pool Settings
    db_pool_min: int = 2
    db_pool_max: int = 10
    db_pool_increment: int = 1


settings = Settings()
