# app/config.py

# ---------- IMPORTS ----------
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------- SETTINGS CLASS ----------
class Settings(BaseSettings):
    nvidia_api_key: str
    nvidia_base_url: str
    llm_model: str
    embedding_model: str
    collection_name: str
    chroma_path: str
    pinecone_api_key: str
    pinecone_index_name: str

    model_config = SettingsConfigDict(env_file=".env")

# ---------- SINGLE INSTANCE ----------
settings = Settings()