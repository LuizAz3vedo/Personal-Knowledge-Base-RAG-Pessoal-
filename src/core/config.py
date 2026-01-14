"""Configuracoes centralizadas do projeto."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuracoes do Personal Knowledge Base."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ================================================================
    # Diretorios
    # ================================================================
    obsidian_vault_path: Path = Field(
        default=Path("./data/obsidian"),
        description="Caminho para o vault do Obsidian",
    )
    pdf_directory: Path = Field(
        default=Path("./data/pdfs"),
        description="Diretorio com PDFs para indexar",
    )
    bookmarks_directory: Path = Field(
        default=Path("./data/bookmarks"),
        description="Diretorio com bookmarks exportados",
    )

    # ================================================================
    # ChromaDB
    # ================================================================
    chroma_db_path: Path = Field(
        default=Path("./chroma_db"),
        description="Caminho para persistencia do ChromaDB",
    )
    chroma_collection_name: str = Field(
        default="personal_knowledge",
        description="Nome da collection no ChromaDB",
    )

    # ================================================================
    # Ollama
    # ================================================================
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="URL base do servidor Ollama",
    )
    ollama_llm_model: str = Field(
        default="qwen2.5:14b",
        description="Modelo LLM do Ollama",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Modelo de embeddings do Ollama",
    )

    # ================================================================
    # Chunking
    # ================================================================
    chunk_size: int = Field(
        default=800,
        ge=100,
        le=2000,
        description="Tamanho dos chunks em caracteres",
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Sobreposicao entre chunks",
    )

    # ================================================================
    # Retrieval
    # ================================================================
    top_k_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Numero de resultados a retornar",
    )
    similarity_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold minimo de similaridade",
    )

    # ================================================================
    # Logging
    # ================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Nivel de logging",
    )


# Instancia global de configuracoes
settings = Settings()
