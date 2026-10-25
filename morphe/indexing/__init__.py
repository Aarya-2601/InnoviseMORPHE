"""MORPHE indexing and search package."""
from morphe.indexing.embeddings import EmbeddingService
from morphe.indexing.faiss_index import ModuleFaissIndex

__all__ = ["EmbeddingService", "ModuleFaissIndex"]

