"""
RAG Tool Package for OpenWeb UI Integration

This package provides a RAG (Retrieval-Augmented Generation) pipeline that can be
used as a custom tool in OpenWeb UI.

The main class is FocusedRAGPipeline which provides methods for:
- Initializing the RAG pipeline with documents
- Querying the pipeline with questions
- Getting responses with optional translation

Example usage:
    from rag_tool import FocusedRAGPipeline
    
    pipeline = FocusedRAGPipeline("./documents", "en")
    pipeline.initialize()
    result = pipeline.query("What is the capital of France?")
"""

from .pipeline import FocusedRAGPipeline
from .document_processor import load_documents, chunk_text, raptor_clustering
from .indexing import MultiRepresentationIndex
from .retrieval import RetrievalSystem
from .translation import OfflineTranslationSystem

__all__ = [
    "FocusedRAGPipeline",
    "load_documents",
    "chunk_text",
    "raptor_clustering",
    "MultiRepresentationIndex",
    "RetrievalSystem",
    "OfflineTranslationSystem"
]

__version__ = "0.1.0"
__author__ = "RAG Tool Development Team"