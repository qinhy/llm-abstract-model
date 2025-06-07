import math
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, ConfigDict
import numpy as np

class AbstractEmbedding(BaseModel):
    """Abstract base class for embedding model implementations.
    
    This class provides a common interface for interacting with various embedding
    models, handling model parameters, and generating embeddings.
    
    Attributes:
        vendor_id: Identifier for the embedding vendor, 'auto' for automatic detection
        embedding_model_name: Name of the specific embedding model to use
        embedding_dim: Dimensionality of the embeddings
        normalize_embeddings: Whether to normalize the embeddings to unit vectors
        max_input_length: Optional limit on input length
        pooling_strategy: Pooling strategy for sentence embeddings
        distance_metric: Metric for comparing embeddings
    """
    vendor_id: str = 'auto'                # Vendor identifier (e.g., OpenAI, Google)
    embedding_model_name: str              # Model name (e.g., "text-embedding-3-small")
    embedding_dim: int                     # Dimensionality of the embeddings, e.g., 768 or 1024
    normalize_embeddings: bool = True      # Whether to normalize the embeddings to unit vectors
    
    max_input_length: Optional[int] = None     # Optional limit on input length (e.g., max tokens or chars)
    pooling_strategy: Optional[str] = 'mean'   # Pooling strategy if working with sentence embeddings (e.g., "mean", "max")
    distance_metric: Optional[str] = 'cosine'  # Metric for comparing embeddings ("cosine", "euclidean", etc.)
    embedding_context: Optional[str] = None # Optional context or description to customize embedding generation
    additional_features: Optional[List[str]] = None  # Additional features for embeddings, e.g., "entity", "syntax"    

    def __call__(self, input_text: str) -> np.ndarray:
        """Generate embedding for the given text."""
        return self.generate_embedding(input_text)

    def generate_embedding(self, input_text: str) -> np.ndarray:
        """Generate embedding for the given text."""
        raise NotImplementedError
        
    def similarity_score(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity score between two embeddings."""
        if embedding1.shape != embedding2.shape:
            raise ValueError("Embeddings must have the same dimensions")

        metrics = {
            "cosine": self._cosine_similarity,
            "euclidean": self._euclidean_distance
        }

        if self.distance_metric not in metrics:
            raise ValueError(f"Unsupported distance metric. Choose from: {list(metrics.keys())}")

        return metrics[self.distance_metric](embedding1, embedding2)

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding vector to unit length."""
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

    def _euclidean_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate Euclidean distance between two embeddings."""
        return np.linalg.norm(embedding1 - embedding2)

    model_config = ConfigDict(arbitrary_types_allowed=True)
