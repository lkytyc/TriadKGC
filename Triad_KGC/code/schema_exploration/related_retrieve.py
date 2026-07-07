import numpy as np
from openai import AsyncOpenAI
from Triad_KGC.code.config import settings
from typing import List, Dict

# Initialize OpenAI client with API settings
openai_client = AsyncOpenAI(
    api_key=settings.API_KEY,
    base_url=settings.BASE_URL
)


async def batch_get_vectors(words: List[str], model: str = settings.EMBEDDING_MODEL) -> Dict[str, List[float]]:
    """
    Get vector embeddings for a batch of words in a single API call

    Args:
        words: List of words/phrases to get embeddings for
        model: Name of the embedding model to use (default from settings)

    Returns:
        Dictionary mapping each word to its corresponding embedding vector
    """
    # Call OpenAI embeddings API asynchronously
    response = await openai_client.embeddings.create(
        model=model,
        input=words
    )

    # Create dictionary pairing each word with its embedding
    return {word: emb for word, emb in zip(words, [x.embedding for x in response.data])}


def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Cosine similarity measures the angle between two vectors, ranging from -1 (opposite)
    to 1 (identical), with 0 indicating orthogonality.

    Args:
        vector_a: First vector for comparison
        vector_b: Second vector for comparison

    Returns:
        Cosine similarity value between -1.0 and 1.0

    Raises:
        ValueError: If vectors have different lengths or are zero vectors
    """
    # Convert input lists to numpy arrays for numerical operations
    a = np.array(vector_a, dtype=float)
    b = np.array(vector_b, dtype=float)

    # Validate input vectors
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    if not np.any(a) or not np.any(b):
        raise ValueError("Zero vectors are not allowed")

    # Calculate dot product and vector norms
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Compute and clamp the cosine similarity to ensure valid range
    return np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0)