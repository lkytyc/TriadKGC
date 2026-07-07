### Embedding Model API ###
### Note: This module is called by related_retrieve.py to obtain vector representations from the embedding model ###

from openai import AsyncOpenAI
from Triad_KGC.code.config import settings
from typing import List


async def get_vector(
        query: str,
        model: str = settings.EMBEDDING_MODEL,
        api_key: str = settings.API_KEY,
        base_url: str = settings.BASE_URL,
) -> List[float]:
    """
    API interface for retrieving vector embeddings.

    :param query: The input text to embed.
    :param model: Name of the embedding model.
    :param api_key: API key for authentication.
    :param base_url: Base URL of the API endpoint.
    """

    # Initialize the asynchronous embedding client
    async_embedding_client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # Request the embedding for the given input
    completion = await async_embedding_client.embeddings.create(
        model=model,
        input=[query]
    )

    # Return the embedding vector
    return completion.data[0].embedding
