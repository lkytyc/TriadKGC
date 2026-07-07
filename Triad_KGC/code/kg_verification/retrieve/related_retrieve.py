import numpy as np
import asyncio
from typing import List, Tuple
from Triad_KGC.code.kg_verification.retrieve.api import get_vector


def cosine_similarity(
        vector_a: List[float],
        vector_b: List[float],
):
    """
    Calculate the cosine similarity between two vectors.
    :param vector_a: First vector
    :param vector_b: Second vector
    :return: Cosine similarity value between -1 and 1, or None if invalid input
    """
    a = np.array(vector_a, dtype=float)
    b = np.array(vector_b, dtype=float)

    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    if not np.any(a) or not np.any(b):
        raise ValueError("Zero vectors are not allowed")

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    return np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0)


class RelatedRetriever:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    async def retrieve_related_blocks(
        self,
        text_blocks: List[str],
        source_entity: str,
        relation: str,
        target_entity: str,
    ) -> List[Tuple[str, float]]:
        """
        Asynchronous method for relevance-based retrieval.
        Finds the top_k most relevant text blocks from text_blocks for the given triple.
        :param text_blocks: List of text blocks
        :param source_entity: Source entity
        :param relation: Relation
        :param target_entity: Target entity
        :return: List of relevant text blocks with similarity scores
        """
        if not text_blocks:
            return []

        query = f"{source_entity} {relation} {target_entity}"
        query_vector = await get_vector(query)
        if not query_vector:
            raise ValueError("get_vector returned empty vector for the query")

        scored_blocks = []
        for block in text_blocks:
            block_vector = await get_vector(block)
            if not block_vector:
                continue
            try:
                similarity = cosine_similarity(query_vector, block_vector)
                # Add to result list if similarity exceeds threshold 0.7
                if similarity > 0.7:
                    scored_blocks.append((block, similarity))
            except ValueError:
                continue

        top_blocks = sorted(scored_blocks, key=lambda x: x[1], reverse=True)[:self.top_k]
        return top_blocks


# ======================== ✅ Test Code ========================= #

if __name__ == "__main__":

    # ✅ Run async test
    async def test_related_retrieve():
        retriever = RelatedRetriever(top_k=2)
        blocks = [
            "Alice works at OpenAI in 2020.",
            "Bob likes sports.",
            "Charlie and Alice are coworkers at OpenAI.",
            "OpenAI is a research company.",
            "Alice was hired in 2020 by OpenAI."
        ]

        print("\n🧪 Running async test case...\n")
        results = await retriever.retrieve_related_blocks(blocks, "Alice", "works_at", "OpenAI")

        print("Top related blocks:")
        for text, score in results:
            print(f" - Score: {score:.4f} | Text: {text}")

        assert len(results) == 2
        assert all(0 <= score <= 1 for _, score in results)

    asyncio.run(test_related_retrieve())
