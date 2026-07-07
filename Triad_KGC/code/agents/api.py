### LLM API Request and Response Handler ###
from openai import AsyncOpenAI
from Triad_KGC.code.config import settings


async def get_response(
        query: str,
        api_key: str = settings.API_KEY,
        base_url: str = settings.BASE_URL,
        model: str = settings.MODEL,
        **kwargs,
) -> str:
    """
    Chat-based LLM API interface.

    :param query: User input or question to send to the model
    :param api_key: API key for authentication
    :param base_url: Base URL for the API endpoint
    :param model: Model name to be used
    :return: The response content from the model
    """
    # Initialize asynchronous OpenAI client
    async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=1200)

    # Call the chat completion API
    completion = await async_client.chat.completions.create(
        temperature=0,
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful KGC (Knowledge Graph Completion) agent."
            },
            {
                "role": "user",
                "content": query
            },
        ]
    )
    return completion.choices[0].message.content


# ========================= ✅ Test Block ========================= #

if __name__ == "__main__":
    import asyncio
    query = (
        "Please complete the following triples:\n"
        "Triple 1:\n"
        "subject: 1\n"
        "predicate: 2\n"
        "object: 3\n"
        "Triple 2:\n"
        "subject: 4\n"
        "predicate: 5\n"
        "object: 6"
    )

    result = asyncio.run(
        get_response(
            query=query,
            api_key=settings.API_KEY,
            base_url=settings.BASE_URL,
            model="gemini-2.5-pro"  # Replace with your actual model if needed
        )
    )
    print(result)
