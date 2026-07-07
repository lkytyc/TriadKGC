import aiohttp
import json
from Triad_KGC.code.config import settings

# Asynchronously sends a POST request to the Spider API for searching a query.
async def spider_search(query: str, limit: int = 3):
    headers = {
        'Authorization': f'Bearer {settings.SPIDER_API_KEY}',
        'Content-Type': 'application/json',
    }
    json_data = {
        "search": query,
        "search_limit": limit,
        "limit": limit,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            'https://api.spider.cloud/search', headers=headers, json=json_data
        ) as response:
            # Raise an error for bad HTTP responses
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return await response.json()
            else:
                # Try parsing manually if not declared as JSON
                raw = await response.text()
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    raise RuntimeError("Spider API returned non-JSON content that couldn't be parsed.")

# Test the function
if __name__ == "__main__":
    import asyncio
    query = "openai`s ceo"
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(spider_search(query))
    print(result)
