import asyncio
import warnings

import aiohttp
from inscriptis import get_text
from Triad_KGC.code.config import settings

async def fetch(session, url):
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            html = await response.text()
            text = get_text(html)
            return text if len(text) >= 200 else None
    except Exception:
        return None

async def strong_fetch(session, url):
    headers = {
        'Authorization': f'Bearer {settings.SPIDER_API_KEY}',
        'Content-Type': 'application/json',
    }
    json_data = {"return_format": "markdown", "url": url}
    try:
        async with session.post('https://api.spider.cloud/scrape', headers=headers, json=json_data) as response:
            response.raise_for_status()
            result = await response.json()
            text = get_text(result['content']) if result else None
            return text if text else None
    except Exception:
        return None

async def crawl_single_page(session, url):
    text = await fetch(session, url)
    if text is not None:
        return text
    return await strong_fetch(session, url)

async def crawl_pages(urls):
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*(crawl_single_page(session, url) for url in urls))
        if all(r for r in results):
            return results
        else:
            # Filter out None items
            results = [r for r in results if r]
            if len(results) == 0:
                # Emit a visible warning using the warnings module
                warnings.warn("All web pages are inaccessible or too short, returning empty list", UserWarning)
            return results
