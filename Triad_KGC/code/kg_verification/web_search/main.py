import warnings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from Triad_KGC.code.kg_verification.web_search.spider_search import spider_search
from Triad_KGC.code.kg_verification.web_search.crawling import crawl_pages
from typing import List


def triple_to_query(
        source_entity: str,
        relation: str,
        target_entity: str,
) -> str:
    """
    Convert a triple (source, relation, target) into a search query.
    :param source_entity: the source entity
    :param relation: the relation between the source and target
    :param target_entity: the target entity
    :return:
    """
    return f"{source_entity} AND {relation} AND {target_entity}"


async def get_web_context_blocks(
        source_entity: str,
        relation: str,
        target_entity: str,
        chunk_size=1000,
        overlap=200,
) -> List[str]:
    """
    Get web context blocks for a given triple (source, relation, target).
    :param source_entity: the source entity
    :param relation: the relation between the source and target
    :param target_entity: the target entity
    :param chunk_size: the size of each chunk
    :param overlap: the overlap between chunks
    :return: the web context blocks
    """
    query = triple_to_query(
        source_entity=source_entity,
        relation=relation,
        target_entity=target_entity,
    )
    search_result = await spider_search(query)
    urls = [r['url'] for r in search_result['content']]

    # Select top 3 URLs; if fewer than 3, issue a warning about insufficient context
    # If no URLs are found, return an empty list and warn
    if len(urls) < 3:
        warnings.warn("Fewer than 3 search results; context may be insufficient", UserWarning)
    else:
        urls = urls[:3]
    if len(urls) == 0:
        warnings.warn("Search results are empty; returning empty list", UserWarning)
        return []

    contents = await crawl_pages(urls)
    print(f"Fetched {len(contents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = text_splitter.create_documents(contents)
    return [chunk.page_content for chunk in chunks]


if __name__ == "__main__":
    import asyncio

    source = "United States"
    relation = "capital"
    target = "Washington, D.C."
    context_blocks = asyncio.run(get_web_context_blocks(source, relation, target))
import warnings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from Triad_KGC.code.kg_verification.web_search.spider_search import spider_search
from Triad_KGC.code.kg_verification.web_search.crawling import crawl_pages
from typing import List


def triple_to_query(
        source_entity: str,
        relation: str,
        target_entity: str,
) -> str:
    """
    Convert a triple (source, relation, target) into a search query.
    :param source_entity: the source entity
    :param relation: the relation between the source and target
    :param target_entity: the target entity
    :return:
    """
    return f"{source_entity} AND {relation} AND {target_entity}"


async def get_web_context_blocks(
        source_entity: str,
        relation: str,
        target_entity: str,
        chunk_size=1000,
        overlap=200,
) -> List[str]:
    """
    Get web context blocks for a given triple (source, relation, target).
    :param source_entity: the source entity
    :param relation: the relation between the source and target
    :param target_entity: the target entity
    :param chunk_size: the size of each chunk
    :param overlap: the overlap between chunks
    :return: the web context blocks
    """
    query = triple_to_query(
        source_entity=source_entity,
        relation=relation,
        target_entity=target_entity,
    )
    search_result = await spider_search(query)
    urls = [r['url'] for r in search_result['content']]

    # Select top 3 URLs; if fewer than 3, issue a warning about insufficient context
    # If no URLs are found, return an empty list and warn
    if len(urls) < 3:
        warnings.warn("Fewer than 3 search results; context may be insufficient", UserWarning)
    else:
        urls = urls[:3]
    if len(urls) == 0:
        warnings.warn("Search results are empty; returning empty list", UserWarning)
        return []

    contents = await crawl_pages(urls)
    print(f"Fetched {len(contents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = text_splitter.create_documents(contents)
    return [chunk.page_content for chunk in chunks]


if __name__ == "__main__":
    import asyncio

    source = "United States"
    relation = "capital"
    target = "Washington, D.C."
    context_blocks = asyncio.run(get_web_context_blocks(source, relation, target))
