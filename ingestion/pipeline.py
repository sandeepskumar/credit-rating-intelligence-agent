"""
Ingestion Pipeline
==================
Fetches rating actions from each agency → parses → embeds → stores in ChromaDB.
"""

import os
import time
import logging
from datetime import datetime
from typing import Optional

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ingestion.agency_configs import AGENCY_CONFIGS, AgencyConfig

load_dotenv()
logger = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/vectorstore")
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY_SECONDS", "2"))
MAX_DOCS = int(os.getenv("MAX_DOCS_PER_AGENCY", "50"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Research Bot - Academic Use)",
    "Accept": "application/rss+xml, application/xml, text/html",
}


def fetch_rss_entries(feed_url: str, agency_code: str) -> list[dict]:
    try:
        feed = feedparser.parse(feed_url)
        logger.info(f"[{agency_code}] RSS: fetched {len(feed.entries)} entries from {feed_url}")
        return feed.entries[:MAX_DOCS]
    except Exception as e:
        logger.error(f"[{agency_code}] RSS fetch failed: {e}")
        return []


def fetch_html_page(url: str, agency_code: str) -> Optional[str]:
    try:
        time.sleep(REQUEST_DELAY)
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.error(f"[{agency_code}] HTML fetch failed for {url}: {e}")
        return None


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def rss_entry_to_document(entry: dict, agency_config: AgencyConfig) -> Document:
    pub_date = datetime.now()
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        pub_date = datetime(*entry.published_parsed[:6])

    content = entry.get("summary", "") or entry.get("content", [{}])[0].get("value", "")

    metadata = {
        "agency": agency_config.short_code,
        "agency_name": agency_config.name,
        "source_url": entry.get("link", ""),
        "title": entry.get("title", ""),
        "publication_date": pub_date.isoformat(),
        "doc_type": "rating_action",
    }
    return Document(page_content=content, metadata=metadata)


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )


def get_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name="credit_ratings",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def ingest_agency(agency_code: str) -> int:
    config = AGENCY_CONFIGS.get(agency_code)
    if not config:
        raise ValueError(f"Unknown agency: {agency_code}")

    logger.info(f"Starting ingestion for {config.name}")
    raw_docs: list[Document] = []

    for feed_url in config.rss_feeds:
        entries = fetch_rss_entries(feed_url, config.short_code)
        for entry in entries:
            doc = rss_entry_to_document(entry, config)
            raw_docs.append(doc)

    if config.scrape_strategy in ("html", "both") and config.press_release_url:
        html = fetch_html_page(config.press_release_url, config.short_code)
        if html:
            text = extract_text_from_html(html)
            raw_docs.append(Document(
                page_content=text,
                metadata={
                    "agency": config.short_code,
                    "agency_name": config.name,
                    "source_url": config.press_release_url,
                    "publication_date": datetime.now().isoformat(),
                    "doc_type": "press_release_page",
                }
            ))

    splitter = get_text_splitter()
    chunks = splitter.split_documents(raw_docs)
    logger.info(f"[{config.short_code}] Split {len(raw_docs)} docs into {len(chunks)} chunks")

    if chunks:
        vs = get_vector_store()
        vs.add_documents(chunks)
        logger.info(f"[{config.short_code}] Stored {len(chunks)} chunks in ChromaDB")

    return len(chunks)


def ingest_all_agencies() -> dict[str, int]:
    results = {}
    for agency_code in AGENCY_CONFIGS:
        try:
            count = ingest_agency(agency_code)
            results[agency_code] = count
        except Exception as e:
            logger.error(f"Ingestion failed for {agency_code}: {e}")
            results[agency_code] = 0
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting full ingestion pipeline...")
    results = ingest_all_agencies()
    print("\nIngestion complete:")
    for agency, count in results.items():
        print(f"  {agency.upper()}: {count} chunks stored")
