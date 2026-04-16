"""
Agency source configurations.
Using Google News RSS — free, reliable, returns actual rating action news
from Reuters, Bloomberg, FT, and agency press releases simultaneously.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgencyConfig:
    name: str
    short_code: str
    base_url: str
    rss_feeds: list[str]
    press_release_url: Optional[str]
    scrape_strategy: str  # "rss" | "html" | "both"
    metadata_fields: list[str] = field(default_factory=list)


AGENCY_CONFIGS = {

    "dbrs": AgencyConfig(
        name="DBRS Morningstar",
        short_code="DBRS",
        base_url="https://dbrs.morningstar.com",
        rss_feeds=[
            "https://news.google.com/rss/search?q=DBRS+Morningstar+rating+action&hl=en-US&gl=US&ceid=US:en",
        ],
        press_release_url="https://dbrs.morningstar.com/research",
        scrape_strategy="rss",
        metadata_fields=["issuer", "rating", "trend", "sector", "date", "action_type"]
    ),

    "fitch": AgencyConfig(
        name="Fitch Ratings",
        short_code="FITCH",
        base_url="https://www.fitchratings.com",
        rss_feeds=[
            "https://news.google.com/rss/search?q=Fitch+Ratings+downgrade+upgrade+affirm&hl=en-US&gl=US&ceid=US:en",
        ],
        press_release_url="https://www.fitchratings.com/research/rating-actions",
        scrape_strategy="rss",
        metadata_fields=["issuer", "rating", "outlook", "sector", "date", "action_type"]
    ),

    "moodys": AgencyConfig(
        name="Moody's Ratings",
        short_code="MCO",
        base_url="https://www.moodys.com",
        rss_feeds=[
            "https://news.google.com/rss/search?q=Moodys+rating+action+downgrade+upgrade&hl=en-US&gl=US&ceid=US:en",
        ],
        press_release_url="https://www.moodys.com/research/rating-action",
        scrape_strategy="rss",
        metadata_fields=["issuer", "rating", "outlook", "sector", "date"]
    ),

    "sp": AgencyConfig(
        name="S&P Global Ratings",
        short_code="SP",
        base_url="https://www.spglobal.com/ratings",
        rss_feeds=[
            "https://news.google.com/rss/search?q=SP+Global+Ratings+downgrade+upgrade+affirm&hl=en-US&gl=US&ceid=US:en",
        ],
        press_release_url="https://www.spglobal.com/ratings/en/research-insights/articles",
        scrape_strategy="rss",
        metadata_fields=["issuer", "rating", "outlook", "sector", "date", "action_type"]
    ),
}
