"""
Agency source configurations.
RSS feeds from major agencies are paywalled/deprecated as of 2024-2025.
Strategy: scrape public press release pages + use PR Newswire/Globe Newswire
which freely aggregate all agency announcements.
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
            # PR Newswire RSS for DBRS/Morningstar announcements — publicly accessible
            "https://www.prnewswire.com/rss/news-releases-list.rss?company=morningstar-dbrs",
        ],
        press_release_url="https://dbrs.morningstar.com/research",
        scrape_strategy="html",
        metadata_fields=["issuer", "rating", "trend", "sector", "date", "action_type"]
    ),

    "fitch": AgencyConfig(
        name="Fitch Ratings",
        short_code="FITCH",
        base_url="https://www.fitchratings.com",
        rss_feeds=[
    "https://www.prnewswire.com/rss/news-releases-list.rss?company=fitch-ratings",
],
        press_release_url="https://www.fitchratings.com/research/rating-actions",
        scrape_strategy="both",
        metadata_fields=["issuer", "rating", "outlook", "sector", "date", "action_type"]
    ),

    "moodys": AgencyConfig(
        name="Moody's Ratings",
        short_code="MCO",
        base_url="https://www.moodys.com",
        rss_feeds=[
            # Moody's press releases on PR Newswire
            "https://www.prnewswire.com/rss/news-releases-list.rss?company=moodys-investors-service",
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
            # S&P press releases on PR Newswire
            "https://www.prnewswire.com/rss/news-releases-list.rss?company=sp-global-ratings",
            # Business Wire also carries S&P
            "https://feed.businesswire.com/rss/home/?rss=G22&keyword=S%26P+Global+Ratings",
        ],
        press_release_url="https://www.spglobal.com/ratings/en/research-insights/articles",
        scrape_strategy="rss",
        metadata_fields=["issuer", "rating", "outlook", "sector", "date", "action_type"]
    ),
}
