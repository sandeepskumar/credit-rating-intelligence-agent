"""
Pydantic models for credit rating documents.
Think of these as the standardized "order ticket" format every agency's
raw data gets converted into before entering our kitchen.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class RatingAction(BaseModel):
    """A single rating action from any agency."""

    # Identity
    issuer_name: str = Field(description="Full legal name of the rated entity")
    issuer_ticker: Optional[str] = Field(None, description="Exchange ticker if available")
    isin: Optional[str] = Field(None, description="ISIN identifier if available")

    # Rating details
    agency: Literal["DBRS", "FITCH", "MCO", "SP"]
    rating: str = Field(description="e.g. 'BBB', 'Aa2', 'A+'")
    outlook: Optional[str] = Field(None, description="Stable / Positive / Negative / Watch")
    action_type: str = Field(
        description="e.g. 'Upgrade', 'Downgrade', 'Affirm', 'Watch Negative', 'New Rating'"
    )

    # Classification
    sector: Optional[str] = Field(None, description="e.g. 'Financials', 'Energy', 'Sovereign'")
    geography: Optional[str] = Field(None, description="e.g. 'Canada', 'US', 'Europe'")
    instrument_type: Optional[str] = Field(None, description="e.g. 'Senior Unsecured', 'Covered Bond'")

    # Source
    source_url: str
    publication_date: datetime
    raw_text: str = Field(description="Full press release / commentary text")
    summary: Optional[str] = Field(None, description="LLM-generated summary, populated after ingestion")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SplitRatingAlert(BaseModel):
    """Raised when two agencies have meaningfully different ratings on the same issuer."""

    issuer_name: str
    ratings: dict[str, str] = Field(description="{ agency: rating_string }")
    outlooks: dict[str, Optional[str]]
    divergence_summary: str = Field(description="Plain English explanation of the gap")
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    severity: Literal["minor", "moderate", "significant"]
