# src/kimyguide/api/schemas.py
from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class Filters(BaseModel):
    level: Optional[List[str]] = Field(default=None, description="Filter by course level(s)")
    format: Optional[List[str]] = Field(default=None, description="Filter by format(s) if available")
    max_duration_hours: Optional[float] = Field(default=None, ge=0, description="Maximum duration in hours")


class RecommendRequest(BaseModel):
    goal: str = Field(..., min_length=3, description="Learner goal in natural language")
    k: int = Field(5, ge=1, le=50, description="Number of recommendations to return")

    # Which ranking pipeline to use
    model: Literal["tfidf", "embedding", "hybrid"] = Field(
        "hybrid",
        description="Recommendation model: tfidf (baseline), embedding (semantic), hybrid (semantic + lexical + metadata prior).",
    )

    # Only used by embedding/hybrid for initial retrieval
    top_n_candidates: int = Field(
        200,
        ge=20,
        le=1000,
        description="Candidate pool size for embedding retrieval (embedding/hybrid only).",
    )

    filters: Optional[Filters] = None
    explain: bool = Field(True, description="Whether to include explanations")


class Evidence(BaseModel):
    matched_terms: List[str] = Field(default_factory=list)
    matched_fields: List[str] = Field(default_factory=lambda: ["title", "description", "tags"])

    # Diagnostics / transparency (great for report + debugging)
    candidate_pool_size: Optional[int] = None
    method: Optional[str] = None
    model_name: Optional[str] = None
    confidence: Optional[float] = None  # semantic strength signal


class RecommendationItem(BaseModel):
    course_id: str
    title: str
    score: float
    why: Optional[str] = None
    evidence: Optional[Evidence] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecommendResponse(BaseModel):
    goal: str
    k: int
    model_version: str
    recommendations: List[RecommendationItem]


class DatasetInfo(BaseModel):
    dataset_name: str
    num_items: int
    fields: List[str]
    created_by: str
    notes: str