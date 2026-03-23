# API request/response schemas (Pydantic models) used by the FastAPI app.

"""API request/response schemas (Pydantic models) used by the FastAPI app.

This module defines the small set of models exchanged between the client
UI and the recommendation API. Keeping these schemas compact and typed
helps with validation, automatic OpenAPI generation, and self-documenting
endpoints.

Key models:
 - RecommendRequest: payload for /recommend requests (goal, model, k, etc.)
 - RecommendResponse: structured response containing RecommendationItem entries
 - Evidence: transparency diagnostics attached to recommendations
 - DatasetInfo: metadata about the dataset (used by UI / dataset endpoints)
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class Filters(BaseModel):
    """Optional filters that can be applied to recommendation requests.

    These are intentionally lightweight: they are evaluated after retrieval
    (client-side or server-side filtering) and can be extended later to
    include provider, subject, etc.
    """

    # Filter by course difficulty/level, e.g. ["Beginner", "Intermediate"]
    level: Optional[List[str]] = Field(default=None, description="Filter by course level(s)")

    # Course format metadata (if present in dataset): video, text, mixed, etc.
    format: Optional[List[str]] = Field(default=None, description="Filter by format(s) if available")

    # Maximum duration in hours (useful to restrict longer courses)
    max_duration_hours: Optional[float] = Field(default=None, ge=0, description="Maximum duration in hours")


class RecommendRequest(BaseModel):
    """Request payload for the `/recommend` endpoint.

    Fields are validated by Pydantic; defaults are sensible for an interactive
    UI (k=5, hybrid model). `top_n_candidates` is only relevant for
    embedding/hybrid retrieval and controls the initial candidate pool size.
    """

    # Natural-language learning goal provided by the user.
    goal: str = Field(..., min_length=3, description="Learner goal in natural language")

    # How many final results to return (after reranking / filtering)
    k: int = Field(5, ge=1, le=50, description="Number of recommendations to return")

    # Which ranking/retrieval pipeline to use. "hybrid" combines signals.
    model: Literal["tfidf", "embedding", "hybrid"] = Field(
        "hybrid",
        description="Recommendation model: tfidf (baseline), embedding (semantic), hybrid (semantic + lexical + metadata prior).",
    )

    # Candidate pool size used during embedding-based retrieval; ignored by TF-IDF.
    top_n_candidates: int = Field(
        200,
        ge=20,
        le=1000,
        description="Candidate pool size for embedding retrieval (embedding/hybrid only).",
    )

    # Optional filters and explanation toggle
    filters: Optional[Filters] = None
    explain: bool = Field(True, description="Whether to include explanations")


class Evidence(BaseModel):
    """Auxiliary information attached to individual recommendations.

    This model is intended for transparency and debugging: it provides a
    short list of matched terms, which fields matched, and optional
    diagnostics such as the retrieval method and a confidence score.
    """

    # Tokens or phrases that matched between the goal and the item
    matched_terms: List[str] = Field(default_factory=list)

    # Which fields were considered when extracting matches (defaults shown)
    matched_fields: List[str] = Field(default_factory=lambda: ["title", "description", "tags"])

    # Diagnostics / transparency (useful for reports and debugging)
    candidate_pool_size: Optional[int] = None
    method: Optional[str] = None
    model_name: Optional[str] = None

    # Optional semantic confidence score (useful for hybrid fallback logic)
    confidence: Optional[float] = None  # semantic strength signal


class RecommendationItem(BaseModel):
    """Single recommendation record returned to clients/UI.

    Contains the minimal display fields (id, title, score) together with an
    optional human-readable `why`, structured `evidence`, and flexible
    `metadata` for UI rendering (provider, subject, url, scores, etc.).
    """

    course_id: str
    title: str
    score: float
    why: Optional[str] = None
    evidence: Optional[Evidence] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecommendResponse(BaseModel):
    """Standardized response for `/recommend`.

    `model_version` includes both the service version and info about the
    ranking method used so clients can display provenance to end users.
    """

    goal: str
    k: int
    model_version: str
    recommendations: List[RecommendationItem]


class DatasetInfo(BaseModel):
    """Metadata about a dataset exposed to UI or admin endpoints.

    This is not heavily used in the current UI but provides a schema for any
    dataset listing / diagnostic pages you might add later.
    """

    dataset_name: str
    num_items: int
    fields: List[str]
    created_by: str
    notes: str