"""
# FastAPI application for KimyGuide: goal-based cold-start MOOC recommender
KimyGuide API (kimyguide.api.app)

This module defines a FastAPI application that exposes a lightweight
goal-based cold-start recommender over an OpenLearn-derived dataset.

High-level overview
- Loads a CSV dataset (default: data/processed/openlearn_courses.csv).
- Instantiates a TF-IDF baseline recommender (always available) and,
    optionally, an embedding-based recommender and a hybrid recommender
    (configurable via environment variables).
- Serves REST endpoints for health, evaluation, recommendation, and a
    small Jinja2-backed UI.

How to run
- From repository root:

Key configuration / environment variables
- KIMYGUIDE_DATA_PATH: override the default dataset CSV path.
- KIMYGUIDE_SKIP_EMBEDDINGS: if "1", embedding and hybrid models are
    skipped to allow fast/test runs without heavy model loading.

Module-level behavior
- On import the module attempts to:
        1. Load the dataset via _load_courses(path) which:
             - Validates required columns: course_id, title, description, tags, text.
             - Coerces these columns to strings, removes rows with empty text,
                 and normalizes course_id to str.
        2. Instantiate the TF-IDF recommender (TfidfGoalRecommender).
        3. Optionally instantiate:
             - EmbeddingRecommender using EmbeddingConfig (caching supported),
                 and HybridRecommender configured by HybridConfig.
- Initialization failures or environment-based skips leave optional model
    variables as None and are reflected in /health and by returning 503
    for embedding/hybrid recommendation requests when appropriate.

Primary endpoints
- GET /            : Landing page (Jinja2 template "home.html").
- GET /health      : Returns JSON health summary including:
        status, model_version, dataset_path, dataset_loaded, num_courses,
        embedding_model (name or None), embedding_enabled, hybrid_enabled.
- GET /api/eval    : Run a small offline evaluation (sanity | subject | desc)
        - Parameters: nq (num queries), k (cutoff), mode, n_boot (bootstrap samples).
        - Returns: metrics per model (MRR/Recall/NDCG with bootstrap CIs),
            coverage, diversity (from embeddings if available), k-sweep series,
            and raw recommendations per query for UI/analysis.
- GET /api/info    : Returns basic endpoints and model/version info.
- POST /recommend  : Main recommendation API.
        - Accepts RecommendRequest (goal text, model choice: tfidf|embedding|hybrid,
            k, explain flag, optional top_n_candidates).
        - Validates input, gates embedding/hybrid requests if embeddings are disabled.
        - Runs selected model and returns RecommendResponse containing:
            goal, k, model_version, and a list of RecommendationItem objects.
        - Each RecommendationItem can include:
            course_id, title, score, optional human-readable "why" (via SimpleExplainer),
            Evidence (matched_terms, matched_fields, candidate_pool_size, method,
            model_name, confidence), and metadata (tags, provider, subject, level,
            duration_hours, url, embedding_score, tfidf_score, meta_prior, level_adjustment).
        - Handles low-confidence and "no-strong-match" cases with configurable
            thresholds (hybrid_cfg.confidence_threshold and no_match_threshold), and
            in severe cases returns an empty recommendation list indicating no strong match.
- UI routes (/ui, /ui/compare, /ui/eval, /ui/dataset) serve Jinja templates and a small
    static JS/CSS bundle mounted at /static.
- GET /dataset/sample : Return a deterministic small sample of dataset rows for UI/tests.
- POST /evaluate      : Backwards-compatible evaluation route that delegates to
    quick_evaluate_models if available.

Important implementation notes
- Explainability: SimpleExplainer produces short, rule-based explanations and
    matched terms by inspecting overlaps between the goal and title/description/tags.
- Diversity: A simple diversity proxy computes 1 - mean cosine similarity across
    returned embedding pairs; returns 0.0 if embeddings unavailable.
- Evaluation: Uses DCG / nDCG calculations, mean reciprocal rank (MRR),
    recall@k and bootstrap confidence intervals for robustness reporting.
- Error handling:
        - Request validation errors return a 422 JSON with details.
        - Missing dataset or uninitialized required models surface as HTTP 500.
        - Invalid model names or bad parameters return 400.
        - Requests for embedding/hybrid when embeddings are disabled return 503.
- Model/version provenance: MODEL_VERSION string is exposed via responses andFastAPI app for KimyGuide (general cold-start).

Dataset:
  data/processed/openlearn_courses.csv

Run:
  PYTHONPATH=src uvicorn kimyguide.api.app:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

# Local application imports: pydantic schemas, explainability helpers and
# the three recommender implementations used by this API.
from kimyguide.api.schemas import Evidence, RecommendRequest, RecommendResponse, RecommendationItem
from kimyguide.explain.simple_explainer import SimpleExplainer
from kimyguide.models.embedding_recommender import EmbeddingConfig, EmbeddingRecommender
from kimyguide.models.hybrid_recommender import HybridConfig, HybridRecommender
from kimyguide.models.tfidf_recommender import TfidfGoalRecommender

# Optional (you already have this file; keep route for backwards compat)
try:
    from kimyguide.ui.evaluation import quick_evaluate_models  # type: ignore
except Exception:
    quick_evaluate_models = None  # type: ignore


app = FastAPI(
    title="KimyGuide API",
    version="0.6.2",
    description="General goal-based cold-start MOOC recommender (OpenLearn curated dataset).",
)

MODEL_VERSION = "v0.6.2-openlearn-general-coldstart"


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "message": "Request validation failed",
            "detail": exc.errors(),
            "received_body": exc.body,
        },
    )


BASE_DIR = Path(__file__).resolve().parents[3]

# Allow tests / dev to override dataset path
DATA_PATH = Path(
    os.getenv("KIMYGUIDE_DATA_PATH", str(BASE_DIR / "data" / "processed" / "openlearn_courses.csv"))
)

REQUIRED_COLUMNS = {"course_id", "title", "description", "tags", "text"}


def _load_courses(path: Path) -> pd.DataFrame:
    """Load a CSV of course metadata and prepare required columns.

    The function validates the presence of expected columns, coerces
    them to strings, removes rows with empty `text`, and standardises
    the id column type. It raises FileNotFoundError or KeyError on
    missing inputs so callers fail fast during startup.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    # Read CSV and validate required columns
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Ensure expected columns exist and are strings
    for col in ["course_id", "title", "description", "tags", "text"]:
        df[col] = df[col].fillna("").astype(str)

    # Standardise id type and drop items with empty text
    df["course_id"] = df["course_id"].astype(str)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    return df


# --- Initialize models ---
#
# At module import / startup I try to load the dataset and instantiate
# the TF-IDF baseline and (optionally) the embedding + hybrid models.
# Tests and lightweight runs can disable embeddings via the
# KIMYGUIDE_SKIP_EMBEDDINGS environment variable to avoid heavy model loads.
# The variables below are module-level and may remain None if initialization
# fails or embeddings are intentionally skipped.
courses_df: Optional[pd.DataFrame] = None
tfidf_model: Optional[TfidfGoalRecommender] = None
embed_model: Optional[EmbeddingRecommender] = None
hybrid_model: Optional[HybridRecommender] = None

# Simple, rule-based explainability helper used to extract matched terms
# and produce the human-readable `why` explanation attached to recommendations.
explainer = SimpleExplainer()

try:
    courses_df = _load_courses(DATA_PATH)

    # TF-IDF baseline (lightweight and always available)
    tfidf_model = TfidfGoalRecommender(courses_df)

    # Optionally skip embeddings (useful for tests / CI / fast local runs).
    # When disabled we leave embed_model and hybrid_model as None and
    # the API will respond with 503 for embedding/hybrid requests.
    skip_embeddings = os.getenv("KIMYGUIDE_SKIP_EMBEDDINGS", "0").strip() == "1"
    if skip_embeddings:
        print("[INFO] KIMYGUIDE_SKIP_EMBEDDINGS=1 -> embeddings + hybrid disabled")
        embed_model = None
        hybrid_model = None
    else:
        cache_path = BASE_DIR / "data" / "processed" / "openlearn_embeddings_cache.npz"
        embed_cfg = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            cache_path=cache_path,
            text_col="text",
            batch_size=64,
        )
        embed_model = EmbeddingRecommender(courses_df, cfg=embed_cfg)

        hybrid_cfg = HybridConfig(
            top_n_candidates=200,
            w_emb=0.75,
            w_tfidf=0.20,
            w_meta=0.05,
            confidence_threshold=0.50,
            no_match_threshold=0.50,
        )
        hybrid_model = HybridRecommender(courses_df, embedder=embed_model, tfidf=tfidf_model, cfg=hybrid_cfg)

    print(f"[INFO] Loaded dataset rows={len(courses_df)} from {DATA_PATH}")

except Exception as e:
    print(f"[ERROR] Failed to initialize models: {e}")


@app.get("/", response_class=HTMLResponse)
def landing_page(request: Request):
    """Landing page: render home template and expose model version."""
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,  # required by Jinja2Templates
            "model_version": MODEL_VERSION,  # show version in UI
        },
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns a JSON-serializable dict with basic status information about
    whether the dataset and the various models were successfully loaded.
    """
    # Basic OK: dataset loaded and TF-IDF model available (the lightweight baseline)
    ok = courses_df is not None and tfidf_model is not None

    # Flags for optional models (may be disabled via env / failed to initialize)
    embeddings_ok = embed_model is not None
    hybrid_ok = hybrid_model is not None

    # Build a concise health payload:
    return {
        # High-level status string for quick checks
        "status": "ok" if ok else "error",

        # Version / provenance info
        "model_version": MODEL_VERSION,
        "dataset_path": str(DATA_PATH),

        # Dataset/model availability details
        "dataset_loaded": bool(ok),
        # Number of courses loaded (0 if dataset missing)
        "num_courses": int(len(courses_df)) if courses_df is not None else 0,

        # Embedding model name (None-safe access). If embeddings disabled this is None.
        "embedding_model": getattr(getattr(embed_model, "cfg", None), "model_name", None),

        # Boolean flags indicating if optional components are enabled
        "embedding_enabled": bool(embeddings_ok),
        "hybrid_enabled": bool(hybrid_ok),
    }


# ============================================================
# Evaluation helpers (single, clean definitions)
# ============================================================

def _dcg(rels: List[int]) -> float:
    """Compute discounted cumulative gain for a list of relevance labels.

    rels should be a list of integers (0/1 for binary relevance). The
    standard DCG formula discounts by log2(rank+1) where rank starts at 1.
    """
    # rels: list of 0/1 relevance
    return sum((rel / math.log2(i + 2)) for i, rel in enumerate(rels))


def _ndcg_at_k(rels: List[int], k: int) -> float:
    """Normalized DCG at cutoff k."""
    rels = rels[:k]
    dcg = _dcg(rels)
    ideal = sorted(rels, reverse=True)
    idcg = _dcg(ideal)
    return float(dcg / idcg) if idcg > 0 else 0.0


def _bootstrap_ci(values: List[float], n_boot: int = 200, seed: int = 42) -> Dict[str, float]:
    """
    Returns dict: {"mean": m, "low": p2.5, "high": p97.5}
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}

    rng = np.random.default_rng(seed)
    means = []
    n = arr.size
    for _ in range(int(n_boot)):
        samp = rng.choice(arr, size=n, replace=True)
        means.append(float(np.mean(samp)))
    means = np.asarray(means)

    return {
        "mean": float(np.mean(arr)),
        "low": float(np.percentile(means, 2.5)),
        "high": float(np.percentile(means, 97.5)),
    }


def _diversity_from_embeddings(rec_ids: List[str], id_to_vec: Optional[Dict[str, np.ndarray]]) -> float:
    """
    Diversity proxy from embeddings: 1 - mean cosine similarity across pairs.
    If embeddings unavailable, returns 0.0.
    """
    if id_to_vec is None:
        return 0.0

    vecs = []
    for rid in rec_ids:
        v = id_to_vec.get(str(rid))
        if v is not None:
            vecs.append(v)

    if len(vecs) < 2:
        return 0.0

    X = np.vstack(vecs).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    S = X @ X.T

    sims = []
    n = S.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(float(S[i, j]))

    return float(1.0 - np.mean(sims)) if sims else 0.0


@app.get("/api/eval")
def api_eval(
    nq: int = 60,
    k: int = 10,
    mode: str = "sanity",   # sanity | subject | desc
    n_boot: int = 200,
) -> Dict[str, Any]:
    """
    Evaluation modes:
      - sanity: query=title, relevant={same course_id} (sanity check only; very easy)
      - subject: query=title, relevant={any course with same subject} (proxy relevance)
      - desc: query=desc snippet, relevant={same subject}, remove source course from results (less leakage)

    Returns:
      rows: per-model metrics (MRR/Recall/NDCG as bootstrap CI dicts) + coverage + diversity
      sweep: robustness curve vs K
      recommendations: per-model rec_ids_by_query (for UI coverage/diversity)
    """
    if courses_df is None or tfidf_model is None:
        raise HTTPException(status_code=500, detail="Dataset/models not initialized.")

    nq = int(max(10, min(nq, 300)))
    k = int(max(3, min(k, 20)))
    n_boot = int(max(50, min(n_boot, 2000)))

    mode = (mode or "sanity").strip().lower()
    if mode not in ("sanity", "subject", "desc"):
        raise HTTPException(status_code=400, detail="mode must be one of: sanity | subject | desc")

    skip_embeddings = os.getenv("KIMYGUIDE_SKIP_EMBEDDINGS", "0").strip() == "1"

    available_models = ["tfidf"]
    if not skip_embeddings and embed_model is not None:
        available_models.append("embedding")
    if not skip_embeddings and hybrid_model is not None:
        available_models.append("hybrid")

    n_items = len(courses_df)
    nq = min(nq, n_items)

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_items, size=nq, replace=False)
    sampled = courses_df.iloc[sample_idx].reset_index(drop=True)

    titles = sampled["title"].fillna("").astype(str).tolist()
    descs = sampled["description"].fillna("").astype(str).tolist()
    ids = sampled["course_id"].fillna("").astype(str).tolist()
    subjects = sampled["subject"].fillna("").astype(str).str.lower().tolist()

    # Build query text
    if mode == "desc":
        queries: List[str] = []
        for d, t in zip(descs, titles):
            d = (d or "").strip()
            if len(d) >= 140:
                queries.append(d[:140])
            elif len(d) >= 60:
                queries.append(d)
            else:
                queries.append(t)
    else:
        queries = titles

    # Subject -> ids map (proxy relevance)
    all_subjects = courses_df["subject"].fillna("").astype(str).str.lower().tolist()
    all_ids = courses_df["course_id"].fillna("").astype(str).tolist()

    subject_to_ids: Dict[str, set] = defaultdict(set)
    for cid, subj in zip(all_ids, all_subjects):
        subj = (subj or "").strip()
        if subj:
            subject_to_ids[subj].add(cid)

    def relevant_set(i: int) -> set:
        if mode == "sanity":
            return {ids[i]}
        subj = (subjects[i] or "").strip()
        return set(subject_to_ids.get(subj, set()))

    # Build id->embedding vector for diversity (if embeddings enabled)
    id_to_vec: Optional[Dict[str, np.ndarray]] = None
    if not skip_embeddings and embed_model is not None:
        try:
            id_to_vec = dict(
                zip(
                    embed_model.items["course_id"].astype(str).tolist(),
                    embed_model.item_embeds,
                )
            )
        except Exception:
            id_to_vec = None

    def run_model(model_name: str, top_k: int) -> Dict[str, Any]:
        per_mrr: List[float] = []
        per_recall: List[float] = []
        per_ndcg: List[float] = []
        per_rec_ids: List[List[str]] = []
        per_div: List[float] = []

        for i, (q, src_id) in enumerate(zip(queries, ids)):
            rel = relevant_set(i)

            if model_name == "tfidf":
                recs = tfidf_model.recommend(goal_text=q, top_k=top_k)
            elif model_name == "embedding":
                recs = embed_model.recommend(goal_text=q, top_k=top_k)  # type: ignore
            elif model_name == "hybrid":
                recs = hybrid_model.recommend(goal_text=q, top_k=top_k)  # type: ignore
            else:
                raise ValueError("unknown model")

            rec_ids = recs["course_id"].fillna("").astype(str).tolist()

            if mode == "desc":
                rec_ids = [rid for rid in rec_ids if rid != src_id][:top_k]

            per_rec_ids.append(rec_ids)

            rels = [1 if rid in rel else 0 for rid in rec_ids]
            per_ndcg.append(_ndcg_at_k(rels, top_k))

            hit = 1 if any(rid in rel for rid in rec_ids) else 0
            per_recall.append(float(hit))

            rr = 0.0
            for rank, rid in enumerate(rec_ids, start=1):
                if rid in rel:
                    rr = 1.0 / rank
                    break
            per_mrr.append(float(rr))

            per_div.append(_diversity_from_embeddings(rec_ids, id_to_vec=id_to_vec))

        return {
            "mrr": _bootstrap_ci(per_mrr, n_boot=n_boot, seed=42),
            "recall": _bootstrap_ci(per_recall, n_boot=n_boot, seed=43),
            "ndcg": _bootstrap_ci(per_ndcg, n_boot=n_boot, seed=44),
            "rec_ids_by_query": per_rec_ids,
            "diversity_mean": float(np.mean(per_div)) if per_div else 0.0,
        }

    # Main @k
    rows = []
    recs_payload: Dict[str, Any] = {}

    for m in available_models:
        out = run_model(m, k)

        unique_recs = set()
        for lst in out["rec_ids_by_query"]:
            unique_recs.update(lst)
        coverage = float(len(unique_recs) / max(1, n_items))

        rows.append(
            {
                "model": m,
                "mrr": out["mrr"],
                "recall": out["recall"],
                "ndcg": out["ndcg"],
                "coverage": coverage,
                "diversity": out["diversity_mean"],
            }
        )
        recs_payload[m] = out["rec_ids_by_query"]

    # K-sweep (robustness)
    sweep_ks = [1, 3, 5, 10, 15, 20]
    sweep_ks = [kk for kk in sweep_ks if kk <= k] or [k]

    sweep_series = []
    for m in available_models:
        mrr_list: List[float] = []
        recall_list: List[float] = []
        ndcg_list: List[float] = []
        for kk in sweep_ks:
            out = run_model(m, kk)
            mrr_list.append(out["mrr"]["mean"])
            recall_list.append(out["recall"]["mean"])
            ndcg_list.append(out["ndcg"]["mean"])
        sweep_series.append({"model": m, "mrr": mrr_list, "recall": recall_list, "ndcg": ndcg_list})

    return {
        "meta": {
            "mode": mode,
            "n_queries": nq,
            "k": k,
            "num_items": int(n_items),
            "models": available_models,
            "note": "sanity=self-retrieval (easy). subject/desc are stronger proxy evaluations.",
        },
        "rows": rows,
        "sweep": {"k": sweep_ks, "series": sweep_series},
        "recommendations": recs_payload,
    }


# ============================================================
# Recommendation API
# ============================================================

def _candidate_pool_size(req: RecommendRequest, model: str, total: int) -> int:
    if model in ("embedding", "hybrid"):
        return min(int(getattr(req, "top_n_candidates", 200)), total)
    return total

@app.get("/api/info")
def api_info():
    return {
        "name": "KimyGuide API",
        "docs": "/docs",
        "health": "/health",
        "ui": "/ui",
        "compare": "/ui/compare",
        "eval": "/ui/eval",
        "dataset": "/ui/dataset",
        "model_version": MODEL_VERSION,
        "dataset_path": str(DATA_PATH),
    }

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    if courses_df is None or tfidf_model is None:
        raise HTTPException(status_code=500, detail="Models not initialized. Check server logs.")

    goal = req.goal.strip()
    if not goal:
        raise HTTPException(status_code=400, detail="goal must not be empty.")

    model = req.model
    k = int(req.k)

    # Model gating if embeddings disabled: return a 503 Service Unavailable
    # when the caller requests embeddings/hybrid but the server was started
    # with embeddings disabled (e.g. in tests or lightweight runs).
    if model in ("embedding", "hybrid"):
        if embed_model is None or hybrid_model is None:
            raise HTTPException(
                status_code=503,
                detail="Embedding/Hybrid models are disabled (KIMYGUIDE_SKIP_EMBEDDINGS=1).",
            )

    # Allow caller to override candidate pool size for hybrid
    if model == "hybrid" and hybrid_model is not None:
        hybrid_model.cfg.top_n_candidates = int(getattr(req, "top_n_candidates", 200))

    # Run model
    if model == "tfidf":
        recs = tfidf_model.recommend(goal_text=goal, top_k=k)
        method = "tfidf"
        model_name = None

    elif model == "embedding":
        assert embed_model is not None
        recs = embed_model.recommend(goal_text=goal, top_k=k)
        method = "embeddings"
        model_name = embed_model.cfg.model_name

    elif model == "hybrid":
        assert embed_model is not None and hybrid_model is not None
        recs = hybrid_model.recommend(goal_text=goal, top_k=k)
        method = "hybrid"
        model_name = embed_model.cfg.model_name

    else:
        raise HTTPException(status_code=400, detail="model must be one of: tfidf | embedding | hybrid")

    pool_size = _candidate_pool_size(req, model, len(courses_df))

    # # Low-confidence handling (only meaningful for hybrid if it emits 'confidence')
    # low_confidence = False
    # confidence: Optional[float] = None
    # if "confidence" in recs.columns and len(recs):
    #     confidence = float(recs["confidence"].iloc[0])
    #     if model in ("embedding", "hybrid") and hybrid_model is not None:
    #         low_confidence = confidence < float(hybrid_model.cfg.confidence_threshold)

        # Low-confidence / no-strong-match handling
    low_confidence = False
    confidence: Optional[float] = None
    no_strong_match = False

    if "confidence" in recs.columns and len(recs):
        confidence = float(recs["confidence"].iloc[0])

        top_tfidf = float(recs["tfidf_score"].iloc[0]) if "tfidf_score" in recs.columns else 0.0
        top_meta = float(recs["meta_prior"].iloc[0]) if "meta_prior" in recs.columns else 0.0

        if model == "hybrid" and hybrid_model is not None:
            low_confidence = confidence < float(hybrid_model.cfg.confidence_threshold)

            no_strong_match = (
                confidence < float(hybrid_model.cfg.no_match_threshold)
                or (
                    confidence < 0.55
                    and top_tfidf < 0.08
                    and top_meta <= 0.0
                )
            )

        elif model == "embedding" and embed_model is not None:
            threshold_cfg = hybrid_model.cfg if hybrid_model is not None else None
            if threshold_cfg is not None:
                low_confidence = confidence < float(threshold_cfg.confidence_threshold)
                no_strong_match = confidence < float(threshold_cfg.no_match_threshold)

    if no_strong_match:
        return RecommendResponse(
            goal=goal,
            k=k,
            model_version=f"{MODEL_VERSION} ({method})",
            recommendations=[],
        )

    # Build the list of RecommendationItem objects to return. For each
    # candidate we compute a concise human-readable explanation (via the
    # SimpleExplainer), attach evidence (matched terms, scores, pool size)
    # and include lightweight metadata useful for display in the UI.
    items: List[RecommendationItem] = []
    for _, row in recs.iterrows():
        title = str(row.get("title", ""))
        description = str(row.get("description", ""))
        tags = str(row.get("tags", ""))

        # Extract a short natural-language explanation and the matched terms
        # (e.g. overlapping tokens across title/description/tags).
        why, matched_terms = explainer.explain(goal=goal, title=title, description=description, tags=tags)

        if low_confidence:
            # Override the why-text for low-confidence hybrid/embedding cases
            why = (
                "No strong matches were found for this goal in the current dataset. "
                "Showing the closest available courses based on semantic similarity."
            )

        evidence = Evidence(
            matched_terms=matched_terms,
            matched_fields=["title", "description", "tags"],
            candidate_pool_size=pool_size,
            method=method,
            model_name=model_name,
            confidence=confidence,
        )

        metadata: Dict[str, Any] = {"tags": tags}

        for col in ["provider", "subject", "level", "duration_hours", "url"]:
            if col in recs.columns:
                metadata[col] = row.get(col, "")

        for col in ["embedding_score", "tfidf_score", "meta_prior", "level_adjustment"]:
            if col in recs.columns:
                metadata[col] = float(row.get(col, 0.0))

        if confidence is not None:
            metadata["confidence"] = float(confidence)

        # Append the constructed RecommendationItem dataclass which will be
        # serialized by FastAPI according to the RecommendResponse schema.
        items.append(
            RecommendationItem(
                course_id=str(row.get("course_id", "")),
                title=title,
                score=float(row.get("score", 0.0)),
                why=why if req.explain else None,
                evidence=evidence if req.explain else None,
                metadata=metadata,
            )
        )

    return RecommendResponse(
        goal=goal,
        k=k,
        model_version=f"{MODEL_VERSION} ({method})",
        recommendations=items,
    )


# ============================================================
# UI (Jinja templates + static JS)
# ============================================================

UI_DIR = BASE_DIR / "src" / "kimyguide" / "ui"
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/ui", response_class=HTMLResponse)
def ui_home(request: Request):
    return templates.TemplateResponse(
        "ui.html",
        {"request": request, "model_version": MODEL_VERSION},
    )


@app.get("/ui/compare", response_class=HTMLResponse)
def ui_compare(request: Request):
    return templates.TemplateResponse(
        "compare.html",
        {"request": request, "model_version": MODEL_VERSION},
    )


@app.get("/ui/eval", response_class=HTMLResponse)
def ui_eval(request: Request):
    return templates.TemplateResponse(
        "eval.html",
        {"request": request, "model_version": MODEL_VERSION},
    )


@app.get("/ui/dataset", response_class=HTMLResponse)
def ui_dataset(request: Request):
    n = int(len(courses_df)) if courses_df is not None else 0
    return templates.TemplateResponse(
        "dataset.html",
        {
            "request": request,
            "model_version": MODEL_VERSION,
            "num_courses": n,
            "dataset_path": str(DATA_PATH),
        },
    )


@app.get("/dataset/sample")
def dataset_sample(limit: int = 50) -> Dict[str, Any]:
    """
    Return a small sample of dataset rows for UI/testing.

    Args:
      limit: requested number of rows (will be clamped to [1, 200]).

    The sample uses a fixed random_state to be reproducible across calls.
    """
    if courses_df is None:
        # dataset must be loaded before serving samples
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    # Clamp the requested limit to a sensible maximum to avoid large payloads
    limit = max(1, min(int(limit), 200))

    # Use a deterministic sample for stability in tests / UI
    sample = courses_df.sample(n=min(limit, len(courses_df)), random_state=7)

    # Select a few useful columns, replace NaNs with empty strings and serialize
    return {
        "rows": sample[["course_id", "title", "subject", "level", "url"]]
        .fillna("")
        .to_dict(orient="records")
    }


# Backwards-compatible POST eval (optional)
@app.post("/evaluate")
def evaluate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    If you still use /evaluate anywhere, this keeps it working.
    Pref: use GET /api/eval for the UI.
    """
    if quick_evaluate_models is None:
        raise HTTPException(status_code=500, detail="quick_evaluate_models not available")

    if courses_df is None or tfidf_model is None:
        raise HTTPException(status_code=500, detail="Models not initialized")

    n_queries = int(payload.get("n_queries", 60))
    k = int(payload.get("k", 10))

    if n_queries < 10 or n_queries > 300:
        raise HTTPException(status_code=400, detail="n_queries must be in [10, 300]")
    if k < 3 or k > 20:
        raise HTTPException(status_code=400, detail="k must be in [3, 20]")

    results = quick_evaluate_models(
        df=courses_df,
        tfidf=tfidf_model,
        embed=embed_model,
        hybrid=hybrid_model,
        n_queries=n_queries,
        k=k,
    )
    results["meta"] = {
        "n_queries": n_queries,
        "k": k,
        "embedding_enabled": embed_model is not None,
        "hybrid_enabled": hybrid_model is not None,
        "dataset_rows": int(len(courses_df)),
        "model_version": MODEL_VERSION,
    }
    return results