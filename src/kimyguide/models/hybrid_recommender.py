"""
Hybrid recommender for KimyGuide (general cold-start).

Goal:
Solve cold-start by recommending from the *first interaction* using:
  - semantic embeddings (SentenceTransformer)
  - lexical signal (TF-IDF cosine)
  - lightweight metadata prior (e.g., "data" goals favor computing/statistics)

The hybrid score is:

  final = w_emb * emb_score + w_tfidf * tfidf_score + w_meta * meta_prior

Confidence:
We expose a simple confidence measure based on the semantic strength of the
best match (emb_score top-1). This helps UX: if confidence is low, we can show
a fallback message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import pandas as pd


@dataclass
class HybridConfig:
    top_n_candidates: int = 200
    w_emb: float = 0.75
    w_tfidf: float = 0.20
    w_meta: float = 0.05
    confidence_threshold: float = 0.45


def _safe_str(x) -> str:
    return "" if x is None else str(x)


def _normalize_01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax - xmin < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / (xmax - xmin)).astype(np.float32)


class HybridRecommender:
    def __init__(
        self,
        items: pd.DataFrame,
        embedder,
        tfidf,
        cfg: Optional[HybridConfig] = None,
        text_col: str = "text",
    ):
        """
        Parameters
        ----------
        items: DataFrame containing course rows
        embedder: EmbeddingRecommender instance (must implement score_all)
        tfidf: TfidfGoalRecommender instance
        """
        if cfg is None:
            cfg = HybridConfig()

        if text_col not in items.columns:
            raise KeyError(f"Expected '{text_col}' in items DataFrame.")

        self.cfg = cfg
        self.items = items.reset_index(drop=True).copy()
        self.embedder = embedder
        self.tfidf = tfidf
        self.text_col = text_col

        # Precompute a TF-IDF similarity helper:
        # We can reuse tfidf.vectorizer and item_matrix from TfidfGoalRecommender
        self._vectorizer = getattr(self.tfidf, "vectorizer", None)
        self._item_matrix = getattr(self.tfidf, "item_matrix", None)

        if self._vectorizer is None or self._item_matrix is None:
            raise ValueError("TF-IDF model is missing vectorizer/item_matrix (unexpected).")

    def _meta_prior(self, goal: str, row: pd.Series) -> float:
        """
        Lightweight bias to improve general cold-start.
        Not meant to dominate ranking — just a small tie-breaker.
        """
        g = goal.lower()
        subject = _safe_str(row.get("subject", "")).lower()
        title = _safe_str(row.get("title", "")).lower()

        # detect goal "domain"
        is_data_goal = any(k in g for k in ["data science", "data", "statistics", "machine learning", "ml", "python"])
        is_ai_goal = any(k in g for k in ["deep learning", "neural", "ai", "artificial intelligence"])
        is_lang_goal = any(k in g for k in ["german", "french", "spanish", "language", "travel"])

        prior = 0.0

        if is_data_goal:
            if any(k in subject for k in ["digital computing", "science maths technology"]) or "data" in title:
                prior += 1.0
        if is_ai_goal:
            if any(k in subject for k in ["digital computing", "science maths technology"]) or any(k in title for k in ["machine", "data", "comput"]):
                prior += 1.0
        if is_lang_goal:
            if "languages" in subject or "german" in title or "language" in title:
                prior += 1.0

        # cap to [0, 1]
        return float(min(1.0, prior))

    def _tfidf_scores(self, goal_text: str) -> np.ndarray:
        # cosine similarity between goal vec and all items (same logic as tfidf recommender)
        goal_vec = self._vectorizer.transform([goal_text])
        # item_matrix likely sparse; dot product via sklearn cosine is ok,
        # but we can approximate with cosine_similarity if available.
        # We'll do safe import to avoid extra dependency changes.
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(goal_vec, self._item_matrix)[0]
        return sims.astype(np.float32)

    def recommend(self, goal_text: str, top_k: int = 5) -> pd.DataFrame:
        if not goal_text or not str(goal_text).strip():
            raise ValueError("goal_text is empty.")

        n_items = len(self.items)
        top_n = min(int(self.cfg.top_n_candidates), n_items)

        # 1) Semantic retrieval over all items, take top-N candidates
        emb_sims = self.embedder.score_all(goal_text).astype(np.float32)
        cand_idx = np.argsort(emb_sims)[::-1][:top_n]

        cand = self.items.iloc[cand_idx].copy()
        cand_emb = emb_sims[cand_idx]

        # 2) Lexical score (TF-IDF) over all, then slice to candidates
        tfidf_all = self._tfidf_scores(goal_text)
        cand_tfidf = tfidf_all[cand_idx]

        # 3) Metadata prior (cheap, explainable)
        meta = np.array([self._meta_prior(goal_text, row) for _, row in cand.iterrows()], dtype=np.float32)

        # Normalize component scores to [0,1] within the candidate set (stabilizes weighting)
        emb_01 = _normalize_01(cand_emb)
        tfidf_01 = _normalize_01(cand_tfidf)
        meta_01 = meta  # already in [0,1]

        final = (
            self.cfg.w_emb * emb_01
            + self.cfg.w_tfidf * tfidf_01
            + self.cfg.w_meta * meta_01
        ).astype(np.float32)

        # Confidence: semantic strength of the best match (raw cosine, not normalized)
        confidence = float(cand_emb[0]) if len(cand_emb) else 0.0

        # Build output
        cand["embedding_score"] = cand_emb
        cand["tfidf_score"] = cand_tfidf
        cand["meta_prior"] = meta_01
        cand["score"] = final
        cand["goal"] = goal_text
        cand["confidence"] = confidence

        # Rank by final hybrid score
        cand = cand.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
        return cand