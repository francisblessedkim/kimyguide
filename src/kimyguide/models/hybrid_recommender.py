"""
Hybrid recommender for KimyGuide (general cold-start).

Goal:
Solve cold-start by recommending from the first interaction using:
  - semantic embeddings (SentenceTransformer)
  - lexical signal (TF-IDF cosine)
  - lightweight metadata prior
  - level-aware reranking (beginner/intermediate/advanced intent)

The hybrid score is:

  final = w_emb * emb_score + w_tfidf * tfidf_score + w_meta * meta_prior + level_adjustment

Confidence:
We expose a simple confidence measure based on the semantic strength of the
best match (emb_score top-1). This helps UX: if confidence is low, we can show
a fallback message or no-results state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class HybridConfig:
    top_n_candidates: int = 200
    w_emb: float = 0.75
    w_tfidf: float = 0.20
    w_meta: float = 0.05
    confidence_threshold: float = 0.45
    no_match_threshold: float = 0.40
    beginner_level_boost: float = 0.12
    intermediate_penalty: float = -0.04
    advanced_penalty: float = -0.10


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
        if cfg is None:
            cfg = HybridConfig()

        if text_col not in items.columns:
            raise KeyError(f"Expected '{text_col}' in items DataFrame.")

        self.cfg = cfg
        self.items = items.reset_index(drop=True).copy()
        self.embedder = embedder
        self.tfidf = tfidf
        self.text_col = text_col

        self._vectorizer = getattr(self.tfidf, "vectorizer", None)
        self._item_matrix = getattr(self.tfidf, "item_matrix", None)

        if self._vectorizer is None or self._item_matrix is None:
            raise ValueError("TF-IDF model is missing vectorizer/item_matrix (unexpected).")

    def _is_beginner_goal(self, goal: str) -> bool:
        g = goal.lower().strip()

        beginner_phrases = [
            "learn",
            "start learning",
            "beginner",
            "beginners",
            "basic",
            "basics",
            "intro",
            "introduction",
            "getting started",
            "get started",
            "new to",
            "first time",
            "foundation",
            "foundations",
        ]

        advanced_phrases = [
            "advanced",
            "intermediate",
            "expert",
            "specialized",
            "specialised",
            "deeper",
            "deep dive",
            "master",
        ]

        has_beginner = any(p in g for p in beginner_phrases)
        has_advanced = any(p in g for p in advanced_phrases)

        return has_beginner and not has_advanced

    def _level_adjustment(self, goal: str, row: pd.Series) -> float:
        level = _safe_str(row.get("level", "")).lower().strip()

        if not self._is_beginner_goal(goal):
            return 0.0

        if "intro" in level or "beginner" in level:
            return self.cfg.beginner_level_boost
        if "intermediate" in level:
            return self.cfg.intermediate_penalty
        if "advanced" in level:
            return self.cfg.advanced_penalty
        return 0.0

    def _meta_prior(self, goal: str, row: pd.Series) -> float:
        g = goal.lower()
        subject = _safe_str(row.get("subject", "")).lower()
        title = _safe_str(row.get("title", "")).lower()
        tags = _safe_str(row.get("tags", "")).lower()
        text = _safe_str(row.get("text", "")).lower()

        is_data_goal = any(
            k in g for k in [
                "data science", "data", "statistics", "machine learning", "ml", "python"
            ]
        )
        is_ai_goal = any(
            k in g for k in [
                "deep learning", "neural", "ai", "artificial intelligence"
            ]
        )
        is_lang_goal = any(
            k in g for k in [
                "german", "french", "spanish", "italian", "chinese",
                "language", "languages", "travel"
            ]
        )

        prior = 0.0
        searchable = f"{subject} {title} {tags} {text}"

        if is_data_goal:
            if any(k in searchable for k in ["digital computing", "data", "statistics", "python", "machine learning"]):
                prior += 1.0

        if is_ai_goal:
            if any(k in searchable for k in ["artificial intelligence", "ai", "machine learning", "neural", "data", "comput"]):
                prior += 1.0

        if is_lang_goal:
            if any(k in searchable for k in ["languages", "language", "french", "german", "spanish", "italian", "chinese"]):
                prior += 1.0

        return float(min(1.0, prior))

    def _tfidf_scores(self, goal_text: str) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity

        goal_vec = self._vectorizer.transform([goal_text])
        sims = cosine_similarity(goal_vec, self._item_matrix)[0]
        return sims.astype(np.float32)

    def recommend(self, goal_text: str, top_k: int = 5) -> pd.DataFrame:
        if not goal_text or not str(goal_text).strip():
            raise ValueError("goal_text is empty.")

        n_items = len(self.items)
        top_n = min(int(self.cfg.top_n_candidates), n_items)

        emb_sims = self.embedder.score_all(goal_text).astype(np.float32)
        cand_idx = np.argsort(emb_sims)[::-1][:top_n]

        cand = self.items.iloc[cand_idx].copy()
        cand_emb = emb_sims[cand_idx]

        tfidf_all = self._tfidf_scores(goal_text)
        cand_tfidf = tfidf_all[cand_idx]

        meta = np.array(
            [self._meta_prior(goal_text, row) for _, row in cand.iterrows()],
            dtype=np.float32
        )

        level_adj = np.array(
            [self._level_adjustment(goal_text, row) for _, row in cand.iterrows()],
            dtype=np.float32
        )

        emb_01 = _normalize_01(cand_emb)
        tfidf_01 = _normalize_01(cand_tfidf)
        meta_01 = meta

        final = (
            self.cfg.w_emb * emb_01
            + self.cfg.w_tfidf * tfidf_01
            + self.cfg.w_meta * meta_01
            + level_adj
        ).astype(np.float32)

        confidence = float(cand_emb[0]) if len(cand_emb) else 0.0

        cand["embedding_score"] = cand_emb
        cand["tfidf_score"] = cand_tfidf
        cand["meta_prior"] = meta_01
        cand["level_adjustment"] = level_adj
        cand["score"] = final
        cand["goal"] = goal_text
        cand["confidence"] = confidence

        cand = cand.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
        return cand