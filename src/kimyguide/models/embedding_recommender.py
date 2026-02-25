"""
Embedding-based recommender for KimyGuide (general cold-start).

Pipeline:
  - Encode each course text with a SentenceTransformer model
  - Encode user goal
  - Rank by cosine similarity

Includes disk caching for course embeddings to speed up API startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"  # strong English baseline
    cache_path: Optional[Path] = None     # if None, no caching
    text_col: str = "text"
    batch_size: int = 64


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


class EmbeddingRecommender:
    def __init__(self, items: pd.DataFrame, cfg: Optional[EmbeddingConfig] = None):
        if cfg is None:
            cfg = EmbeddingConfig()

        if cfg.text_col not in items.columns:
            raise KeyError(f"Expected a '{cfg.text_col}' column in items DataFrame.")

        self.cfg = cfg
        self.items = items.reset_index(drop=True).copy()

        self.model = SentenceTransformer(cfg.model_name)

        self.item_embeds = self._load_or_build_embeddings()

    def _cache_key(self) -> str:
        # A lightweight cache signature: model + number of rows
        return f"{self.cfg.model_name}__n={len(self.items)}__col={self.cfg.text_col}"

    def _load_or_build_embeddings(self) -> np.ndarray:
        cp = self.cfg.cache_path
        if cp is not None and cp.exists():
            try:
                data = np.load(cp, allow_pickle=True)
                if str(data.get("cache_key", "")) == self._cache_key():
                    embeds = data["embeddings"].astype(np.float32)
                    return _l2_normalize(embeds)
            except Exception:
                pass  # fall through to rebuild

        texts = self.items[self.cfg.text_col].fillna("").astype(str).tolist()
        embeds = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # SentenceTransformers can do this
        ).astype(np.float32)

        if cp is not None:
            cp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cp, embeddings=embeds, cache_key=self._cache_key())

        return embeds

    def score_all(self, goal_text: str) -> np.ndarray:
        if not goal_text or not str(goal_text).strip():
            raise ValueError("goal_text is empty.")

        g = self.model.encode(
            [str(goal_text)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)[0]

        # cosine similarity since everything is normalized
        sims = np.dot(self.item_embeds, g)
        return sims

    def recommend(self, goal_text: str, top_k: int = 5) -> pd.DataFrame:
        sims = self.score_all(goal_text)
        top_idx = np.argsort(sims)[::-1][:top_k]

        recs = self.items.iloc[top_idx].copy()
        recs["score"] = sims[top_idx]
        recs["goal"] = goal_text
        return recs
