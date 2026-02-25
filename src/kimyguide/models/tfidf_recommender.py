"""A tiny TF-IDF based recommender used for prototyping.

This module implements a straightforward pipeline:
  - Fit a TF-IDF vectorizer on item texts
  - Represent a free-text user goal in the same vector space
  - Score items by cosine similarity and return the top-K
"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..features.text_embeddings import build_tfidf_matrix


class TfidfGoalRecommender:
    """Simple goal → content recommender using TF-IDF and cosine similarity."""

    def __init__(self, items: pd.DataFrame):
        if "text" not in items.columns:
            raise KeyError("Expected a 'text' column in items DataFrame.")

        self.items = items.reset_index(drop=True).copy()
        self.vectorizer, self.item_matrix = build_tfidf_matrix(self.items, text_col="text")

    def score_all(self, goal_text: str) -> np.ndarray:
        if not goal_text:
            raise ValueError("goal_text is empty.")

        goal_vec = self.vectorizer.transform([goal_text])
        sims = cosine_similarity(goal_vec, self.item_matrix)[0]
        return sims

    def recommend(self, goal_text: str, top_k: int = 5) -> pd.DataFrame:
        sims = self.score_all(goal_text)
        top_indices = np.argsort(sims)[::-1][:top_k]

        recs = self.items.iloc[top_indices].copy()
        recs["score"] = sims[top_indices]
        recs["goal"] = goal_text
        return recs
