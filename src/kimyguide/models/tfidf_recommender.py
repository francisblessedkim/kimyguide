"""A tiny TF-IDF based recommender used for prototyping.

This module implements a straightforward pipeline:
  - Fit a TF-IDF vectorizer on item texts
  - Represent a free-text user goal in the same vector space
  - Score items by cosine similarity and return the top-K

This is intentionally minimal to keep the prototype easy to
understand and modify.
"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..features.text_embeddings import build_tfidf_matrix


class TfidfGoalRecommender:
    """Simple goal → content recommender using TF-IDF and cosine similarity."""

    def __init__(self, items: pd.DataFrame):
        """Initialise the recommender and precompute item features.

        Parameters
        - items: DataFrame containing at least an 'item_id' column and a
          'text' column with the item-level text used for TF-IDF.
        """
        if "text" not in items.columns:
            raise KeyError("Expected a 'text' column in items DataFrame.")

        # Keep a local copy of items and build TF-IDF features once
        self.items = items.reset_index(drop=True).copy()
        self.vectorizer, self.item_matrix = build_tfidf_matrix(self.items, text_col="text")

    def recommend(
        self,
        goal_text: str,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Return the top_k most similar items to the provided goal_text.

        The function transforms the user goal into TF-IDF space and
        computes cosine similarity against the precomputed item matrix.
        Returns a DataFrame of the selected items with added 'score'
        and 'goal' columns.
        """
        if not goal_text:
            raise ValueError("goal_text is empty.")

        # Transform the goal text into the same TF-IDF feature space
        goal_vec = self.vectorizer.transform([goal_text])

        # Compute cosine similarity between the single goal vector and
        # all item vectors; result shape is (1, n_items)
        sims = cosine_similarity(goal_vec, self.item_matrix)[0]  # shape (n_items,)

        # Pick top-k indices (largest similarity first)
        top_indices = np.argsort(sims)[::-1][:top_k]

        # Return the selected rows together with scores and the original goal
        recs = self.items.iloc[top_indices].copy()
        recs["score"] = sims[top_indices]
        recs["goal"] = goal_text
        return recs
