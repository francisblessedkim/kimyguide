"""Text feature construction utilities.

Right now this module exposes a single helper to build a TF-IDF
matrix from a DataFrame of items. The vectorizer is configured to use
unigrams and bigrams and removes English stop words by default.
"""

from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_matrix(
    items: pd.DataFrame,
    text_col: str = "text",
    max_features: int = 50000,
) -> Tuple[TfidfVectorizer, any]:
    """Fit a TF-IDF vectorizer on item texts and return the fitted
    vectorizer together with the item-term sparse matrix.

    The function accepts a DataFrame and a column name containing text
    for each item. The returned vectorizer can be used to transform
    arbitrary query text (e.g. user goals) into the same feature
    space.
    """
    # Ensure we have string values and a deterministic corpus list
    corpus = items[text_col].fillna("").astype(str).tolist()

    # Configure TF-IDF for short to medium-length texts.
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
    )

    # Fit on item corpus and return both components
    item_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, item_matrix
