"""Simple, transparent explanation helpers for recommendations.

These functions aim to produce short, human-friendly explanations by
extracting a few keywords from the user goal and the course text and
reporting their overlap. The approach is intentionally lightweight —
good enough for prototyping explainability UX.
"""

import re
from typing import List
import pandas as pd

# Domain-agnostic stopwords + MOOC filler words. These are removed
# during keyword extraction to avoid explaining recommendations with
# noisy tokens like "course" or "learn".
STOPWORDS = {
    "want", "learn", "learning", "basics", "basic", "course", "lesson",
    "cover", "covers", "about", "this", "that", "with", "from", "into",
    "able", "will", "also", "discuss", "focuses", "focus", "using", "use",
    "real", "projects", "practical", "introduction", "foundations",
    "hands", "hand", "on", "bootcamp", "build", "building", "systems",
    "apply", "applications", "end"
}


def extract_keywords(text: str, max_words: int = 6) -> List[str]:
    """Extract a short list of meaningful keywords from free text.

    The function lowercases the input, tokenises on alphabetic
    sequences, filters out short tokens and common filler words, and
    preserves the original token order up to `max_words` unique
    keywords.
    """
    # Tokenise on alphabetic runs only (removes punctuation/numbers).
    # Lowercasing here ensures matching is case-insensitive.
    tokens = re.findall(r"\b[a-zA-Z]+\b", str(text).lower())

    # Filter out short tokens and noisy filler words (see STOPWORDS).
    # The length threshold (>=4) is a cheap heuristic to drop many stop
    # words while keeping meaningful stems like 'train', 'learn', 'data'.
    tokens = [t for t in tokens if len(t) >= 4 and t not in STOPWORDS]

    seen = set()
    keywords = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            keywords.append(token)
        if len(keywords) >= max_words:
            break

    return keywords


def add_explanations(recs: pd.DataFrame) -> pd.DataFrame:
    """Attach short textual explanations to a DataFrame of recommendations.

    The input `recs` is expected to contain a 'goal' column (the user
    goal text) and 'text' column (item text). The function returns a
    copy of the DataFrame with an added 'explanation' column.
    """
    recs = recs.copy()

    if recs.empty:
        recs["explanation"] = []
        return recs

    goal_text = recs["goal"].iloc[0]
    goal_keywords = extract_keywords(goal_text, max_words=6)

    explanations = []

    for _, row in recs.iterrows():
        item_text = row.get("text", "")
        item_keywords = extract_keywords(item_text, max_words=10)

        # Find which goal keywords appear in the item keywords
        overlap = [kw for kw in goal_keywords if kw in item_keywords]

        if overlap:
            # Prefer to show a small number of overlapping tokens
            because = ", ".join(overlap[:4])
            covered = ", ".join(item_keywords[:3]) if item_keywords else "key concepts"
            explanation = (
                f"Recommended because your goal mentions {because}. "
                f"This course also covers {covered}."
            )
        else:
            # Fallback: show a short topic hint derived from item keywords
            topic_hint = ", ".join(item_keywords[:4]) if item_keywords else "related concepts"
            explanation = (
                "Recommended because it matches your learning goal based on content similarity "
                f"and covers {topic_hint}."
            )

        explanations.append(explanation)

    recs["explanation"] = explanations
    return recs

class SimpleExplainer:
    """Lightweight explainer wrapper used by the API.

    This thin class keeps the functional helpers but exposes a single
    `explain()` method that the FastAPI endpoints call. The goal is to
    provide a consistent tuple (why:str, matched_terms:List[str]) which
    can be attached to a `RecommendationItem.evidence` or rendered by the
    UI. The implementation is intentionally simple and deterministic so
    it is easy to reason about and test.
    """

    def explain(self, goal: str, title: str = "", description: str = "", tags: str = ""):
        """Return a short explanation and the list of matched terms.

        Parameters
        - goal: user-provided goal text
        - title/description/tags: item fields to consider when extracting
          keywords for matching

        Returns
        - why: short human-readable explanation
        - matched: list of matched keyword tokens (may be empty)
        """

        # Extract compact keyword lists for both goal and item text
        goal_keywords = extract_keywords(goal, max_words=6)

        item_text = f"{title} {description} {tags}"
        item_keywords = extract_keywords(item_text, max_words=10)

        # Intersection in preserved goal order - prefer goal tokens present in item
        matched = [kw for kw in goal_keywords if kw in item_keywords]

        if matched:
            because = ", ".join(matched[:6])
            why = (
                f"Recommended because your goal mentions {because}, "
                "and this course directly covers those topics."
            )
        else:
            # Generic fallback explanation when explicit overlap is small
            why = (
                "Recommended because it is related to your learning goal based on content similarity."
            )

        return why, matched
