from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def _mrr_at_k(results: list[list[str]], gold: list[str], k: int) -> float:
    """Mean Reciprocal Rank@k."""
    rr_sum = 0.0
    for preds, g in zip(results, gold):
        preds_k = preds[:k]
        try:
            rank = preds_k.index(g) + 1
            rr_sum += 1.0 / rank
        except ValueError:
            rr_sum += 0.0
    return rr_sum / max(1, len(gold))


def _recall_at_k(results: list[list[str]], gold: list[str], k: int) -> float:
    """Recall@k where each query has exactly 1 relevant item."""
    hits = 0
    for preds, g in zip(results, gold):
        if g in preds[:k]:
            hits += 1
    return hits / max(1, len(gold))


def quick_evaluate_models(
    df: pd.DataFrame,
    tfidf: Any,
    embed: Optional[Any],
    hybrid: Optional[Any],
    n_queries: int = 60,
    k: int = 10,
    random_state: int = 7,
) -> Dict[str, Any]:
    """
    Build evaluation queries from course titles.
    For each sampled course:
      query = title
      relevant = that course_id
    Then compute MRR@k + Recall@k for each model.
    """

    sample = df.sample(n=min(n_queries, len(df)), random_state=random_state).reset_index(drop=True)
    queries = sample["title"].fillna("").astype(str).tolist()
    gold_ids = sample["course_id"].astype(str).tolist()

    def run_model(name: str) -> Optional[Dict[str, Any]]:
        if name == "tfidf":
            model = tfidf
        elif name == "embedding":
            if embed is None:
                return None
            model = embed
        elif name == "hybrid":
            if hybrid is None:
                return None
            model = hybrid
        else:
            return None

        all_preds: list[list[str]] = []
        for q in queries:
            recs = model.recommend(goal_text=q, top_k=k)
            preds = recs["course_id"].astype(str).tolist()
            all_preds.append(preds)

        return {
            "MRR@k": _mrr_at_k(all_preds, gold_ids, k),
            "Recall@k": _recall_at_k(all_preds, gold_ids, k),
        }

    out: Dict[str, Any] = {"results": {}}
    out["results"]["tfidf"] = run_model("tfidf")
    out["results"]["embedding"] = run_model("embedding")
    out["results"]["hybrid"] = run_model("hybrid")

    return out