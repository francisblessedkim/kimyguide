import pandas as pd
from kimyguide.models.tfidf_recommender import TfidfGoalRecommender
from kimyguide.models.embedding_recommender import EmbeddingRecommender, EmbeddingConfig
from kimyguide.models.hybrid_recommender import HybridRecommender, HybridConfig


def test_hybrid_recommender_returns_k_and_scores(tmp_path):
    df = pd.DataFrame(
        [
            {"course_id": "c1", "title": "Intro Data", "description": "data science stats", "tags": "", "text": "data science stats"},
            {"course_id": "c2", "title": "Neural Nets", "description": "deep learning neural networks", "tags": "", "text": "deep learning neural networks"},
            {"course_id": "c3", "title": "German Travel", "description": "german travel phrases", "tags": "", "text": "german travel phrases"},
        ]
    )

    tfidf = TfidfGoalRecommender(df)

    cache = tmp_path / "embeds.npz"
    emb = EmbeddingRecommender(df, cfg=EmbeddingConfig(cache_path=cache))

    hybrid = HybridRecommender(df, embedder=emb, tfidf=tfidf, cfg=HybridConfig(top_n_candidates=3))

    recs = hybrid.recommend("deep learning neural networks", top_k=2)
    assert len(recs) == 2
    assert "score" in recs.columns
    assert "embedding_score" in recs.columns
    assert "tfidf_score" in recs.columns
    assert "confidence" in recs.columns