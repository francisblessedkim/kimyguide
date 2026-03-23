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

    # Request top-2 recommendations for the given query text.
    recs = hybrid.recommend("deep learning neural networks", top_k=2)

    # Expect exactly two recommendations returned.
    assert len(recs) == 2

    # Ensure the DataFrame includes the combined score for each recommendation.
    assert "score" in recs.columns

    # Ensure the embedding-based similarity score is present.
    assert "embedding_score" in recs.columns

    # Ensure the TF-IDF based similarity score is present.
    assert "tfidf_score" in recs.columns

    # Ensure a confidence metric (e.g., normalized or calibrated combined score) is provided.
    assert "confidence" in recs.columns