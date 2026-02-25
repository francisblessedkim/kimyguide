import importlib
import pandas as pd
from fastapi.testclient import TestClient


def _write_tiny_dataset(csv_path):
    df = pd.DataFrame(
        [
            {
                "course_id": "c1",
                "title": "Intro to Data Science",
                "description": "Learn data science fundamentals, Python, and statistics.",
                "tags": "data science, python",
                "text": "Intro to Data Science Learn data science fundamentals Python statistics data science python",
                "provider": "OpenLearn",
                "subject": "digital computing",
                "level": "Introductory",
                "duration_hours": 10.0,
                "url": "https://example.com/c1",
            },
            {
                "course_id": "c2",
                "title": "Neural Networks Basics",
                "description": "Understand neural networks, deep learning, and backpropagation.",
                "tags": "deep learning, neural networks",
                "text": "Neural Networks Basics Understand neural networks deep learning backpropagation deep learning neural networks",
                "provider": "OpenLearn",
                "subject": "science maths technology",
                "level": "Intermediate",
                "duration_hours": 12.0,
                "url": "https://example.com/c2",
            },
            {
                "course_id": "c3",
                "title": "Beginners German for Travel",
                "description": "Learn German phrases for travel, greetings and food.",
                "tags": "german, travel",
                "text": "Beginners German for Travel Learn German phrases travel greetings food german travel",
                "provider": "OpenLearn",
                "subject": "languages",
                "level": "Introductory",
                "duration_hours": 8.0,
                "url": "https://example.com/c3",
            },
        ]
    )
    df.to_csv(csv_path, index=False)


def _load_test_app(tmp_path, monkeypatch):
    csv_path = tmp_path / "openlearn_courses.csv"
    _write_tiny_dataset(csv_path)

    # Set env vars first so the app module picks them up on import
    monkeypatch.setenv("KIMYGUIDE_DATA_PATH", str(csv_path))
    monkeypatch.setenv("KIMYGUIDE_SKIP_EMBEDDINGS", "1")

    import sys

    if "kimyguide.api.app" in sys.modules:
        del sys.modules["kimyguide.api.app"]

    import kimyguide.api.app as app_module

    # Reload so models init against our dataset + env
    importlib.reload(app_module)
    return app_module


def _assert_recommend_shape(body, k_expected):
    assert "goal" in body
    assert "k" in body
    assert "model_version" in body
    assert "recommendations" in body
    assert len(body["recommendations"]) == k_expected

    for rec in body["recommendations"]:
        assert "course_id" in rec
        assert "title" in rec
        assert "score" in rec
        assert isinstance(rec["score"], (int, float))


def test_recommend_tfidf(tmp_path, monkeypatch):
    app_module = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app_module.app)

    r = client.post("/recommend", json={"goal": "learn german for travel", "k": 2, "model": "tfidf"})
    assert r.status_code == 200
    body = r.json()
    _assert_recommend_shape(body, 2)


def test_recommend_embedding_disabled_when_skip_embeddings(tmp_path, monkeypatch):
    app_module = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app_module.app)

    r = client.post(
        "/recommend",
        json={"goal": "deep learning neural networks", "k": 2, "model": "embedding", "top_n_candidates": 50},
    )
    assert r.status_code == 503
    body = r.json()
    assert "detail" in body
    assert "disabled" in body["detail"].lower()


def test_recommend_hybrid_disabled_when_skip_embeddings(tmp_path, monkeypatch):
    app_module = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app_module.app)

    r = client.post(
        "/recommend",
        json={"goal": "deep learning neural networks", "k": 2, "model": "hybrid", "top_n_candidates": 50},
    )
    assert r.status_code == 503
    body = r.json()
    assert "detail" in body
    assert "disabled" in body["detail"].lower()