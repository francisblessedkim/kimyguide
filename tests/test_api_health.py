import pandas as pd
from fastapi.testclient import TestClient


def _write_tiny_dataset(csv_path):
    df = pd.DataFrame(
        [
            {
                "course_id": "c1",
                "title": "Intro to Data Science",
                "description": "Learn data science fundamentals, data analysis, and basic machine learning.",
                "tags": "data science, ml",
                "text": "Intro to Data Science Learn data science fundamentals data analysis machine learning data science ml",
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
                "description": "Learn useful German phrases for travel, greetings, and ordering food.",
                "tags": "german, travel",
                "text": "Beginners German for Travel Learn useful German phrases travel greetings ordering food german travel",
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

    # Set env vars BEFORE importing the app so the module reads them on import
    monkeypatch.setenv("KIMYGUIDE_DATA_PATH", str(csv_path))
    monkeypatch.setenv("KIMYGUIDE_SKIP_EMBEDDINGS", "1")

    import sys
    import importlib

    # Ensure a clean import so env vars are respected
    if "kimyguide.api.app" in sys.modules:
        del sys.modules["kimyguide.api.app"]

    import kimyguide.api.app as app_module
    importlib.reload(app_module)
    return app_module


def test_root_and_health(tmp_path, monkeypatch):
    """Smoke test for the landing page, /api/info, and /health endpoint."""
    app_module = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app_module.app)

    # Root now serves the landing HTML page
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]

    # API info endpoint returns the metadata previously expected at "/"
    r = client.get("/api/info")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "KimyGuide API"
    assert body["docs"] == "/docs"
    assert body["health"] == "/health"
    assert body["ui"] == "/ui"
    assert "model_version" in body

    # /health endpoint should report service status and dataset info
    r = client.get("/health")
    assert r.status_code == 200
    h = r.json()
    assert h["status"] == "ok"
    assert h["dataset_loaded"] is True
    assert h["num_courses"] == 3