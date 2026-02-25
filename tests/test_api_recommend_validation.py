import importlib
import pandas as pd
from fastapi.testclient import TestClient


def _write_tiny_dataset(csv_path):
    df = pd.DataFrame(
        [
            {
                "course_id": "c1",
                "title": "Intro to Data Science",
                "description": "Learn data science fundamentals.",
                "tags": "data science",
                "text": "Intro to Data Science Learn data science fundamentals",
            }
        ]
    )
    df.to_csv(csv_path, index=False)


def _load_test_app(tmp_path, monkeypatch):
    csv_path = tmp_path / "openlearn_courses.csv"
    _write_tiny_dataset(csv_path)

    # Set env vars before importing so the app picks them up on module load
    monkeypatch.setenv("KIMYGUIDE_DATA_PATH", str(csv_path))
    monkeypatch.setenv("KIMYGUIDE_SKIP_EMBEDDINGS", "1")

    import sys
    import importlib

    if "kimyguide.api.app" in sys.modules:
        del sys.modules["kimyguide.api.app"]

    import kimyguide.api.app as app_module
    importlib.reload(app_module)
    return app_module


def test_recommend_requires_goal(tmp_path, monkeypatch):
    app_module = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app_module.app)

    # missing goal => 422 (pydantic)
    r = client.post("/recommend", json={"k": 5, "model": "hybrid", "top_n_candidates": 50})
    assert r.status_code == 422

    # empty goal => 400 (your explicit check)
    r = client.post("/recommend", json={"goal": "   ", "k": 5, "model": "hybrid", "top_n_candidates": 50})
    assert r.status_code == 400


def test_recommend_invalid_model(tmp_path, monkeypatch):
    app_module = _load_test_app(tmp_path, monkeypatch)
    client = TestClient(app_module.app)

    r = client.post("/recommend", json={"goal": "learn data", "k": 5, "model": "badmodel", "top_n_candidates": 50})
    assert r.status_code == 422  # schema should reject if Literal is used