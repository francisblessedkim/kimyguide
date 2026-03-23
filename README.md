# kimyguide

Small prototype for a goal-based course recommender (TF-IDF + optional
embeddings hybrid). This README explains how to install, run, and test
the project on macOS (or Linux) and Windows.

Contents
- `src/kimyguide`: Python package (API, models, explainers, features)
- `scripts/generate_mooclike_dataset.py`: helper to create a small CSV dataset
- `run_cli.py`: simple CLI demo to query the recommender

Prerequisites
- Python 3.9+ (the repository was developed on Python 3.10+; tests run in a
  virtualenv). On macOS you can use the system Python or pyenv. On Windows
  use the official Python installer from python.org.

Quick setup (macOS / Linux)

1. Open a terminal and cd into the project root (where this README sits).

2. Create and activate a virtual environment:

```bash
# create venv
python -m venv .venv

# activate (macOS / Linux - zsh/bash)
source .venv/bin/activate
```

3. Upgrade pip and install requirements:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Quick setup (Windows - PowerShell)

```powershell
# create venv
python -m venv .venv

# activate (PowerShell)
.\.venv\Scripts\Activate.ps1

# upgrade pip and install
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Environment variables
- `KIMYGUIDE_DATA_PATH` — optional path to a CSV dataset (the loader expects
  specific columns; see `scripts/generate_mooclike_dataset.py` for structure).
- `KIMYGUIDE_SKIP_EMBEDDINGS` — set to `1` to skip loading heavy embedding
  models (useful for tests or when you don't have sentence-transformers
  installed).

Example (run with generated sample data):

```bash
# generate a tiny dataset used by demos and tests
python scripts/generate_mooclike_dataset.py --out data/raw/courses_mooclike.csv

# run the CLI demo
python run_cli.py --data-path data/raw/courses_mooclike.csv
```

Run the web API (development)

The FastAPI app is available as `kimyguide.api.app:app` and can be started
with `uvicorn`. For a development run with auto-reload:

```bash
# make sure PYTHONPATH includes src so imports resolve
PYTHONPATH=src uvicorn kimyguide.api.app:app --reload --host 127.0.0.1 --port 8000
```

Once the server starts you can view the running application in your browser:

- Landing UI: http://127.0.0.1:8000/ (Jinja2-backed home page)
- Alternative UI routes: http://127.0.0.1:8000/ui and http://127.0.0.1:8000/ui/compare
- API (OpenAPI) docs: http://127.0.0.1:8000/docs

Open whichever URL matches your workflow — the landing page shows the
simple web UI, while `/docs` exposes the interactive API explorer.

Running tests

Before running tests make sure the venv is active and dependencies are
installed. The test suite expects environment variables to be set before the
application is imported; the included tests set those themselves, but if you
want to run a particular test interactively you can set:

```bash
# skip embeddings for fast test runs
export KIMYGUIDE_SKIP_EMBEDDINGS=1

# run the whole test suite
pytest -q

# run a single test file
pytest -q tests/test_hybrid_recommender_unit.py -q
```

Notes & troubleshooting
- If imports fail when running `uvicorn`, ensure you used `PYTHONPATH=src` or
  run from a context where `src` is on sys.path (for example `pip install -e .`).
- If you get memory or model-download errors, set `KIMYGUIDE_SKIP_EMBEDDINGS=1`
  to avoid loading embedding models.
- For Windows PowerShell the `Activate.ps1` script may be blocked by execution
  policy; you can enable it temporarily with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

Contributing and contact
- This repo is a small prototype — contributions and issue reports are
  welcome. Please open an issue or pull request with proposed changes.

License: MIT
