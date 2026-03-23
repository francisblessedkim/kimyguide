```markdown
# KimyGuide 

KimyGuide is a goal-based educational recommender system designed to generate meaningful course recommendations under **cold-start conditions**, where no prior user data is available.

The system combines:
- **TF-IDF (lexical model)** → traditional keyword-based approach  
- **Sentence-BERT embeddings (semantic model)** → semantic understanding  
#!/usr/bin/env text
# KimyGuide

Goal-based course recommender prototype (TF-IDF + optional embeddings + hybrid).

This repository contains a small FastAPI application, simple CLI demo and
recommender implementations used for experimenting with cold-start
recommendation (no user history required).

<!-- toc -->
- [Quick links](#quick-links)
- [Features](#features)
- [Contents](#contents)
- [Prerequisites](#prerequisites)
- [Setup (macOS / Linux)](#setup-macos--linux)
- [Setup (Windows)](#setup-windows)
- [Environment variables](#environment-variables)
- [Run the app (development)](#run-the-app-development)
- [Access the app](#access-the-app)
- [CLI demo](#cli-demo)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
<!-- /toc -->

## Quick links

- Landing UI: `http://127.0.0.1:8000/`
- Compare UI: `http://127.0.0.1:8000/ui/compare`
- API docs (OpenAPI): `http://127.0.0.1:8000/docs`

## Features

- Goal-based recommendations (free-text goal → ranked courses)
- TF-IDF baseline, sentence-transformer embeddings, and a hybrid reranker
- Small Jinja2-backed UI and an interactive OpenAPI explorer

## Contents

Top-level layout (important files/folders):

```
src/kimyguide/         # Python package (API, models, features, explainers, UI)
data/                  # example and processed datasets
run_cli.py             # small CLI demo
requirements.txt       # runtime dependencies
tests/                 # pytest test suite
README.md
```

## Prerequisites

- Python 3.9+ (3.10+ recommended)
- Git

## Setup (macOS / Linux)

```bash
git clone https://github.com/francisblessedkim/kimyguide.git
cd kimyguide

# create and activate venv
python -m venv .venv
source .venv/bin/activate

# upgrade packaging tools and install requirements
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Setup (Windows - PowerShell)

```powershell
git clone https://github.com/francisblessedkim/kimyguide.git
cd kimyguide

# create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# upgrade and install
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Environment variables

- `KIMYGUIDE_DATA_PATH` — optional, path to your dataset CSV. Defaults to
	`data/processed/openlearn_courses.csv`.
- `KIMYGUIDE_SKIP_EMBEDDINGS` — set to `1` to skip embedding/hybrid model
	initialization (useful in CI / low-memory environments).

## Run the app (development)

The FastAPI app is exposed at `kimyguide.api.app:app`. When running from the
repo root remember to add `src/` to `PYTHONPATH` so the package imports
resolve correctly.

```bash
# from repo root
PYTHONPATH=src uvicorn kimyguide.api.app:app --reload --host 127.0.0.1 --port 8000
```

## Access the app

- Landing UI: https://127.0.0.1:8000/ or http://127.0.0.1:8000/
- Compare UI: https://127.0.0.1:8000/ui/compare
- API docs (OpenAPI): https://127.0.0.1:8000/docs

Open whichever suits your workflow — `/` is the Jinja2-powered UI, `/docs`
exposes the API endpoints and request/response schemas.

## CLI demo

The repository contains a small CLI example that can be used with a local
CSV dataset.

```bash
python run_cli.py --data-path data/raw/openlearn_courses.csv
```

## Testing

Run tests from the repo root with the venv active. The suite sets environment
variables where needed, but for a fast run you can disable embeddings:

```bash
# skip embeddings for faster/local runs
export KIMYGUIDE_SKIP_EMBEDDINGS=1

pytest -q
```

Run a single file:

```bash
pytest -q tests/test_hybrid_recommender_unit.py
```

## Troubleshooting

- If imports fail when running `uvicorn`, ensure `PYTHONPATH=src` or install
	the package in editable mode: `pip install -e .` (you may need a simple
	`pyproject.toml`/`setup.cfg` for editable installs).
- If embedding models fail to download or consume too much memory, set
	`KIMYGUIDE_SKIP_EMBEDDINGS=1` to run only TF-IDF.

## Contributing

As of right now contributions are not welcome and this repo is for examining purposes. 


## License

MIT
