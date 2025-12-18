# kimyguide

Small prototype for a goal-based course recommender (TF-IDF based).

Contents
- `src/kimyguide`: package modules (data loader, TF-IDF features, recommender, explainers)
- `scripts/generate_mooclike_dataset.py`: utility to generate a small synthetic CSV for development
- `run_cli.py`: tiny command-line demo to try the recommender locally

Quick start
1. (Optional) create a virtual environment and install requirements (scikit-learn, pandas).
2. Generate a small dataset: `python scripts/generate_mooclike_dataset.py`
3. Run the interactive demo: `python run_cli.py`

Notes
- This repository is a prototype. Large/raw data files are excluded by
  default from tracking (see `.gitignore`). Use the scripts to
  generate small synthetic data for experiments.

License: MIT
