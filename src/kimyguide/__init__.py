"""kimyguide package

Lightweight package glue for the KimyGuide prototype. The package
exposes small modules for loading data, building feature matrices,
running a TF-IDF recommender and generating simple explanations.

This file intentionally keeps exports minimal — import submodules
explicitly by path (for example:
`from src.kimyguide.data.loader import load_courses`).
"""

__all__ = [
	"data",
	"features",
	"models",
	"explain",
]
