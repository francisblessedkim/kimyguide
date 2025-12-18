"""Utility to generate a synthetic, MOOC-like courses CSV.

This script is used during development to create a small corpus of
course items with titles, descriptions, tags and metadata. The
generated CSV mimics a MOOCCube-like dataset for prototyping the
recommender pipeline.
"""

from __future__ import annotations

import random
from pathlib import Path
import csv

# Project layout helpers -------------------------------------------------
# PROJECT_ROOT points to the repository root from the current file's
# location; OUT_PATH is where the synthetic CSV will be written.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "courses_mooclike.csv"

# Make results reproducible during development
random.seed(42)

# Topic / template configuration -----------------------------------------
# A short taxonomy of topics each paired with a set of plausible skills
# that will be used to fill templates for titles and descriptions.
TOPICS = [
    ("Machine Learning", ["classification", "regression", "model evaluation", "feature engineering", "overfitting", "cross-validation"]),
    ("Deep Learning", ["neural networks", "backpropagation", "CNNs", "RNNs", "transformers", "regularization"]),
    ("Natural Language Processing", ["tokenization", "embeddings", "BERT", "topic modeling", "text classification", "prompting"]),
    ("Data Science", ["EDA", "data cleaning", "visualization", "statistics", "hypothesis testing", "pipelines"]),
    ("Python", ["functions", "data structures", "pandas", "numpy", "debugging", "APIs"]),
    ("SQL", ["SELECT queries", "joins", "group by", "indexes", "window functions", "data modeling"]),
    ("Recommender Systems", ["collaborative filtering", "content-based", "cold start", "ranking metrics", "hybrid models", "explanations"]),
    ("Product Analytics", ["funnels", "retention", "cohort analysis", "A/B testing", "instrumentation", "dashboards"]),
    ("Cloud & Deployment", ["APIs", "Docker", "CI/CD", "model serving", "monitoring", "reproducibility"]),
    ("Time Series", ["forecasting", "seasonality", "ARIMA", "anomaly detection", "feature windows", "evaluation"]),
]

LEVELS = ["Beginner", "Intermediate", "Advanced"]
FORMATS = ["Video", "Reading", "Project", "Mixed"]

TITLE_TEMPLATES = [
    "Introduction to {topic}",
    "{topic} Foundations",
    "Practical {topic} for Real Projects",
    "{topic}: From Basics to Applications",
    "Hands-on {topic} Bootcamp",
    "{topic} with Python",
    "Applied {topic} and Evaluation",
    "Building Systems with {topic}",
]

DESC_TEMPLATES = [
    "In this course, you will learn {skill_a}, {skill_b}, and {skill_c}. "
    "You will work through real examples, short quizzes, and mini-projects to build confidence. "
    "By the end, you will be able to apply these ideas to practical problems and explain your decisions using clear reasoning.",

    "This course focuses on {skill_a} and {skill_b}, with an emphasis on {skill_c}. "
    "You will learn how to avoid common mistakes, interpret results, and improve model performance using disciplined evaluation. "
    "The course includes guided exercises and a capstone-style task.",

    "You will explore {skill_a}, {skill_b}, and {skill_c} through step-by-step lessons and hands-on practice. "
    "We will cover key terminology, best practices, and how to make trade-offs when building real systems. "
    "Ideal for learners who want structured progress and measurable outcomes.",

    "Learn {skill_a} and {skill_b} while developing intuition for {skill_c}. "
    "The course combines conceptual lessons with implementation-focused labs, helping you connect theory to practice. "
    "You will finish with a portfolio-ready mini project.",
]


def make_course(i: int) -> dict:
    """Create a single synthetic course record.

    The function randomly selects a topic, level and format, and fills
    title/description templates using a small sampled set of skills. The
    returned dict matches the CSV fieldnames used when writing the
    dataset.
    """
    topic, skills = random.choice(TOPICS)
    level = random.choices(LEVELS, weights=[0.45, 0.40, 0.15])[0]
    fmt = random.choice(FORMATS)

    # Duration ranges by level
    if level == "Beginner":
        duration = random.randint(4, 12)
    elif level == "Intermediate":
        duration = random.randint(8, 20)
    else:
        duration = random.randint(12, 30)

    title = random.choice(TITLE_TEMPLATES).format(topic=topic)

    s1, s2, s3 = random.sample(skills, 3)
    description = random.choice(DESC_TEMPLATES).format(skill_a=s1, skill_b=s2, skill_c=s3)

    # Add a bit more MOOC-like richness for specific topics
    extra = []
    if "Recommender Systems" in topic:
        extra.append("We also discuss cold-start challenges and how to evaluate ranking quality using Recall@k and NDCG@k.")
    if "Machine Learning" in topic or "Deep Learning" in topic:
        extra.append("You will practice using train/validation splits and cross-validation to estimate generalisation performance.")
    if "Product Analytics" in topic:
        extra.append("You will learn how to define success metrics and run simple A/B tests responsibly.")
    if extra:
        description += " " + " ".join(extra)

    tags = [topic] + random.sample(skills, 2)
    return {
        "course_id": i,
        "title": title,
        "description": description,
        "tags": ", ".join(tags),
        "level": level,
        "duration_hours": duration,
        "format": fmt,
    }


def main(n: int = 120) -> None:
    """Write `n` synthetic courses to the output CSV.

    The function ensures the parent directory exists and writes a CSV
    with a header row suitable for downstream loading by
    `src.kimyguide.data.loader.load_courses`.
    """
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [make_course(i) for i in range(1, n + 1)]
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {n} courses to: {OUT_PATH}")


if __name__ == "__main__":
    main()
