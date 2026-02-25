from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_IN = ROOT / "data" / "raw" / "openlearn_courses_raw.csv"
OUT = ROOT / "data" / "processed" / "openlearn_courses.csv"


def clean_text(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_englishish(text: str) -> bool:
    """
    Cheap language filter:
    keep rows with enough latin letters.
    (Good enough for OpenLearn; avoids Welsh/other non-English content.)
    """
    t = str(text or "")
    letters = len(re.findall(r"[A-Za-z]", t))
    return letters >= 50  # adjust if needed


if __name__ == "__main__":
    if not RAW_IN.exists():
        raise FileNotFoundError(f"Missing {RAW_IN}. Run openlearn_collect.py first.")

    df = pd.read_csv(RAW_IN)

    # Basic cleaning
    for col in ["course_id", "title", "description", "provider", "subject", "level", "url", "tags"]:
        if col in df.columns:
            df[col] = df[col].fillna("").map(clean_text)

    # # Drop duplicates + weak rows
    # df = df.drop_duplicates(subset=["url"]).drop_duplicates(subset=["course_id"])
    # df = df[df["title"].str.len() > 5].copy()
    # df = df[df["description"].str.len() >= 60].copy()

    # Keep only real course pages (remove About/FAQ/newsletter/get-started pages, etc.)
    course_url_mask = df["url"].str.contains(r"/openlearn/", case=False, na=False)

    # Drop obvious non-course sections
    non_course_mask = df["url"].str.contains(
        r"/about-openlearn/|/get-started/|/subscribe|/news|/faq|/frequently-asked|/about-us|/sitemap|/contact",
        case=False,
        na=False,
    )

    df = df[course_url_mask & ~non_course_mask].copy()

    # Optional but recommended: enforce course-like structure
    course_like = df["url"].str.contains(r"content-section-", case=False, na=False) | (
        ~df["url"].str.contains(r"/about|/get-started|/news|/faq|/subscribe", case=False, na=False)
    )
    df = df[course_like].copy()

    # Now do quality filters + dedupe
    df = df.drop_duplicates(subset=["url"]).drop_duplicates(subset=["course_id"])
    df = df[df["title"].str.len() > 5].copy()
    df = df[df["description"].str.len() >= 60].copy()

    # English-ish filter (optional but helps)
    df = df[df["description"].map(is_englishish)].copy()

    # Build fields your API expects
    if "tags" not in df.columns:
        df["tags"] = ""
    df["tags"] = df["tags"].fillna("")

    df["text"] = (df["title"] + " " + df["description"] + " " + df["tags"]).map(clean_text)

    # Keep a clean schema for modeling
    out = df[
        ["course_id", "title", "description", "tags", "text",
         "provider", "subject", "level", "duration_hours", "url"]
    ].copy()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"[DONE] wrote {OUT} rows={len(out)}")
    print("[INFO] sample:")
    print(out.head(3)[["course_id", "title", "subject", "level", "duration_hours"]].to_string(index=False))
