from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]  # project root
RAW = ROOT / "data" / "raw" / "mooccube" / "MOOCCube"
OUT = ROOT / "data" / "processed" / "mooccube_courses.csv"

# def _load_json(path: Path) -> Any:
#     """
#     Load either:
#       - standard JSON (single object / list)
#       - JSON Lines (one JSON object per line)
#     """
#     with path.open("r", encoding="utf-8") as f:
#         first = f.read(1)
#         f.seek(0)

#         # If it starts with '[' or '{' it's probably normal JSON
#         if first in ("[", "{"):
#             return json.load(f)

#         # Otherwise treat as JSONL
#         # rows = []
#         # for line in f:
#         #     line = line.strip()
#         #     if not line:
#         #         continue
#         #     rows.append(json.loads(line))
#         # return rows

#         rows = []
#         bad = 0
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 line = line.replace("\x00", "")
#                 rows.append(json.loads(line, strict=False))
#             except json.JSONDecodeError:
#                 bad += 1
#                 # skip bad lines (we'll report how many)
#                 continue

#         if bad:
#             print(f"[WARN] Skipped {bad} invalid JSON lines in {path.name}")
#         return rows


def _load_json(path: Path) -> Any:
    """
    Load either:
    - a standard JSON file (single object or array)
    - or a JSON Lines file (one JSON object per line)

    MOOCCube entity files like course.json are JSONL.
    """
    # Try: standard JSON first (object or array)
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        pass  # fall back to JSONL

    # Fallback: JSON Lines
    rows: List[Dict] = []
    bad = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                line = line.replace("\x00", "")
                rows.append(json.loads(line, strict=False))
            except json.JSONDecodeError:
                bad += 1
                continue

    if bad:
        print(f"[WARN] Skipped {bad} invalid JSON lines in {path}")

    return rows



def build_mooccube_courses(
    raw_dir: Path = RAW,
    out_path: Path = OUT,
    max_courses: int | None = None,
) -> Path:
    """
    Build a lightweight courses table from MOOCCube JSON files.

    Output columns:
      - course_id, title, description, tags, text
    """
    entities_dir = raw_dir / "entities"
    relations_dir = raw_dir / "relations"

    course_path = entities_dir / "course.json"
    concept_path = entities_dir / "concept.json"
    course_concept_path = relations_dir / "course-concept.json"

    if not course_path.exists():
        raise FileNotFoundError(f"Missing: {course_path}")
    if not concept_path.exists():
        raise FileNotFoundError(f"Missing: {concept_path}")
    if not course_concept_path.exists():
        raise FileNotFoundError(f"Missing: {course_concept_path}")

    courses = _load_json(course_path)
    concepts = _load_json(concept_path)
    course_concepts = _load_json(course_concept_path)

    # --- concept_id -> concept_name ---
    concept_name: Dict[str, str] = {}
    for c in concepts:
        cid = str(c.get("id") or c.get("_id") or c.get("concept_id") or "")
        name = str(c.get("name") or c.get("concept_name") or "").strip()
        if cid and name:
            concept_name[cid] = name

    # --- course_id -> [concept_name, ...] ---
    course_to_tags: Dict[str, List[str]] = {}
    for rel in course_concepts:
        # Common patterns: {"course_id": "...", "concept_id": "..."} or {"head":..., "tail":...}
        course_id = str(rel.get("course_id") or rel.get("head") or rel.get("h") or "")
        concept_id = str(rel.get("concept_id") or rel.get("tail") or rel.get("t") or "")
        if not course_id or not concept_id:
            continue
        name = concept_name.get(concept_id)
        if not name:
            continue
        course_to_tags.setdefault(course_id, []).append(name)

    rows = []
    for item in courses[: max_courses or len(courses)]:
        course_id = str(item.get("id") or item.get("_id") or item.get("course_id") or "").strip()
        title = str(item.get("name") or item.get("title") or "").strip()

        # Some versions store description/introduction under different keys.
        desc = (
            item.get("description")
            or item.get("intro")
            or item.get("introduction")
            or item.get("about")
            or ""
        )
        description = str(desc).strip()

        tags_list = course_to_tags.get(course_id, [])
        # Deduplicate tags while preserving order
        seen = set()
        tags_list = [t for t in tags_list if not (t in seen or seen.add(t))]
        tags = ", ".join(tags_list[:20])  # cap for readability

        text = " ".join([title, description, tags]).strip()

        if not course_id or not title:
            continue

        rows.append(
            {
                "course_id": course_id,
                "title": title,
                "description": description,
                "tags": tags,
                "text": text,
            }
        )

    df = pd.DataFrame(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    p = build_mooccube_courses()
    print(f"Wrote: {p}")
