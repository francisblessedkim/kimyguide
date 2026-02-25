from __future__ import annotations

import re
import time
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import pandas as pd
from bs4 import BeautifulSoup


BASE = "https://www.open.edu"
CATALOGUE_URL = "https://www.open.edu/openlearn/free-courses/full-catalogue?page={page}"

ROOT = Path(__file__).resolve().parents[1]  # project root
RAW_OUT = ROOT / "data" / "raw" / "openlearn_courses_raw.csv"

HEADERS = {
    "User-Agent": "KimyGuideDatasetCuration/1.0 (academic project; polite crawl)",
    "Accept-Language": "en-GB,en;q=0.9",
}


@dataclass
class CourseRow:
    course_id: str
    title: str
    description: str
    provider: str
    subject: str
    level: str
    duration_hours: Optional[float]
    url: str
    tags: str


def stable_id(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"openlearn_{h}"


def clean_text(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_html(url: str, session: requests.Session) -> str:
    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def get_soup(url: str, session: requests.Session) -> BeautifulSoup:
    return BeautifulSoup(get_html(url, session), "html.parser")


def parse_duration_hours(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*hrs?", text.lower())
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def normalize_to_course_landing(url: str) -> str:
    """
    Turn:
      .../content-section-0
      .../content-section-overview
    into:
      .../<topic>/<slug>
    because landing pages are more likely to contain summary metadata.
    """
    url = url.split("#")[0].split("?")[0].rstrip("/")
    m = re.match(r"^(https?://www\.open\.edu/openlearn/[^/]+/[^/]+)(/content-section.*)?$", url)
    if m:
        return m.group(1)
    return url


def looks_like_course_path(path: str) -> bool:
    # Exclude known non-course areas
    banned_prefixes = [
        "/openlearn/theme/",
        "/openlearn/pluginfile.php",
        "/openlearn/mod/",
        "/openlearn/local/",
        "/openlearn/blocks/",
        "/openlearn/lib/",
        "/openlearn/webservice/",
        "/openlearn/user/",
        "/openlearn/login/",
        "/openlearn/enrol/",
        "/openlearn/message/",
        "/openlearn/course/",
        "/openlearn/my/",
        "/openlearn/about-openlearn/",
    ]
    if any(path.startswith(p) for p in banned_prefixes):
        return False
    if "/free-courses" in path or "/search" in path:
        return False

    parts = [x for x in path.split("/") if x]
    if len(parts) < 3:
        return False

    topic = parts[1]
    slug = parts[2]

    if not re.fullmatch(r"[a-z0-9\-]+", slug):
        return False

    return True


def extract_paths_from_catalogue(html: str) -> List[str]:
    paths = re.findall(r"(/openlearn/[^\s\"'<>]+)", html)

    seen = set()
    uniq = []
    for p in paths:
        p = p.split("#")[0].split("?")[0].rstrip(".,);]\"'")
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    # Keep only paths that look like courses
    out = []
    for p in uniq:
        p = p.rstrip("/")
        if looks_like_course_path(p):
            out.append(p)
    return out


def extract_catalogue_courses(page_url: str, session: requests.Session) -> List[str]:
    html = get_html(page_url, session)
    paths = extract_paths_from_catalogue(html)

    # Convert to full URLs, normalize to landing pages
    urls = [normalize_to_course_landing(urljoin(BASE, p)) for p in paths]

    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def extract_jsonld_description(soup: BeautifulSoup) -> str:
    """
    Try JSON-LD blocks first (often contains description).
    """
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.get_text(strip=True) or "{}")
        except Exception:
            continue

        # JSON-LD can be dict or list
        candidates = []
        if isinstance(data, dict):
            candidates.append(data)
        elif isinstance(data, list):
            candidates.extend([d for d in data if isinstance(d, dict)])

        for d in candidates:
            desc = d.get("description")
            if desc and len(str(desc)) >= 60:
                return clean_text(desc)

    return ""


def parse_course_page(course_url: str, session: requests.Session) -> Dict:
    soup = get_soup(course_url, session)

    # Title
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = clean_text(h1.get_text(" ", strip=True))

    # Description priority:
    # 1) JSON-LD description
    desc = extract_jsonld_description(soup)

    # 2) OpenGraph description
    if not desc:
        og = soup.find("meta", attrs={"property": "og:description"})
        if og and og.get("content"):
            d = clean_text(og["content"])
            if len(d) >= 60:
                desc = d

    # 3) meta description
    if not desc:
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            d = clean_text(meta["content"])
            if len(d) >= 60:
                desc = d

    # 4) first meaningful paragraphs in <main>
    if not desc:
        main = soup.find("main")
        if main:
            ps = main.find_all("p")
            chunks = [clean_text(p.get_text(" ", strip=True)) for p in ps[:12]]
            chunks = [c for c in chunks if len(c) >= 60]
            if chunks:
                # take first long paragraph; or join first two
                desc = chunks[0]
                if len(desc) < 140 and len(chunks) > 1:
                    desc = (desc + " " + chunks[1]).strip()
                desc = desc[:1600]

    # Subject from URL (/openlearn/<subject>/...)
    subject = ""
    m = re.match(r"^https?://www\.open\.edu/openlearn/([^/]+)/", course_url)
    if m:
        subject = clean_text(m.group(1).replace("-", " "))

    # Level (best-effort keywords)
    level = ""
    page_text = soup.get_text(" ", strip=True)
    for lv in ["Introductory", "Intermediate", "Advanced"]:
        if re.search(rf"\b{lv}\b", page_text):
            level = lv
            break

    # Duration (hrs)
    duration_hours = parse_duration_hours(page_text)

    return {
        "title": title,
        "description": desc,
        "subject": subject,
        "level": level,
        "duration_hours": duration_hours,
    }


def collect_openlearn(max_pages: int = 120, sleep_s: float = 1.0) -> pd.DataFrame:
    session = requests.Session()

    # ---- collect landing URLs ----
    all_urls: List[str] = []
    seen = set()

    for page in range(1, max_pages + 1):
        page_url = CATALOGUE_URL.format(page=page)
        urls = extract_catalogue_courses(page_url, session)
        if not urls:
            print(f"[CAT] page={page} found=0 -> stopping")
            break

        added = 0
        for u in urls:
            if u not in seen:
                seen.add(u)
                all_urls.append(u)
                added += 1

        print(f"[CAT] page={page} found={len(urls)} added={added} total_unique={len(all_urls)}")
        if page == 1:
            print("[DEBUG] sample urls:", all_urls[:10])

        if added == 0:
            break

        time.sleep(sleep_s)

    # ---- enrich pages ----
    rows: List[CourseRow] = []
    total = len(all_urls)
    print(f"[INFO] Enriching {total} course landing pages...")

    for i, url in enumerate(all_urls, start=1):
        try:
            info = parse_course_page(url, session)
        except Exception as e:
            if i <= 20:
                print("[SKIP]", url, "->", e)
            continue

        title = info.get("title") or ""
        desc = info.get("description") or ""

        # Skip clearly non-course or empty pages early
        if (not title) or (len(desc) < 60):
            continue

        rows.append(
            CourseRow(
                course_id=stable_id(url),
                title=title,
                description=desc,
                provider="OpenLearn (The Open University)",
                subject=info.get("subject") or "",
                level=info.get("level") or "",
                duration_hours=info.get("duration_hours"),
                url=url,
                tags="",
            )
        )

        if i % 25 == 0:
            print(f"[ENRICH] {i}/{total} kept={len(rows)}")

        time.sleep(sleep_s)

    return pd.DataFrame([asdict(r) for r in rows])


if __name__ == "__main__":
    df = collect_openlearn(max_pages=120, sleep_s=1.0)

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_OUT, index=False)
    print(f"[DONE] wrote {RAW_OUT} rows={len(df)}")
