"""Crawl CTU's official major pages and safely refresh ctu_majors.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LISTING_URL = "https://tuyensinh.ctu.edu.vn/gioi-thieu-nganh.html?limit=0"
OFFICIAL_HOST = "tuyensinh.ctu.edu.vn"
DEFAULT_OUTPUT = Path(__file__).with_name("ctu_majors.json")
ARTICLE_SELECTOR = ".article-content"


def build_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "CTU-Major-Knowledge-Refresh/1.0 "
                "(academic project; contact: repository maintainer)"
            )
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def canonical_url(raw_url: str) -> str | None:
    parsed = urlparse(urljoin(LISTING_URL, raw_url))
    if parsed.hostname != OFFICIAL_HOST or "/gioi-thieu-nganh/" not in parsed.path:
        return None
    clean_path = re.sub(r"/{2,}", "/", parsed.path)
    return urlunparse(("https", OFFICIAL_HOST, clean_path, "", "", ""))


def discover_major_pages(session: requests.Session, timeout: float) -> list[tuple[str, str]]:
    response = session.get(LISTING_URL, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    discovered: dict[str, str] = {}
    for anchor in soup.select('a[href*="/gioi-thieu-nganh/"]'):
        url = canonical_url(anchor.get("href", ""))
        if not url:
            continue
        label = " ".join(anchor.get_text(" ", strip=True).split())
        discovered.setdefault(url, label)
    return sorted(discovered.items(), key=lambda item: item[0])


def normalize_article(article) -> str:
    for unwanted in article.select("script, style, noscript, iframe"):
        unwanted.decompose()
    for br in article.find_all("br"):
        br.replace_with("\n")
    for block in article.select("h1, h2, h3, h4, p, li, table, div"):
        block.append("\n")
    lines = []
    for line in article.get_text(" ").splitlines():
        normalized = re.sub(r"[ \t\xa0]+", " ", line).strip()
        if normalized and (not lines or normalized != lines[-1]):
            lines.append(normalized)
    return "\n".join(lines)


def extract_name(content: str, listing_label: str, soup: BeautifulSoup) -> str:
    match = re.search(
        r"(?:Tên ngành|Tên chuyên ngành)\s*:\s*([^\n]+)",
        content,
        flags=re.IGNORECASE,
    )
    if match:
        value = re.split(r"\s{2,}|Mã ngành\s*:", match.group(1), maxsplit=1)[0]
        return value.strip(" .-–")
    heading = soup.select_one("h1, .page-header h2, .item-title")
    fallback = heading.get_text(" ", strip=True) if heading else listing_label
    return " ".join(fallback.split()) or "Chưa xác định"


def fetch_major(
    session: requests.Session,
    url: str,
    listing_label: str,
    timeout: float,
) -> dict[str, str]:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    article = soup.select_one(ARTICLE_SELECTOR)
    if article is None:
        raise ValueError(f"Không tìm thấy vùng nội dung {ARTICLE_SELECTOR}")
    content = normalize_article(article)
    if len(content) < 200:
        raise ValueError("Nội dung bài viết quá ngắn, có thể cấu trúc trang đã đổi")
    return {
        "ten_nganh": extract_name(content, listing_label, soup),
        "url": url,
        "noi_dung": content,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} phải chứa một JSON array")
    return data


def merge_records(existing: list[dict], crawled: list[dict]) -> list[dict]:
    previous_by_url = {item.get("url"): item for item in existing if item.get("url")}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    merged = []
    crawled_urls = set()
    for fresh in crawled:
        url = fresh["url"]
        crawled_urls.add(url)
        previous = previous_by_url.get(url, {})
        previous_hash = previous.get("content_sha256") or hashlib.sha256(
            str(previous.get("noi_dung", "")).encode("utf-8")
        ).hexdigest()
        if previous and previous_hash == fresh["content_sha256"]:
            unchanged = dict(previous)
            unchanged.setdefault("content_sha256", fresh["content_sha256"])
            unchanged.setdefault("updated_at", now)
            merged.append(unchanged)
        else:
            merged.append({**fresh, "updated_at": now})

    # Do not lose a known page because a single weekly request failed.
    merged.extend(
        item for url, item in previous_by_url.items() if url not in crawled_urls
    )
    return sorted(merged, key=lambda item: (item.get("ten_nganh", ""), item.get("url", "")))


def atomic_write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(records, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Crawl and validate without writing")
    parser.add_argument("--max-pages", type=int, help="Limit pages for a dry-run smoke test")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--delay", type=float, default=0.15, help="Polite delay between pages")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.max_pages and not args.dry_run:
        print("--max-pages chỉ được dùng cùng --dry-run để tránh ghi đè dữ liệu một phần", file=sys.stderr)
        return 2

    existing = load_existing(args.output)
    session = build_session()
    pages = discover_major_pages(session, args.timeout)
    if args.max_pages:
        pages = pages[: args.max_pages]
    print(f"Đã phát hiện {len(pages)} trang ngành từ nguồn CTU chính thức.")

    crawled = []
    failures = []
    for index, (url, label) in enumerate(pages, start=1):
        try:
            crawled.append(fetch_major(session, url, label, args.timeout))
            print(f"[{index}/{len(pages)}] OK {url}")
        except Exception as error:  # keep the previous record when one page is unavailable
            failures.append((url, str(error)))
            print(f"[{index}/{len(pages)}] ERROR {url}: {error}", file=sys.stderr)
        if args.delay:
            time.sleep(args.delay)

    minimum_success = max(80, int(len(pages) * 0.7)) if not args.max_pages else len(pages)
    if len(crawled) < minimum_success:
        print(
            f"Dừng an toàn: chỉ tải được {len(crawled)}/{len(pages)} trang; "
            f"cần tối thiểu {minimum_success}.",
            file=sys.stderr,
        )
        return 1

    merged = merge_records(existing, crawled)
    print(
        f"Hợp nhất: {len(existing)} bản ghi cũ -> {len(merged)} bản ghi; "
        f"{len(failures)} lỗi được giữ lại từ dữ liệu cũ."
    )
    if args.dry_run:
        print("Dry-run hoàn tất, không ghi file.")
    else:
        atomic_write(args.output, merged)
        print(f"Đã cập nhật {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
