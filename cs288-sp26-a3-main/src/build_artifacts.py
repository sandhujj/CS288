from __future__ import annotations

import argparse
import json
import re
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as xml_tree

from .html_tools import extract_page
from .sparse_index import SparsePassageIndex
from .text_utils import build_passages, clean_answer, normalize_space


allowed_hosts = {"eecs.berkeley.edu", "www2.eecs.berkeley.edu"}
ignored_suffixes = {
    ".7z",
    ".avi",
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".json",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".ps",
    ".rar",
    ".svg",
    ".tar",
    ".tgz",
    ".txt",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}
www2_seed_urls = [
    "https://www2.eecs.berkeley.edu/Courses/CS/",
    "https://www2.eecs.berkeley.edu/Courses/EE/",
    "https://www2.eecs.berkeley.edu/Faculty/Homepages/",
    "https://www2.eecs.berkeley.edu/Pubs/Faculty/",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/",
    "https://www2.eecs.berkeley.edu/Scheduling/CS/",
    "https://www2.eecs.berkeley.edu/Scheduling/EE/",
    "https://www2.eecs.berkeley.edu/Students/",
]
important_urls = [
    "https://eecs.berkeley.edu/resources/undergrads/cs/advising/",
    "https://eecs.berkeley.edu/news/graduate-student-syed-tahmid-mahbub-awarded-paul-daisy-soros-fellowship/",
    "https://eecs.berkeley.edu/people/faculty/in-memoriam/",
    "https://eecs.berkeley.edu/people/students-2/awards/",
    "https://eecs.berkeley.edu/book/faculty/",
    "https://eecs.berkeley.edu/book/phd/coursework/",
    "https://www2.eecs.berkeley.edu/Courses/CS/",
    "https://www2.eecs.berkeley.edu/Faculty/Homepages/abbeel.html",
    "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-27.html",
    "https://www2.eecs.berkeley.edu/Scheduling/EE/schedule-draft.html",
]
markdown_link_pattern = re.compile(r"\((https?://[^)\s]+)\)")


def download_text(url: str, timeout_seconds: int = 15) -> tuple[str, str]:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            content_type = response.headers.get("Content-Type", "")
            body = response.read().decode("utf-8", errors="ignore")
        return content_type, body
    except (HTTPError, URLError, TimeoutError, ValueError, socket.timeout):
        if urlparse(url).netloc == "www2.eecs.berkeley.edu":
            mirror_url = f"https://r.jina.ai/http://{urlparse(url).netloc}{urlparse(url).path}"
            mirror_request = Request(
                mirror_url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "text/plain,text/markdown,*/*",
                },
            )
            with urlopen(mirror_request, timeout=12) as response:
                body = response.read().decode("utf-8", errors="ignore")
            return "text/markdown", body
        raise


def normalize_url(candidate_url: str) -> str:
    parsed_url = urlparse(candidate_url)
    normalized_path = parsed_url.path or "/"
    if normalized_path.endswith("/index.html"):
        normalized_path = normalized_path[:-10] + "/"
    if not normalized_path:
        normalized_path = "/"
    if normalized_path != "/" and normalized_path.endswith("/"):
        normalized_path = normalized_path[:-1]
    return f"{parsed_url.scheme}://{parsed_url.netloc}{normalized_path}"


def is_allowed_url(candidate_url: str) -> bool:
    parsed_url = urlparse(candidate_url)
    if parsed_url.scheme not in {"http", "https"}:
        return False
    if parsed_url.netloc not in allowed_hosts:
        return False
    if parsed_url.query:
        return False
    if "login" in parsed_url.path.lower() or "protected" in parsed_url.path.lower():
        return False
    lowered_path = parsed_url.path.lower()
    return not any(lowered_path.endswith(suffix) for suffix in ignored_suffixes)


def extract_mirror_page(url: str, raw_markdown: str) -> dict[str, object]:
    lines = raw_markdown.splitlines()
    title = ""
    text_lines: list[str] = []
    links: list[str] = []
    in_markdown_content = False
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("Title:"):
            title = stripped_line.partition(":")[2].strip()
            continue
        if stripped_line == "Markdown Content:":
            in_markdown_content = True
            continue
        if not in_markdown_content:
            continue
        links.extend(markdown_link_pattern.findall(stripped_line))
        cleaned_line = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", stripped_line)
        cleaned_line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned_line)
        cleaned_line = re.sub(r"[#>*_`]", " ", cleaned_line)
        cleaned_line = normalize_space(cleaned_line)
        if cleaned_line:
            text_lines.append(cleaned_line)
    unique_links = list(dict.fromkeys(links))
    return {
        "url": url,
        "title": normalize_space(title),
        "text": "\n".join(text_lines),
        "links": unique_links,
    }


def load_sitemap_urls(limit: int) -> list[str]:
    sitemap_index_url = "https://eecs.berkeley.edu/sitemap_index.xml"
    _, sitemap_index_body = download_text(sitemap_index_url)
    sitemap_index_root = xml_tree.fromstring(sitemap_index_body)
    namespace = {"site": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    candidate_urls: list[str] = []
    for sitemap_element in sitemap_index_root.findall("site:sitemap", namespace):
        location = sitemap_element.findtext("site:loc", default="", namespaces=namespace)
        if not location:
            continue
        if not any(
            location.endswith(suffix)
            for suffix in (
                "page-sitemap.xml",
                "book-sitemap.xml",
                "post-sitemap.xml",
                "news_post-sitemap.xml",
                "news_post-sitemap2.xml",
                "media_mention-sitemap.xml",
            )
        ):
            continue
        _, sitemap_body = download_text(location)
        sitemap_root = xml_tree.fromstring(sitemap_body)
        for url_element in sitemap_root.findall("site:url", namespace):
            page_url = url_element.findtext("site:loc", default="", namespaces=namespace)
            if is_allowed_url(page_url):
                candidate_urls.append(normalize_url(page_url))
    unique_urls = list(dict.fromkeys(candidate_urls))
    return unique_urls[:limit]


def crawl_www2_urls(limit: int, max_depth: int = 2, workers: int = 8) -> list[str]:
    discovered_urls: list[str] = []
    visited_urls: set[str] = set()
    current_level_urls = [normalize_url(url) for url in www2_seed_urls]
    for depth in range(max_depth + 1):
        batch_urls: list[str] = []
        seen_batch_urls: set[str] = set()
        for candidate_url in current_level_urls:
            if candidate_url in visited_urls or candidate_url in seen_batch_urls:
                continue
            if not is_allowed_url(candidate_url):
                continue
            seen_batch_urls.add(candidate_url)
            batch_urls.append(candidate_url)
        if not batch_urls:
            break
        next_level_urls: list[str] = []
        with ThreadPoolExecutor(max_workers=min(workers, len(batch_urls))) as executor:
            future_by_url = {executor.submit(download_text, url): url for url in batch_urls[: limit * 4]}
            for future in as_completed(future_by_url):
                current_url = future_by_url[future]
                visited_urls.add(current_url)
                try:
                    content_type, body = future.result()
                except (HTTPError, URLError, TimeoutError, ValueError, socket.timeout):
                    continue
                if "text/html" not in content_type and "text/markdown" not in content_type:
                    continue
                page = extract_page(current_url, body) if "text/html" in content_type else extract_mirror_page(current_url, body)
                if len(str(page["text"])) < 200:
                    continue
                discovered_urls.append(current_url)
                if len(discovered_urls) >= limit:
                    return discovered_urls[:limit]
                if depth < max_depth:
                    for link in page["links"]:
                        normalized_link = normalize_url(link)
                        if normalized_link not in visited_urls and is_allowed_url(normalized_link):
                            next_level_urls.append(normalized_link)
        current_level_urls = next_level_urls[: limit * 6]
    return discovered_urls[:limit]


def fetch_page(url: str) -> dict[str, object] | None:
    try:
        content_type, body = download_text(url)
    except (HTTPError, URLError, TimeoutError, ValueError, socket.timeout):
        return None
    if "text/html" not in content_type and "text/markdown" not in content_type:
        return None
    page = extract_page(url, body) if "text/html" in content_type else extract_mirror_page(url, body)
    if len(str(page["text"])) < 200:
        return None
    return page


def page_to_passages(page: dict[str, object]) -> list[dict[str, object]]:
    page_text = str(page["text"])
    page_title = str(page["title"])
    passages: list[dict[str, object]] = []
    for passage_text in build_passages(page_text, max_words=120, overlap_words=30):
        cleaned_text = clean_answer(passage_text)
        if len(cleaned_text.split()) < 15:
            continue
        passages.append(
            {
                "url": str(page["url"]),
                "title": page_title,
                "text": cleaned_text,
            }
        )
    return passages


def build_artifacts(output_dir: Path, eecs_limit: int, www2_limit: int, workers: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sitemap_urls = load_sitemap_urls(eecs_limit)
    www2_urls = crawl_www2_urls(www2_limit)
    normalized_important_urls = [normalize_url(url) for url in important_urls]
    candidate_urls = list(dict.fromkeys(normalized_important_urls + sitemap_urls + www2_urls))
    pages: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_by_url = {executor.submit(fetch_page, url): url for url in candidate_urls}
        for future in as_completed(future_by_url):
            page = future.result()
            if page:
                pages.append(page)
    fetched_urls = {normalize_url(str(page["url"])) for page in pages}
    for url in normalized_important_urls:
        if url in fetched_urls:
            continue
        page = fetch_page(url)
        if page:
            pages.append(page)
            fetched_urls.add(url)
    pages.sort(key=lambda page: str(page["url"]))
    passages: list[dict[str, object]] = []
    for page in pages:
        passages.extend(page_to_passages(page))
    index = SparsePassageIndex.build(passages)
    corpus_path = output_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for page in pages:
            handle.write(json.dumps(page, ensure_ascii=True) + "\n")
    index.save(output_dir / "indexes" / "sparse_passages.pkl")
    metadata = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages),
        "passage_count": len(passages),
        "hosts": sorted({urlparse(str(page["url"])).netloc for page in pages}),
    }
    with (output_dir / "build_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--eecs-limit", type=int, default=700)
    parser.add_argument("--www2-limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=10)
    arguments = parser.parse_args()
    build_artifacts(
        output_dir=Path(arguments.output_dir),
        eecs_limit=arguments.eecs_limit,
        www2_limit=arguments.www2_limit,
        workers=arguments.workers,
    )


if __name__ == "__main__":
    main()
