from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import urljoin

from .text_utils import clean_text, normalize_space


skip_tags = {"script", "style", "noscript", "svg", "path", "img", "picture", "video", "audio", "iframe"}
block_tags = {
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dt",
    "figure",
    "figcaption",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "section",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
}
decorative_markers = {
    "breadcrumb",
    "cookie",
    "footer",
    "header",
    "menu",
    "nav",
    "pagination",
    "search",
    "share",
    "sidebar",
    "social",
    "subscribe",
}


class PageParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[str] = []
        self.text_parts: list[str] = []
        self.title_parts: list[str] = []
        self.skip_depth = 0
        self.title_depth = 0

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        attribute_map = {key: value or "" for key, value in attributes}
        if tag == "a":
            href = attribute_map.get("href", "").strip()
            if href:
                self.links.append(urljoin(self.base_url, href))
        if self.skip_depth:
            self.skip_depth += 1
            return
        if tag in skip_tags or self._is_decorative(attribute_map):
            self.skip_depth = 1
            return
        if tag == "title":
            self.title_depth += 1
        if tag in block_tags:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth:
            self.skip_depth -= 1
            return
        if tag == "title" and self.title_depth:
            self.title_depth -= 1
        if tag in block_tags:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        text = normalize_space(data)
        if not text:
            return
        self.text_parts.append(text)
        self.text_parts.append(" ")
        if self.title_depth:
            self.title_parts.append(text)

    def _is_decorative(self, attribute_map: dict[str, str]) -> bool:
        attribute_blob = " ".join(
            attribute_map.get(key, "") for key in {"class", "id", "role", "aria-label", "data-testid"}
        ).lower()
        return any(marker in attribute_blob for marker in decorative_markers)


def extract_page(url: str, raw_html: str) -> dict[str, object]:
    parser = PageParser(url)
    parser.feed(raw_html)
    parser.close()
    raw_text = clean_text("".join(parser.text_parts))
    text_lines: list[str] = []
    seen_lines: set[str] = set()
    for line in raw_text.splitlines():
        normalized_line = normalize_space(line)
        if not normalized_line:
            continue
        if normalized_line in seen_lines:
            continue
        seen_lines.add(normalized_line)
        text_lines.append(normalized_line)
    text = "\n".join(text_lines)
    title = normalize_space(" ".join(parser.title_parts))
    unique_links: list[str] = []
    seen_links: set[str] = set()
    for link in parser.links:
        if link not in seen_links:
            seen_links.add(link)
            unique_links.append(link)
    return {"url": url, "title": title, "text": text, "links": unique_links}
