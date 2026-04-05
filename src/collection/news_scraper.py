from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable
import csv
import json
import os
import re
import time
from time import perf_counter
from urllib.parse import quote, urljoin
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup


ParserFunc = Callable[[BeautifulSoup], str]
MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "Â ": " ",
    "\xa0": " ",
}


@dataclass
class ApiPaginationConfig:
    url: str
    method: str = "POST"
    start_offset: int = 0
    step: int = 20
    build_payload: Callable[[int], dict[str, Any]] = field(default_factory=lambda: (lambda offset: {"offset": offset}))
    extract_edges: Callable[[dict[str, Any]], list[dict[str, Any]]] = field(
        default_factory=lambda: (lambda data: data.get("posts", {}).get("edges", []))
    )
    edge_url_keys: tuple[str, ...] = ("uri", "link")
    edge_date_key: str = "date"
    max_pages: int = 200
    sleep_seconds: float = 0.25


@dataclass
class SiteConfig:
    name: str
    base_url: str
    list_urls: list[str]
    allowed_url_regex: str
    category_label: str = ""
    article_title_selector: str = "h1"
    article_body_selector: str = "article p, p"
    date_from_url_regex: str | None = None
    request_headers: dict[str, str] = field(default_factory=lambda: {"User-Agent": "Mozilla/5.0"})
    timeout_seconds: int = 20
    api_pagination: ApiPaginationConfig | None = None
    sitemap_urls: list[str] = field(default_factory=list)
    body_parser: ParserFunc | None = None
    blocked_url_regexes: list[str] = field(default_factory=list)

    def allowed_url_pattern(self) -> re.Pattern[str]:
        return re.compile(self.allowed_url_regex)

    def date_url_pattern(self) -> re.Pattern[str] | None:
        if not self.date_from_url_regex:
            return None
        return re.compile(self.date_from_url_regex)


def parse_iso_datetime(dt_str: str | None) -> datetime | None:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_api_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        dt = parse_iso_datetime(value)
        if dt is not None:
            return dt
        if value.strip().isdigit():
            try:
                value = int(value.strip())
            except ValueError:
                return None
        else:
            return None
    if isinstance(value, (int, float)):
        ts = float(value)
        # Treat values above ~year 33658 in seconds as milliseconds.
        if ts > 1_000_000_000_000:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts)
        except (OverflowError, OSError, ValueError):
            return None
    return None


def get_date_from_url(url: str, pattern: re.Pattern[str] | None) -> date | None:
    if pattern is None:
        return None
    match = pattern.search(url)
    if not match:
        return None
    try:
        year, month, day = map(int, match.groups()[:3])
        return date(year, month, day)
    except (ValueError, IndexError):
        return None


def extract_text_default(soup: BeautifulSoup, selector: str) -> str:
    blocks = [node.get_text(" ", strip=True) for node in soup.select(selector)]
    return "\n".join(line for line in blocks if line)


def normalize_text(text: str) -> str:
    fixed = text
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        fixed = fixed.replace(bad, good)
    return re.sub(r"\s+", " ", fixed).strip()


def clean_article_body(text: str) -> str:
    if not text:
        return ""
    normalized = normalize_text(text)
    line_seen: set[str] = set()
    lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = normalize_text(raw_line)
        if not line:
            continue
        low = line.lower()
        if low == "related" or low.startswith("related "):
            continue
        if line in line_seen:
            continue
        line_seen.add(line)
        lines.append(line)

    sentence_seen: set[str] = set()
    sentences: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(lines)):
        part = normalize_text(sentence)
        if not part:
            continue
        key = part.lower()
        if key in sentence_seen:
            continue
        sentence_seen.add(key)
        sentences.append(part)
    return " ".join(sentences)


def extract_published_date_from_soup(soup: BeautifulSoup) -> str:
    probes = [
        ('meta[property="article:published_time"]', "content"),
        ('meta[name="article:published_time"]', "content"),
        ('meta[property="og:published_time"]', "content"),
        ('meta[name="publish-date"]', "content"),
        ('meta[name="pubdate"]', "content"),
        ('meta[name="date"]', "content"),
        ('meta[itemprop="datePublished"]', "content"),
        ("time[datetime]", "datetime"),
    ]
    for selector, attr in probes:
        node = soup.select_one(selector)
        if not node:
            continue
        raw = node.get(attr)
        dt = parse_api_datetime(raw)
        if dt is not None:
            return dt.date().isoformat()
    return ""


def resolve_cutoff_date(months_back: int, start_date: date | str | None = None) -> date:
    if isinstance(start_date, date):
        return start_date
    if isinstance(start_date, str):
        return date.fromisoformat(start_date)
    return date.today() - timedelta(days=30 * months_back)


def scrape_article(url: str, config: SiteConfig) -> dict[str, str] | None:
    response: requests.Response | None = None
    for attempt in range(4):
        try:
            response = requests.get(
                url,
                headers=config.request_headers,
                timeout=config.timeout_seconds,
                allow_redirects=True,
            )
            if response.encoding is None or response.encoding.lower() in {"iso-8859-1", "latin-1"}:
                if response.apparent_encoding:
                    response.encoding = response.apparent_encoding
            if response.status_code == 429:
                wait_seconds = min(30.0, 2.5 * (attempt + 1))
                print(f"[{config.name}] 429 on article {url}; retrying in {wait_seconds:.1f}s")
                time.sleep(wait_seconds)
                continue
            if response.status_code in {403, 404}:
                print(f"[{config.name}] Skipping article {url}: HTTP {response.status_code}")
                response = None
                break
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == 3:
                print(f"[{config.name}] Skipping article {url}: {exc}")
                response = None
            else:
                time.sleep(1.0 + attempt)
    if response is None:
        return None

    final_url = response.url or url
    allowed = config.allowed_url_pattern()
    if not allowed.search(final_url):
        return None
    for blocked_pattern in config.blocked_url_regexes:
        if re.search(blocked_pattern, final_url):
            return None

    soup = BeautifulSoup(response.text, "html.parser")
    title_node = soup.select_one(config.article_title_selector)
    title = title_node.get_text(" ", strip=True) if title_node else ""
    if not title:
        og_title = soup.select_one('meta[property="og:title"]')
        if og_title and og_title.get("content"):
            title = str(og_title.get("content")).strip()
        elif soup.title:
            title = soup.title.get_text(" ", strip=True)
    title = normalize_text(title)

    if config.body_parser is not None:
        body = config.body_parser(soup)
    else:
        body = extract_text_default(soup, config.article_body_selector)
    body = clean_article_body(body)

    url_date = get_date_from_url(url, config.date_url_pattern())
    published_date = url_date.isoformat() if url_date else extract_published_date_from_soup(soup)
    return {
        "source": config.name,
        "category": config.category_label,
        "url": final_url,
        "title": title,
        "date": published_date,
        "body": body,
    }


def get_links_from_list_pages(config: SiteConfig, cutoff_date: date | None = None) -> set[str]:
    allowed = config.allowed_url_pattern()
    date_pattern = config.date_url_pattern()
    links: set[str] = set()
    total_pages = len(config.list_urls)
    stagnant_pages = 0
    stagnant_limit = 30
    old_pages = 0
    old_pages_limit = 5
    for idx, list_url in enumerate(config.list_urls, start=1):
        before_count = len(links)
        response: requests.Response | None = None
        for attempt in range(4):
            try:
                response = requests.get(list_url, headers=config.request_headers, timeout=config.timeout_seconds)
                if response.status_code == 429:
                    wait_seconds = min(20.0, 2.0 * (attempt + 1))
                    print(f"[{config.name}] 429 on {list_url}; retrying in {wait_seconds:.1f}s")
                    time.sleep(wait_seconds)
                    continue
                if response.status_code in {403, 404}:
                    print(f"[{config.name}] Skipping list page {list_url}: HTTP {response.status_code}")
                    response = None
                    break
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == 3:
                    print(f"[{config.name}] List page failed {list_url}: {exc}")
                    response = None
                else:
                    time.sleep(1.0 + attempt)
        if response is None:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        page_dates: list[date] = []
        for anchor in soup.find_all("a"):
            href = anchor.get("href") or ""
            if not href:
                continue
            full_url = urljoin(config.base_url, href)
            full_url = full_url.split("#", 1)[0].strip()
            if allowed.search(full_url):
                is_blocked = any(re.search(p, full_url) for p in config.blocked_url_regexes)
                if is_blocked:
                    continue
                links.add(full_url)
                url_date = get_date_from_url(full_url, date_pattern)
                if url_date is not None:
                    page_dates.append(url_date)

        if len(links) == before_count:
            stagnant_pages += 1
        else:
            stagnant_pages = 0
        if cutoff_date is not None and page_dates:
            newest_on_page = max(page_dates)
            if newest_on_page < cutoff_date:
                old_pages += 1
            else:
                old_pages = 0

        # Keep a modest pace to reduce rate limiting.
        time.sleep(0.2)
        if idx == 1 or idx % 25 == 0 or idx == total_pages:
            print(
                f"[{config.name}] List page progress: {idx}/{total_pages} "
                f"(unique candidate links: {len(links)})"
            )
        if stagnant_pages >= stagnant_limit:
            print(
                f"[{config.name}] No new links for {stagnant_limit} consecutive pages; "
                "stopping list-page scan early"
            )
            break
        if old_pages >= old_pages_limit:
            print(
                f"[{config.name}] Newest URL date is older than cutoff for {old_pages_limit} "
                "consecutive pages; stopping list-page scan early"
            )
            break

    print(f"[{config.name}] HTML list pages yielded {len(links)} unique links")
    return links


def _extract_edge_url(edge: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    node = edge.get("node", edge)
    for key in keys:
        value = node.get(key)
        if value:
            return str(value)
    return None


def _sitemap_month_from_url(url: str) -> date | None:
    match = re.search(r"\.(\d{4})(\d{2})\.xml(?:\.gz)?$", url)
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2))
    try:
        return date(year, month, 1)
    except ValueError:
        return None


def _is_old_sitemap_month(sitemap_url: str, cutoff_date: date | None) -> bool:
    if cutoff_date is None:
        return False
    month_date = _sitemap_month_from_url(sitemap_url)
    if month_date is None:
        return False
    cutoff_month = date(cutoff_date.year, cutoff_date.month, 1)
    return month_date < cutoff_month


def _is_known_bad_sitemap(sitemap_url: str) -> bool:
    return sitemap_url.rstrip("/").endswith("sitemap-root.xml")


def fetch_links_from_sitemaps(
    config: SiteConfig,
    cutoff_date: date | None = None,
    max_links: int | None = None,
) -> set[str]:
    if not config.sitemap_urls:
        return set()

    allowed = config.allowed_url_pattern()
    date_pattern = config.date_url_pattern()
    links: set[str] = set()
    to_visit = list(config.sitemap_urls)
    visited: set[str] = set()

    while to_visit:
        sitemap_url = to_visit.pop(0)
        if sitemap_url in visited:
            continue
        visited.add(sitemap_url)
        if _is_known_bad_sitemap(sitemap_url):
            continue
        if _is_old_sitemap_month(sitemap_url, cutoff_date):
            continue

        try:
            response = requests.get(sitemap_url, headers=config.request_headers, timeout=config.timeout_seconds)
            response.raise_for_status()
            root = ET.fromstring(response.text)
        except (requests.RequestException, ET.ParseError) as exc:
            print(f"[{config.name}] Sitemap fetch/parse failed {sitemap_url}: {exc}")
            continue

        child_sitemaps = [
            node.text.strip()
            for node in root.findall(".//{*}sitemap/{*}loc")
            if node.text and node.text.strip()
        ]
        if child_sitemaps:
            for child in child_sitemaps:
                if (
                    child not in visited
                    and not _is_known_bad_sitemap(child)
                    and not _is_old_sitemap_month(child, cutoff_date)
                ):
                    to_visit.append(child)
            continue

        url_entries = [
            node.text.strip()
            for node in root.findall(".//{*}url/{*}loc")
            if node.text and node.text.strip()
        ]
        for entry in url_entries:
            full_url = urljoin(config.base_url, entry)
            is_blocked = any(re.search(p, full_url) for p in config.blocked_url_regexes)
            if not allowed.search(full_url) or is_blocked:
                continue
            if cutoff_date is not None:
                url_date = get_date_from_url(full_url, date_pattern)
                if url_date is not None and url_date < cutoff_date:
                    continue
            links.add(full_url)
            if max_links is not None and len(links) >= max_links:
                print(f"[{config.name}] Sitemap links reached max_links={max_links}")
                return links

    print(f"[{config.name}] Sitemaps yielded {len(links)} unique links")
    return links


def extract_malaysiakini_stories(data: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []

    stories = data.get("stories")
    if not isinstance(stories, list):
        page_props = data.get("pageProps")
        if not isinstance(page_props, dict):
            props = data.get("props")
            if isinstance(props, dict):
                page_props = props.get("pageProps")
        if not isinstance(page_props, dict):
            return []
        stories = page_props.get("stories", [])
    if not isinstance(stories, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in stories:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        sid = row.get("sid")
        if sid and not any(row.get(key) for key in ("url", "link", "alias", "path")):
            row["alias"] = f"/news/{sid}"
        normalized.append(row)
    return normalized


def extract_next_data_json(html_text: str) -> dict[str, Any] | None:
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html_text, re.S)
    if not match:
        return None
    payload = match.group(1).strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def fetch_links_from_api(
    config: SiteConfig,
    cutoff_date: date | None = None,
    max_links: int | None = None,
) -> set[str]:
    api = config.api_pagination
    if api is None:
        return set()

    allowed = config.allowed_url_pattern()
    all_links: set[str] = set()
    offset = api.start_offset
    pages = 0

    while pages < api.max_pages:
        payload = api.build_payload(offset)
        api_url = api.url.format(offset=offset)
        try:
            if api.method.upper() == "GET":
                response = requests.get(
                    api_url,
                    params=payload,
                    headers=config.request_headers,
                    timeout=config.timeout_seconds,
                )
            else:
                response = requests.post(
                    api_url,
                    json=payload,
                    headers=config.request_headers,
                    timeout=config.timeout_seconds,
                )
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError:
                next_data = extract_next_data_json(response.text)
                if next_data is None:
                    raise
                data = next_data
            if (
                api.method.upper() == "GET"
                and isinstance(data, dict)
                and data.get("recaptchaRequired")
                and data.get("recaptchaPublickey")
                and "captchaHash" not in payload
            ):
                payload = dict(payload)
                payload["captchaHash"] = str(data.get("recaptchaPublickey"))
                response = requests.get(
                    api_url,
                    params=payload,
                    headers=config.request_headers,
                    timeout=config.timeout_seconds,
                )
                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError:
                    next_data = extract_next_data_json(response.text)
                    if next_data is None:
                        raise
                    data = next_data
        except requests.RequestException as exc:
            print(f"[{config.name}] API request failed at offset {offset}: {exc}")
            break
        except ValueError as exc:
            print(f"[{config.name}] API returned unsupported payload at offset {offset}: {exc}")
            break

        edges = api.extract_edges(data)
        if not edges:
            print(f"[{config.name}] API returned 0 edges at offset {offset}, stopping")
            break

        oldest: datetime | None = None
        new_count = 0
        for edge in edges:
            node = edge.get("node", edge)
            dt = parse_api_datetime(node.get(api.edge_date_key))
            if dt and (oldest is None or dt < oldest):
                oldest = dt
            if cutoff_date is not None and dt is not None and dt.date() < cutoff_date:
                continue

            edge_url = _extract_edge_url(edge, api.edge_url_keys)
            if not edge_url:
                continue
            full_url = urljoin(config.base_url, edge_url)
            is_blocked = any(re.search(p, full_url) for p in config.blocked_url_regexes)
            if allowed.search(full_url) and not is_blocked and full_url not in all_links:
                all_links.add(full_url)
                new_count += 1

        print(f"[{config.name}] API offset {offset}: +{new_count} new links (total {len(all_links)})")
        if new_count == 0:
            print(f"[{config.name}] No new links at offset {offset}, stopping")
            break

        if max_links is not None and len(all_links) >= max_links:
            print(f"[{config.name}] Reached max_links={max_links} during API pagination, stopping")
            break

        if cutoff_date is not None and oldest is not None and oldest.date() < cutoff_date:
            print(f"[{config.name}] Oldest item {oldest.date()} older than cutoff {cutoff_date}, stopping")
            break

        pages += 1
        offset += api.step
        time.sleep(api.sleep_seconds)

    return all_links


def collect_articles(
    config: SiteConfig,
    months_back: int = 6,
    start_date: date | str | None = None,
    max_articles: int | None = None,
    progress_every: int = 25,
) -> list[dict[str, str]]:
    started = perf_counter()
    cutoff = resolve_cutoff_date(months_back=months_back, start_date=start_date)
    print(f"[{config.name}] Cutoff date: {cutoff}")

    candidate_urls = get_links_from_list_pages(config, cutoff_date=cutoff)
    target_candidates = None
    if max_articles is not None:
        # Keep a buffer above max_articles because some URLs will fail or have empty body.
        target_candidates = max(max_articles * 2, max_articles + 100)
    candidate_urls.update(fetch_links_from_sitemaps(config, cutoff_date=cutoff, max_links=target_candidates))
    candidate_urls.update(fetch_links_from_api(config, cutoff_date=cutoff, max_links=target_candidates))
    print(f"[{config.name}] Candidate URLs: {len(candidate_urls)}")

    date_pattern = config.date_url_pattern()
    articles: list[dict[str, str]] = []
    # Prefer newest items first before max_articles truncation.
    sorted_urls = sorted(candidate_urls, reverse=True)
    total_candidates = len(sorted_urls)
    for idx, url in enumerate(sorted_urls, start=1):
        url_date = get_date_from_url(url, date_pattern)
        if url_date and url_date < cutoff:
            continue
        article = scrape_article(url, config)
        # Avoid hammering a single host with article requests.
        time.sleep(0.25)
        if article is None:
            continue
        if not article["body"]:
            continue
        articles.append(article)
        if len(articles) == 1 or len(articles) % progress_every == 0:
            print(
                f"[{config.name}] Articles collected: {len(articles)} "
                f"(processed {idx}/{total_candidates} candidates)"
            )
        if max_articles is not None and len(articles) >= max_articles:
            print(f"[{config.name}] Reached max_articles={max_articles}, stopping article scraping")
            break

    elapsed = perf_counter() - started
    print(f"[{config.name}] Collected articles: {len(articles)}")
    print(f"[{config.name}] Elapsed: {elapsed:.1f}s ({elapsed/60:.2f} min)")
    return articles


def collect_multiple_sites(
    configs: list[SiteConfig],
    months_back: int = 6,
    start_date: date | str | None = None,
    max_articles_per_site: int | None = None,
    progress_every: int = 25,
) -> list[dict[str, str]]:
    started = perf_counter()
    all_rows: list[dict[str, str]] = []
    for config in configs:
        rows = collect_articles(
            config,
            months_back=months_back,
            start_date=start_date,
            max_articles=max_articles_per_site,
            progress_every=progress_every,
        )
        all_rows.extend(rows)
    elapsed = perf_counter() - started
    print(f"[multi-site] Total rows across {len(configs)} sites: {len(all_rows)}")
    print(f"[multi-site] Elapsed: {elapsed:.1f}s ({elapsed/60:.2f} min)")
    return all_rows


def save_articles_csv(rows: list[dict[str, str]], output_path: str) -> None:
    fieldnames = ["source", "category", "url", "title", "date", "body"]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {output_path}")


def build_freemalaysiatoday_config() -> SiteConfig:
    base = "https://www.freemalaysiatoday.com"
    return SiteConfig(
        name="freemalaysiatoday_local_business",
        base_url=base,
        list_urls=[f"{base}/category/category/business/local-business"],
        allowed_url_regex=r"https://www\.freemalaysiatoday\.com/category/business/\d{4}/\d{2}/\d{2}/",
        category_label="business",
        article_title_selector="h1",
        article_body_selector="article p, .entry-content p, p",
        date_from_url_regex=r"/(\d{4})/(\d{2})/(\d{2})/",
        api_pagination=ApiPaginationConfig(
            url=f"{base}/api/more-vertical-posts",
            method="POST",
            start_offset=0,
            step=20,
            build_payload=lambda offset: {"categorySlug": "local-business", "offset": offset},
            extract_edges=lambda data: data.get("posts", {}).get("edges", []),
            edge_url_keys=("uri", "link"),
            edge_date_key="date",
            max_pages=200,
            sleep_seconds=0.1,
        ),
        sitemap_urls=[
            "https://cms.freemalaysiatoday.com/sitemap.xml",
        ],
    )


def build_generic_site_config(
    *,
    name: str,
    base_url: str,
    list_urls: list[str],
    allowed_url_regex: str,
    title_selector: str = "h1",
    body_selector: str = "article p, p",
    date_from_url_regex: str | None = r"/(\d{4})/(\d{2})/(\d{2})/",
) -> SiteConfig:
    return SiteConfig(
        name=name,
        base_url=base_url,
        list_urls=list_urls,
        allowed_url_regex=allowed_url_regex,
        category_label="",
        article_title_selector=title_selector,
        article_body_selector=body_selector,
        date_from_url_regex=date_from_url_regex,
        api_pagination=None,
    )


def build_theedgemalaysia_config(category_slug: str) -> SiteConfig:
    base = "https://theedgemalaysia.com"
    api_url = f"{base}/api/loadMoreCategories"
    param_key = "categories"
    if category_slug == "politics":
        api_url = f"{base}/api/loadMoreOption"
        param_key = "option"
    elif category_slug not in {"economy", "corporate"}:
        raise ValueError("category_slug must be one of: politics, economy, corporate")
    return SiteConfig(
        name=f"theedgemalaysia_{category_slug}",
        base_url=base,
        list_urls=[
            f"{base}/categories/{category_slug}",
        ],
        allowed_url_regex=r"https://theedgemalaysia\.com/node/\d+",
        category_label=category_slug,
        article_title_selector="h1",
        article_body_selector="article p, .field--name-body p, .node__content p, p",
        date_from_url_regex=None,
        api_pagination=ApiPaginationConfig(
            url=api_url,
            method="GET",
            start_offset=0,
            step=10,
            build_payload=lambda offset: {"offset": offset, param_key: category_slug},
            extract_edges=lambda data: data.get("results", []) if isinstance(data, dict) else [],
            edge_url_keys=("alias",),
            edge_date_key="created",
            max_pages=250,
            sleep_seconds=0.1,
        ),
    )


def extract_theedge_search_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    page_props = data.get("props", {}).get("pageProps", {})
    if not isinstance(page_props, dict):
        return []
    items = page_props.get("newsArticleData", [])
    if not isinstance(items, list):
        return []
    return [row for row in items if isinstance(row, dict)]


def build_theedgemalaysia_search_config(
    keywords: str,
    label: str,
    *,
    from_date: str = "2023-01-01",
    to_date: str = "2025-09-30",
    max_pages: int = 5000,
) -> SiteConfig:
    base = "https://theedgemalaysia.com"
    encoded_keywords = quote(keywords)
    return SiteConfig(
        name=f"theedgemalaysia_search_{label}",
        base_url=base,
        list_urls=[],
        allowed_url_regex=r"https://theedgemalaysia\.com/node/\d+",
        category_label=label,
        article_title_selector="h1",
        article_body_selector="article p, .field--name-body p, .node__content p, p",
        date_from_url_regex=None,
        api_pagination=ApiPaginationConfig(
            url=(
                f"{base}/news-search-results?keywords={encoded_keywords}"
                f"&from={from_date}&to={to_date}&language=english&offset={{offset}}"
            ),
            method="GET",
            start_offset=0,
            step=10,
            build_payload=lambda offset: {},
            extract_edges=extract_theedge_search_results,
            edge_url_keys=("alias",),
            edge_date_key="created",
            max_pages=max_pages,
            sleep_seconds=0.15,
        ),
    )


def build_theedge_search_windowed_configs(
    *,
    keywords: str,
    base_label: str,
    from_date: str,
    to_date: str,
    window_days: int = 90,
    max_pages_per_window: int = 1200,
) -> list[SiteConfig]:
    start = date.fromisoformat(from_date)
    end = date.fromisoformat(to_date)
    if start > end:
        raise ValueError(f"from_date ({from_date}) must be <= to_date ({to_date})")

    configs: list[SiteConfig] = []
    cursor = start
    idx = 1
    while cursor <= end:
        window_end = min(cursor + timedelta(days=window_days - 1), end)
        window_from = cursor.isoformat()
        window_to = window_end.isoformat()
        label = f"{base_label}_w{idx:02d}_{window_from}_to_{window_to}"
        configs.append(
            build_theedgemalaysia_search_config(
                keywords,
                label,
                from_date=window_from,
                to_date=window_to,
                max_pages=max_pages_per_window,
            )
        )
        cursor = window_end + timedelta(days=1)
        idx += 1
    return configs


def build_malaysiakini_config() -> SiteConfig:
    base = "https://www.malaysiakini.com"
    category = "news"
    return SiteConfig(
        name="malaysiakini_news",
        base_url=base,
        list_urls=[
            f"{base}/en/latest/news",
            f"{base}/en/latest/business",
        ],
        allowed_url_regex=r"https://www\.malaysiakini\.com/(?:en/)?news/\d+",
        category_label=category,
        article_title_selector="h1",
        article_body_selector="article p, .mk-content p, .article-body p, p",
        date_from_url_regex=None,
        api_pagination=ApiPaginationConfig(
            url=f"{base}/api/en/latest/{category}" + "/{offset}",
            method="GET",
            start_offset=1,
            step=1,
            build_payload=lambda page: {"limit": 32},
            extract_edges=extract_malaysiakini_stories,
            edge_url_keys=("url", "link", "alias", "path"),
            edge_date_key="datePubUnix",
            max_pages=120,
            sleep_seconds=0.1,
        ),
    )


def build_businesstoday_config() -> SiteConfig:
    base = "https://www.businesstoday.com.my"
    marketing_pages = [f"{base}/category/marketing/"]
    # Conservative default window to avoid 429 throttling; increase if needed.
    marketing_pages.extend(f"{base}/category/marketing/page/{idx}/" for idx in range(2, 181))
    return SiteConfig(
        name="businesstoday_marketing",
        base_url=base,
        list_urls=marketing_pages,
        allowed_url_regex=r"https://www\.businesstoday\.com\.my/\d{4}/\d{2}/\d{2}/[^?#/]+/?$",
        category_label="marketing",
        article_title_selector="h1, .entry-title",
        article_body_selector=".entry-content p, .td-post-content p, .post-content p, article .entry-content p",
        date_from_url_regex=r"/(\d{4})/(\d{2})/(\d{2})/",
        blocked_url_regexes=[
            r"/about-reach-publishing/?$",
            r"/join-us/?$",
            r"/subscribe-to-the-print-edition/?$",
        ],
    )


def build_malaymail_config() -> SiteConfig:
    base = "https://www.malaymail.com"
    money_pages = [f"{base}/morearticles/money"]
    money_pages.extend(f"{base}/morearticles/money?pgno={idx}" for idx in range(2, 5001))
    return SiteConfig(
        name="malaymail_money",
        base_url=base,
        list_urls=money_pages,
        allowed_url_regex=r"https://www\.malaymail\.com/news/money/\d{4}/\d{2}/\d{2}/[^?#/]+/\d+/?$",
        category_label="money",
        article_title_selector="h1",
        article_body_selector="article p, .article-body p, .article-content p, p",
        date_from_url_regex=r"/(\d{4})/(\d{2})/(\d{2})/",
        request_headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{base}/news/money",
        },
        sitemap_urls=[
            "https://www.malaymail.com/sitemap.xml",
        ],
    )


def build_requested_malaysia_site_configs() -> list[SiteConfig]:
    configs: list[SiteConfig] = [
        # build_freemalaysiatoday_config(),
        # build_businesstoday_config(),
        # build_theedgemalaysia_config("politics"),
        # build_theedgemalaysia_config("economy"),
        # build_theedgemalaysia_config("corporate"),
        # build_theedgemalaysia_search_config(
        #     "corporate",
        #     "corporate",
        #     from_date="2023-01-01",
        #     to_date="2025-09-30",
        # ),
        # build_theedgemalaysia_search_config(
        #     "economy malaysia",
        #     "economy_malaysia",
        #     from_date="2023-01-01",
        #     to_date="2025-09-30",
        # ),
        # build_malaysiakini_config(),
        # build_malaymail_config(),
    ]
    configs.extend(
        build_theedge_search_windowed_configs(
            keywords="corporate",
            base_label="corporate",
            from_date="2023-01-01",
            to_date="2025-09-30",
            window_days=90,
            max_pages_per_window=1200,
        )
    )
    return configs


def validate_site_config(
    config: SiteConfig,
    sample_size: int = 5,
    months_back: int = 6,
    start_date: date | str | None = None,
) -> list[dict[str, str]]:
    cutoff = resolve_cutoff_date(months_back=months_back, start_date=start_date)
    candidate_set = (
        get_links_from_list_pages(config, cutoff_date=cutoff)
        .union(fetch_links_from_sitemaps(config, cutoff_date=cutoff))
        .union(fetch_links_from_api(config, cutoff_date=cutoff))
    )

    candidate_urls = sorted(candidate_set)
    candidate_urls = candidate_urls[:sample_size]

    results: list[dict[str, str]] = []
    for url in candidate_urls:
        article = scrape_article(url, config)
        if article is None:
            results.append({"url": url, "status": "fail", "reason": "fetch/redirect/filter failed"})
            continue
        if not article["title"]:
            results.append({"url": article["url"], "status": "fail", "reason": "missing title"})
            continue
        if len(article["body"]) < 200:
            results.append({"url": article["url"], "status": "warn", "reason": "short body (<200 chars)"})
            continue
        results.append({"url": article["url"], "status": "pass", "reason": "ok"})

    passed = sum(1 for r in results if r["status"] == "pass")
    warned = sum(1 for r in results if r["status"] == "warn")
    failed = sum(1 for r in results if r["status"] == "fail")
    print(f"[{config.name}] validation -> pass={passed}, warn={warned}, fail={failed}")
    return results


def validate_requested_malaysia_site_configs(
    sample_size: int = 5,
    months_back: int = 6,
    start_date: date | str | None = None,
) -> dict[str, list[dict[str, str]]]:
    report: dict[str, list[dict[str, str]]] = {}
    for config in build_requested_malaysia_site_configs():
        report[config.name] = validate_site_config(
            config,
            sample_size=sample_size,
            months_back=months_back,
            start_date=start_date,
        )
    return report


def main() -> None:
    configs = build_requested_malaysia_site_configs()
    rows = collect_multiple_sites(
        configs,
        start_date="2023-01-01",
        max_articles_per_site=None,
    )
    output_path = "data/raw/news_sources/theedgemalaysia_corporate_2023-01-01_to_2025-09-30.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_articles_csv(rows, output_path)


if __name__ == "__main__":
    main()
