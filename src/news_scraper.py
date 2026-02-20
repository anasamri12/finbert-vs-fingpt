from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable
import csv
import re
import time
from time import perf_counter
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


ParserFunc = Callable[[BeautifulSoup], str]


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
    try:
        response = requests.get(
            url,
            headers=config.request_headers,
            timeout=config.timeout_seconds,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[{config.name}] Skipping article {url}: {exc}")
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

    if config.body_parser is not None:
        body = config.body_parser(soup)
    else:
        body = extract_text_default(soup, config.article_body_selector)

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


def get_links_from_list_pages(config: SiteConfig) -> set[str]:
    allowed = config.allowed_url_pattern()
    links: set[str] = set()
    for list_url in config.list_urls:
        try:
            response = requests.get(list_url, headers=config.request_headers, timeout=config.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[{config.name}] List page failed {list_url}: {exc}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for anchor in soup.find_all("a"):
            href = anchor.get("href") or ""
            if not href:
                continue
            full_url = urljoin(config.base_url, href)
            if allowed.search(full_url):
                is_blocked = any(re.search(p, full_url) for p in config.blocked_url_regexes)
                if is_blocked:
                    continue
                links.add(full_url)

    print(f"[{config.name}] HTML list pages yielded {len(links)} unique links")
    return links


def _extract_edge_url(edge: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    node = edge.get("node", edge)
    for key in keys:
        value = node.get(key)
        if value:
            return str(value)
    return None


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
            data = response.json()
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
                data = response.json()
        except requests.RequestException as exc:
            print(f"[{config.name}] API request failed at offset {offset}: {exc}")
            break
        except ValueError as exc:
            print(f"[{config.name}] API returned non-JSON at offset {offset}: {exc}")
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

    candidate_urls = get_links_from_list_pages(config)
    target_candidates = None
    if max_articles is not None:
        # Keep a buffer above max_articles because some URLs will fail or have empty body.
        target_candidates = max(max_articles * 2, max_articles + 100)
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


def build_requested_malaysia_site_configs() -> list[SiteConfig]:
    return [
        build_freemalaysiatoday_config(),
        build_theedgemalaysia_config("politics"),
        build_theedgemalaysia_config("economy"),
        build_theedgemalaysia_config("corporate"),
        build_malaysiakini_config(),
    ]


def validate_site_config(
    config: SiteConfig,
    sample_size: int = 5,
    months_back: int = 6,
    start_date: date | str | None = None,
) -> list[dict[str, str]]:
    cutoff = resolve_cutoff_date(months_back=months_back, start_date=start_date)
    candidate_set = get_links_from_list_pages(config).union(fetch_links_from_api(config, cutoff_date=cutoff))

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


if __name__ == "__main__":
    configs = build_requested_malaysia_site_configs()
    rows = collect_multiple_sites(
        configs,
        start_date="2023-01-01",
        max_articles_per_site=None,
    )
    save_articles_csv(rows, "malaysia_news_since_2023.csv")
