"""Read-only MediaWiki API client for Wikimedia Commons.

Follows the Wikimedia API etiquette: descriptive User-Agent with contact
information, serial requests with a small delay, maxlag, and back-off on
429/5xx. Nothing is ever written to Commons.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import requests

from .textutil import clean_category_name

log = logging.getLogger(__name__)

API_URL = "https://commons.wikimedia.org/w/api.php"


def user_agent(contact: str) -> str:
    contact = contact or os.environ.get("COMMONS_CONTACT", "")
    if not contact:
        log.warning("Set COMMONS_CONTACT (e-mail or project URL) so Wikimedia can reach you; "
                    "anonymous clients may be rate-limited or blocked.")
        contact = "contact not set"
    return "CommonsCategorySuggester/1.0 (research prototype; %s) python-requests/%s" % (
        contact, requests.__version__)


def _chunks(items: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _merge_page(dst: dict, src: dict) -> None:
    """Merge a page object from a continuation response into an accumulated one."""
    for key, val in src.items():
        if isinstance(val, list) and isinstance(dst.get(key), list):
            existing = dst[key]
            for item in val:
                if item not in existing:
                    existing.append(item)
        elif key not in dst:
            dst[key] = val


class CommonsClient:
    def __init__(self, api_url: str = API_URL, contact: str = "", sleep: float = 0.1,
                 max_retries: int = 6, timeout: float = 60.0):
        self.api_url = api_url
        self.sleep = sleep
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent(contact)

    # -- transport ------------------------------------------------------------
    def _get(self, params: Dict) -> Dict:
        p = dict(params)
        p.setdefault("format", "json")
        p.setdefault("formatversion", "2")
        p.setdefault("maxlag", "5")
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(self.api_url, params=p, timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("request failed (%s); retry %d", exc, attempt + 1)
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After", delay))
                log.warning("HTTP %d; waiting %.0fs", r.status_code, wait)
                time.sleep(wait)
                delay *= 2
                continue
            r.raise_for_status()
            data = r.json()
            err = data.get("error")
            if err:
                if err.get("code") == "maxlag":
                    time.sleep(float(r.headers.get("Retry-After", 5)))
                    continue
                raise RuntimeError("API error %s: %s" % (err.get("code"), err.get("info")))
            time.sleep(self.sleep)
            return data
        raise RuntimeError("giving up after %d retries" % self.max_retries)

    def query(self, params: Dict) -> Iterator[Dict]:
        """Yield the 'query' part of each response, following continuation."""
        base = dict(params)
        base["action"] = "query"
        cont: Dict = {}
        while True:
            data = self._get(dict(base, **cont))
            if "query" in data:
                yield data["query"]
            if "continue" not in data:
                break
            cont = data["continue"]

    def query_pages(self, params: Dict) -> Tuple[Dict[str, dict], Dict[str, str]]:
        """Run a page query, merge pages across continuations.

        Returns (pages keyed by title, title map from requested -> final title).
        """
        pages: Dict[str, dict] = {}
        title_map: Dict[str, str] = {}
        for q in self.query(params):
            for item in q.get("normalized", []):
                title_map[item["from"]] = item["to"]
            for item in q.get("redirects", []):
                title_map[item["from"]] = item["to"]
            for page in q.get("pages", []):
                key = page.get("title") or str(page.get("pageid"))
                if key in pages:
                    _merge_page(pages[key], page)
                else:
                    pages[key] = page
        return pages, title_map

    # -- high-level calls -----------------------------------------------------
    def category_members(self, category: str, member_type: str, limit: int) -> List[dict]:
        """Files ('file') or subcategories ('subcat') of a category, newest additions first."""
        out: List[dict] = []
        params = {
            "list": "categorymembers",
            "cmtitle": "Category:" + clean_category_name(category),
            "cmtype": member_type,
            "cmlimit": str(min(500, max(1, limit))),
            "cmprop": "ids|title|type|timestamp",
            "cmsort": "timestamp",
            "cmdir": "desc",
        }
        for q in self.query(params):
            out.extend(q.get("categorymembers", []))
            if len(out) >= limit:
                break
        return out[:limit]

    def file_info(self, pageids: Sequence[int], thumb_width: int) -> Dict[int, dict]:
        """Non-hidden categories, image info (thumbnail, size, uploader, licence,
        description, date) and primary coordinates for files, 50 per request."""
        out: Dict[int, dict] = {}
        for chunk in _chunks(list(pageids), 50):
            params = {
                "pageids": "|".join(str(p) for p in chunk),
                "prop": "categories|imageinfo|coordinates",
                "clshow": "!hidden",
                "cllimit": "max",
                "iiprop": "url|timestamp|user|size|mime|extmetadata",
                "iiurlwidth": str(thumb_width),
                "iiextmetadatafilter": "ImageDescription|DateTimeOriginal|LicenseShortName",
                "iiextmetadatalanguage": "en",
                "colimit": "max",
            }
            pages, _ = self.query_pages(params)
            for page in pages.values():
                if "pageid" in page and not page.get("missing"):
                    out[int(page["pageid"])] = page
        return out

    def file_info_by_title(self, titles: Sequence[str], thumb_width: int) -> Dict[str, dict]:
        """Same as file_info but addressed by 'File:...' titles (used by the live worker)."""
        out: Dict[str, dict] = {}
        for chunk in _chunks(list(titles), 50):
            params = {
                "titles": "|".join(chunk),
                "prop": "categories|imageinfo|coordinates",
                "clshow": "!hidden",
                "cllimit": "max",
                "iiprop": "url|timestamp|user|size|mime|extmetadata",
                "iiurlwidth": str(thumb_width),
                "iiextmetadatafilter": "ImageDescription|DateTimeOriginal|LicenseShortName",
                "iiextmetadatalanguage": "en",
                "colimit": "max",
            }
            pages, _ = self.query_pages(params)
            for title, page in pages.items():
                if not page.get("missing"):
                    out[title] = page
        return out

    def category_parents(self, categories: Iterable[str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Non-hidden parent categories. Returns (parents by name, hard-redirect map)."""
        names = sorted(set(clean_category_name(c) for c in categories))
        parents: Dict[str, List[str]] = {}
        redirects: Dict[str, str] = {}
        for chunk in _chunks(names, 50):
            params = {
                "titles": "|".join("Category:" + n for n in chunk),
                "prop": "categories",
                "clshow": "!hidden",
                "cllimit": "max",
                "redirects": "1",
            }
            pages, title_map = self.query_pages(params)
            for n in chunk:
                title = "Category:" + n
                final = title
                hops = 0
                while final in title_map and hops < 5:
                    final = title_map[final]
                    hops += 1
                final_name = clean_category_name(final)
                if final_name != n:
                    redirects[n] = final_name
                page = pages.get(final)
                if page is None or page.get("missing"):
                    continue
                parents[final_name] = [clean_category_name(c["title"]) for c in page.get("categories", [])]
        return parents, redirects

    def soft_redirects(self, categories: Iterable[str]) -> List[str]:
        """Categories tagged with {{Category redirect}} (soft redirects) - dropped from labels."""
        names = sorted(set(clean_category_name(c) for c in categories))
        out: List[str] = []
        for chunk in _chunks(names, 50):
            params = {
                "titles": "|".join("Category:" + n for n in chunk),
                "prop": "categories",
                "clcategories": "Category:Category redirects",
                "cllimit": "max",
            }
            pages, _ = self.query_pages(params)
            for title, page in pages.items():
                if page.get("categories"):
                    out.append(clean_category_name(title))
        return out


def extmeta_value(imageinfo: dict, key: str) -> str:
    try:
        return str(imageinfo["extmetadata"][key]["value"])
    except (KeyError, TypeError):
        return ""


def page_to_record(page: dict) -> Optional[dict]:
    """Flatten an API page object into a dataset record (None if not an image)."""
    infos = page.get("imageinfo") or []
    if not infos:
        return None
    ii = infos[0]
    mime = ii.get("mime", "")
    if not mime.startswith("image/"):
        return None
    lat = lon = None
    for coord in page.get("coordinates", []) or []:
        if coord.get("globe", "earth") == "earth":
            lat, lon = coord.get("lat"), coord.get("lon")
            if coord.get("primary"):
                break
    return {
        "pageid": page.get("pageid"),
        "title": page.get("title", ""),
        "thumb_url": ii.get("thumburl") or ii.get("url", ""),
        "page_url": ii.get("descriptionurl", ""),
        "width": ii.get("width"),
        "height": ii.get("height"),
        "mime": mime,
        "timestamp": ii.get("timestamp", ""),
        "uploader": ii.get("user"),
        "license": extmeta_value(ii, "LicenseShortName"),
        "description": extmeta_value(ii, "ImageDescription"),
        "date_original": extmeta_value(ii, "DateTimeOriginal"),
        "lat": lat,
        "lon": lon,
        "categories": [clean_category_name(c["title"]) for c in page.get("categories", []) or []],
    }
