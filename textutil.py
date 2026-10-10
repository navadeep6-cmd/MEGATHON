"""Text normalisation helpers (standard library only).

Used for category names, file titles and descriptions: HTML stripping,
tokenisation, a light English stemmer, year extraction and script detection.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import FrozenSet, List, Optional

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_YEAR_RE = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)")
_EXT_RE = re.compile(r"\.[A-Za-z0-9]{2,5}$")
# Camera / phone filename noise: IMG_1234, DSC09876, P1010001, PXL_..., long digit runs.
_CAMERA_TOKEN_RE = re.compile(
    r"^(?:img|dsc|dscf|dscn|dcim|pict|imag|pano|mvimg|pxl|vid|wp|p)\d*$|^\d{5,}$",
    re.IGNORECASE,
)

STOPWORDS = frozenset(
    """
    a an and are as at be by for from in into is it of on or the to with without
    de la le les el los las del da do dos das di du und der die das im am von zu
    category categories file files image images photo photos photograph photographs
    picture pictures view views
    """.split()
)


def strip_html(text: Optional[str]) -> str:
    """Remove HTML tags/entities and collapse whitespace."""
    if not text:
        return ""
    text = _TAG_RE.sub(" ", str(text))
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def clean_category_name(title: str) -> str:
    """'Category:Bridges_in_Paris' -> 'Bridges in Paris'."""
    name = title.strip()
    if name.lower().startswith("category:"):
        name = name[len("category:"):]
    name = name.replace("_", " ")
    return _WS_RE.sub(" ", name).strip()


def clean_file_title(title: str) -> str:
    """'File:IMG_2041 Eiffel_Tower-at night.jpg' -> 'Eiffel Tower at night'."""
    name = title.strip()
    if name.lower().startswith("file:"):
        name = name[len("file:"):]
    name = _EXT_RE.sub("", name)
    words: List[str] = []
    after_camera = False
    for word in re.split(r"[\s_\-]+", name):
        core = _strip_punct(word)
        if not core:
            continue
        if _CAMERA_TOKEN_RE.match(core):
            after_camera = True
            continue
        if after_camera and core.isdigit():  # sequence number after IMG_/DSC_ etc.
            continue
        after_camera = False
        words.append(word)
    return " ".join(words)


def _strip_punct(word: str) -> str:
    return word.strip(".,;:!?()[]{}'\"")


def fold_accents(text: str) -> str:
    """Drop combining accents that sit on ASCII letters (São -> Sao).

    Marks on non-Latin base characters (e.g. Devanagari) are kept so that
    words in other scripts are not mangled.
    """
    out: List[str] = []
    prev_ascii = False
    for ch in unicodedata.normalize("NFKD", text):
        if unicodedata.combining(ch):
            if prev_ascii:
                continue
            out.append(ch)
            continue
        out.append(ch)
        prev_ascii = ord(ch) < 128
    return unicodedata.normalize("NFC", "".join(out))


def light_stem(token: str) -> str:
    """Very small plural stripper; applied identically to both sides of a match."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es"):
        stem = token[:-2]
        if stem.endswith(("s", "x", "z", "ch", "sh")):
            return stem
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    """Lower-cased word tokens with accents folded."""
    if not text:
        return []
    return _TOKEN_RE.findall(fold_accents(text).lower())


def content_tokens(text: str) -> FrozenSet[str]:
    """Stemmed tokens without stopwords, camera noise or single letters."""
    toks = set()
    for tok in tokenize(text):
        if tok in STOPWORDS or _CAMERA_TOKEN_RE.match(tok):
            continue
        if len(tok) == 1 and not tok.isdigit():
            continue
        toks.add(light_stem(tok))
    return frozenset(toks)


def category_tokens(name: str) -> FrozenSet[str]:
    """Content tokens of a category name; falls back to all tokens if empty."""
    toks = content_tokens(name)
    if toks:
        return toks
    return frozenset(light_stem(t) for t in tokenize(name))


def find_year(text: Optional[str], lo: int = 1000, hi: int = 2099) -> Optional[int]:
    """First plausible 4-digit year in text, or None."""
    if not text:
        return None
    for match in _YEAR_RE.finditer(text):
        year = int(match.group(1))
        if lo <= year <= hi:
            return year
    return None


def meta_year(date_original: Optional[str]) -> Optional[int]:
    """Year from an EXIF/extmetadata DateTimeOriginal value (may contain HTML)."""
    return find_year(strip_html(date_original), lo=1800, hi=2100)


def name_script(name: str) -> str:
    """'ascii', 'latin-extended' or 'non-latin' - used for language/script audits."""
    letters = [ch for ch in name if ch.isalpha()]
    if not letters:
        return "ascii"
    if all(ord(ch) < 128 for ch in letters):
        return "ascii"
    non_latin = 0
    for ch in letters:
        if ord(ch) < 128:
            continue
        try:
            uname = unicodedata.name(ch)
        except ValueError:
            uname = ""
        if "LATIN" not in uname:
            non_latin += 1
    return "non-latin" if non_latin * 2 >= len(letters) else "latin-extended"


def build_meta_text(title_clean: str, description: str, max_chars: int = 400) -> str:
    """Single string fed to the text tower for the metadata scorer."""
    parts = [p for p in (title_clean.strip(), description.strip()) if p]
    text = ". ".join(parts)
    return text[:max_chars]
