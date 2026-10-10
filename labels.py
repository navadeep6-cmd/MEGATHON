"""Label cleaning and category metadata (standard library only).

* Removes maintenance / tracking / provenance categories that describe the
  upload process rather than image content.
* Assigns each category a coarse semantic type (event / abstract / scene /
  object) used for stratified evaluation.
* Frequency bands (head / torso / tail / zero-shot) for long-tail reporting.
* Licence allow-list and a person-category detector (we never suggest
  categories that identify individual people).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence

from .textutil import clean_category_name

_MEDIA_NOUN = r"(?:files?|media|images?|photos?|photographs?|pictures?|videos?|scans?|sounds?|audio)"
_PROCESS_WORD = (
    r"(?:by|from|with|without|uploaded|lacking|needing|missing|requiring|using|created|taken|"
    r"contributed|transferred|requested|to be|for|in need|of the day|made|supported|produced|"
    r"generated|derived|edited|cropped|restored|retouched|where|that|which)"
)

MAINTENANCE_PATTERNS: List[str] = [
    # licence and copyright tags
    r"^(?:cc|gfdl|pd|fal|cc0)(?:[- ]|$)",
    r"^(?:creative commons|public domain|copyright|copyrighted|licen[cs]e|permission|otrs|vrt|"
    r"personality rights|trademark|deletion|speedy deletion)\b",
    r"\blicen[cs]e (?:tags?|templates?|migration|review|layout)\b",
    r"^(?:self[- ]published|own work)\b",
    # provenance: "Files from Flickr", "Photographs taken on 2019-05-04", "Images by User:X"
    r"^" + _MEDIA_NOUN + r"\s+" + _PROCESS_WORD + r"\b",
    r"^pages?\s+(?:with|using|where|needing|that)\b",
    r"^uploaded\s+(?:with|by|via|from|using)\b",
    r"^(?:taken|shot)\s+(?:with|on|by)\b",
    r"\buser:",
    r"\b(?:supported by|contributed by|created with|made with|uploaded with|edited with|scanned by)\b",
    r"^(?:scans?|scanned|digiti[sz]ed)\s+(?:by|from|with)\b",
    r"^(?:djvu|pdf|svg|png|jpeg|jpg|tiff|gif|webm|ogg|webp)\s+(?:files|images|media)\b",
    # assessments
    r"^(?:quality|featured|valued)\s+(?:images|pictures|media|photographs|sounds|videos)\b",
    r"^(?:pictures?|images?|media|photos?)\s+of\s+the\s+(?:day|week|month|year)\b",
    # maintenance queues
    r"^(?:uncategori[sz]ed|unidentified|unknown|check categories|categories requiring|hidden categories|"
    r"wikimedia commons\b)",
    r"\b(?:maintenance|tracking|backlog|cleanup|clean-up)\b",
    r"\b(?:needing|needs|requiring)\b",
    r"\bto be (?:checked|categori[sz]ed|identified|renamed|deleted|cropped|improved|moved|merged|split)\b",
    # contests and projects (process, not content)
    r"^(?:wiki loves|wikimania|wiki science competition|wikipedia|wikidata|wikimedia|wikivoyage|"
    r"wikisource|wikibooks|wikinews)\b",
    r"^retouched\b",
]

_MAINT_RE = [re.compile(p, re.IGNORECASE) for p in MAINTENANCE_PATTERNS]

# Open licences accepted (LicenseShortName from extmetadata).
_LICENSE_OK_RE = re.compile(
    r"^(?:cc0|cc[- ]?by|cc[- ]?pd|public domain|pd\b|pd-|gfdl|fal|free art|attribution|"
    r"copyrighted free use|no restrictions|ogl|odbl|etalab|gpl|lgpl|bsd|mit|apache|"
    r"licence ouverte|open government)",
    re.IGNORECASE,
)

_PERSON_PARENT_RE = re.compile(r"^(?:men|women|people|persons|males|females)\b.*\bby name$", re.IGNORECASE)

# ---- semantic type heuristic -------------------------------------------------

_YEAR_IN_NAME = re.compile(r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2})s?(?!\d)")
_EVENT_WORDS = frozenset(
    """festival festivals protest protests demonstration demonstrations match ceremony ceremonies
    election elections wedding weddings championship championships tournament tournaments concert
    concerts parade parades exhibition exhibitions conference conferences olympic olympics war wars
    battle battles funeral funerals celebration celebrations carnival carnivals marathon marathons
    races rally rallies march marches strike strikes visit visits meeting meetings summit summits fair
    fairs procession processions holi diwali eid christmas easter ramadan puja durga navratri pongal
    hanami competition competitions contest contests ceremony inauguration opening event events
    fireworks performance performances""".split()
)
_ABSTRACT_WORDS = frozenset(
    """art symbol symbols emotion emotions love poverty religion religions concept concepts style styles
    pattern patterns reflection reflections silhouette silhouettes minimalism abstract abstraction happiness
    sadness joy fear anger freedom democracy peace death beauty humor humour shadow shadows
    color colour colors colours texture textures geometry symmetry culture tradition traditions solitude
    loneliness friendship family childhood work labour labor wealth hope faith spirituality motion
    movement contrast perspective harmony chaos time memory nostalgia architecture""".split()
)
_SCENE_WORDS = frozenset(
    """landscape landscapes view views street streets beach beaches mountain mountains interior interiors
    skyline skylines cityscape cityscapes panorama panoramas river rivers lake lakes sea seas coast coasts
    coastline forest forests park parks garden gardens market markets station stations square squares
    road roads harbour harbours harbor harbors port ports desert deserts field fields valley valleys
    village villages night sunset sunsets sunrise sunrises waterfall waterfalls island islands
    countryside town towns city cities neighbourhood neighborhood neighbourhoods neighborhoods canyon
    canyons cave caves glacier glaciers hill hills meadow meadows plaza plazas bazaar bazaars
    sky skies clouds snow fog weather underwater aerial""".split()
)
_HEAD_SPLIT = re.compile(r"\s(?:in|of|at|by|from|on|near|during|for)\s", re.IGNORECASE)

CATEGORY_TYPES = ("object", "scene", "event", "abstract")


def is_maintenance(name: str, extra_patterns: Sequence[str] = ()) -> bool:
    """True if a category encodes process/provenance rather than content."""
    name = clean_category_name(name)
    if not name:
        return True
    for rx in _MAINT_RE:
        if rx.search(name):
            return True
    for pat in extra_patterns:
        if re.search(pat, name, re.IGNORECASE):
            return True
    return False


def license_ok(short_name: Optional[str]) -> bool:
    """True if the extmetadata LicenseShortName is an open licence."""
    if not short_name:
        return False
    return bool(_LICENSE_OK_RE.match(short_name.strip()))


def is_person_category(parents: Iterable[str]) -> bool:
    """True if any direct parent is a '<people> by name' category."""
    return any(_PERSON_PARENT_RE.search(clean_category_name(p)) for p in parents)


def clean_labels(
    categories: Iterable[str],
    redirects: Optional[Dict[str, str]] = None,
    drop: Optional[Iterable[str]] = None,
    extra_patterns: Sequence[str] = (),
) -> List[str]:
    """Normalise names, resolve redirects, drop maintenance/dropped categories, dedupe."""
    redirects = redirects or {}
    drop_set = set(drop or ())
    out: List[str] = []
    seen = set()
    for raw in categories:
        name = clean_category_name(raw)
        hops = 0
        while name in redirects and hops < 5:
            name = redirects[name]
            hops += 1
        if not name or name in seen or name in drop_set:
            continue
        if is_maintenance(name, extra_patterns):
            continue
        seen.add(name)
        out.append(name)
    return out


def category_type(name: str) -> str:
    """Coarse semantic type from the category name (heuristic, documented in README)."""
    name = clean_category_name(name)
    lower = name.lower()
    words = set(re.findall(r"[a-z]+", lower))
    if _YEAR_IN_NAME.search(name) or words & _EVENT_WORDS:
        return "event"
    head = _HEAD_SPLIT.split(" " + lower + " ", maxsplit=1)[0]
    head_words = set(re.findall(r"[a-z]+", head))
    if head_words & _ABSTRACT_WORDS:
        return "abstract"
    if head_words & _SCENE_WORDS:
        return "scene"
    return "object"


def frequency_band(train_count: int, tail_max: int = 9, torso_max: int = 49) -> str:
    """zero-shot (0) / tail (1..tail_max) / torso (..torso_max) / head (>torso_max)."""
    if train_count <= 0:
        return "zero-shot"
    if train_count <= tail_max:
        return "tail"
    if train_count <= torso_max:
        return "torso"
    return "head"


def resolution_band(width: Optional[int], height: Optional[int]) -> str:
    """Band of the original file's long side in pixels."""
    if not width or not height:
        return "unknown"
    side = max(int(width), int(height))
    if side < 800:
        return "<800px"
    if side < 2000:
        return "800-1999px"
    if side < 4000:
        return "2000-3999px"
    return ">=4000px"
