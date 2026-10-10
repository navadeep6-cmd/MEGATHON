"""Geographic helpers: great-circle distance and coarse world regions.

`region_of` uses rough bounding boxes (documented as approximate) so that
evaluation can be stratified by region for the geographic-bias audit.
"""

from __future__ import annotations

import math
from typing import Optional

EARTH_RADIUS_KM = 6371.0088

REGIONS = (
    "Africa",
    "Asia",
    "West Asia",
    "Europe",
    "North America",
    "Latin America",
    "Oceania",
    "Antarctica",
    "no geotag",
)


def _valid(lat: Optional[float], lon: Optional[float]) -> bool:
    if lat is None or lon is None:
        return False
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    if math.isnan(lat_f) or math.isnan(lon_f):
        return False
    return -90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def haversine_km_np(lat: float, lon: float, lats, lons):
    """Vectorised distance from one point to arrays of points (numpy)."""
    import numpy as np

    p1 = np.radians(lat)
    p2 = np.radians(lats)
    dphi = p2 - p1
    dlmb = np.radians(lons - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def region_of(lat: Optional[float], lon: Optional[float]) -> str:
    """Approximate continent-level region for a coordinate (bounding-box rules)."""
    if not _valid(lat, lon):
        return "no geotag"
    lat, lon = float(lat), float(lon)
    if lat < -60:
        return "Antarctica"
    if lon <= -25:
        if lon < -140 and lat < 30:
            return "Oceania"
        return "North America" if lat >= 15 else "Latin America"
    if (lon > 110 and lat < -10) or (lon > 140 and lat < 0):
        return "Oceania"
    if lon > 160:
        return "Oceania" if lat < 30 else "Asia"
    if lon <= 40 and (
        lat >= 37.5 or (lat >= 36 and lon < -1.8) or (lat >= 34.5 and 12 <= lon <= 34.5)
    ):
        return "Europe"
    if lat >= 50:
        return "Europe" if lon <= 60 else "Asia"
    if 34 < lon <= 63 and lat > 12:
        return "West Asia"
    if lon <= 60 and lat < 38:
        return "Africa"
    return "Asia"
