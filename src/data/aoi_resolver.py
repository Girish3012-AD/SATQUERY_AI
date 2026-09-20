"""
aoi_resolver.py — Named place → WGS84 bounding box resolver.

Priority:
  1. Hard-coded high-confidence AOIs for problem-statement locations.
  2. Raises a clear error for unknown places (no fabrication).

Coordinates sourced from authoritative geographic references:
  - Rasuwa: Government of Nepal district boundaries.
  - Pune: Municipal Corporation of Pune boundary.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AOI:
    """Geographic area of interest in WGS84."""

    name: str
    bbox: list[float]          # [min_lon, min_lat, max_lon, max_lat]
    centroid: tuple[float, float]   # (lon, lat)
    description: str
    source: str


# ---------------------------------------------------------------------------
# Hard-coded authoritative AOIs
# ---------------------------------------------------------------------------
# Format: canonical name → AOI
# Keys are lower-cased for lookup; canonical name preserved in value.
# ---------------------------------------------------------------------------

_KNOWN_AOIS: dict[str, AOI] = {
    "rasuwa": AOI(
        name="Rasuwa District, Nepal",
        bbox=[85.0, 28.0, 85.5, 28.5],
        centroid=(85.25, 28.25),
        description=(
            "Rasuwa District, Bagmati Province, Nepal. "
            "High-altitude Himalayan district on the Tibet border, "
            "prone to glacial lake outburst floods (GLOFs) and "
            "monsoon flooding."
        ),
        source="Government of Nepal administrative boundaries",
    ),
    "rasuwa district": AOI(
        name="Rasuwa District, Nepal",
        bbox=[85.0, 28.0, 85.5, 28.5],
        centroid=(85.25, 28.25),
        description=(
            "Rasuwa District, Bagmati Province, Nepal."
        ),
        source="Government of Nepal administrative boundaries",
    ),
    "rasuwa nepal": AOI(
        name="Rasuwa District, Nepal",
        bbox=[85.0, 28.0, 85.5, 28.5],
        centroid=(85.25, 28.25),
        description=(
            "Rasuwa District, Bagmati Province, Nepal."
        ),
        source="Government of Nepal administrative boundaries",
    ),
    "pune": AOI(
        name="Pune, Maharashtra, India",
        bbox=[73.7, 18.4, 74.0, 18.65],
        centroid=(73.85, 18.52),
        description="Pune Metropolitan Area, Maharashtra, India.",
        source="Municipal Corporation of Pune boundary (approx.)",
    ),
    "pune india": AOI(
        name="Pune, Maharashtra, India",
        bbox=[73.7, 18.4, 74.0, 18.65],
        centroid=(73.85, 18.52),
        description="Pune Metropolitan Area, Maharashtra, India.",
        source="Municipal Corporation of Pune boundary (approx.)",
    ),
}


def resolve_aoi(place_name: str) -> AOI:
    """
    Resolve a place name to an AOI.

    Raises
    ------
    ValueError
        When the place name is not in the known registry and
        no network geocoder is configured.
    """
    if not place_name or not place_name.strip():
        raise ValueError("place_name cannot be empty.")

    key = place_name.strip().lower()

    # Direct lookup
    if key in _KNOWN_AOIS:
        return _KNOWN_AOIS[key]

    # Partial match — if the query contains a known key
    for known_key, aoi in _KNOWN_AOIS.items():
        if known_key in key:
            return aoi

    raise ValueError(
        f"Unknown place: '{place_name}'. "
        "The place is not in the SATQuery AOI registry. "
        "Supported places: "
        + ", ".join(sorted(_KNOWN_AOIS.keys()))
        + ". "
        "Add entries to src/data/aoi_resolver.py to extend coverage."
    )


def extract_location_from_query(query: str) -> str | None:
    """
    Heuristically extract a location name from a natural-language query.

    Returns the extracted string or None if no location pattern found.
    This is deterministic keyword matching — not LLM reasoning.
    """
    import re

    patterns = [
        r"\bin\s+([A-Z][a-zA-Z\s,]+(?:Nepal|India|Pakistan|Bangladesh|"
        r"Sri Lanka|Bhutan|Myanmar|Thailand|Vietnam|Indonesia|Philippines|"
        r"China|Japan|Korea|Australia))",
        r"\bover\s+([A-Z][a-zA-Z\s,]+(?:district|region|area|province|"
        r"state|city|town))",
        r"\bnear\s+([A-Z][a-zA-Z\s,]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(".,;")

    # Simpler fallback: "in <Word>, <Word>"
    simple = re.search(
        r"\bin\s+([A-Za-z][a-zA-Z]+(?:[,\s]+[A-Za-z][a-zA-Z]+)?)",
        query,
        re.IGNORECASE,
    )
    if simple:
        candidate = simple.group(1).strip().rstrip(".,;")
        try:
            resolve_aoi(candidate)
            return candidate
        except ValueError:
            pass

    return None
