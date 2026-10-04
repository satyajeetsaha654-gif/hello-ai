# ============================================================
# HELLO AI — IMPROVED LOCATION SERVICE
# Addresses, roads, postal codes, coordinates and map links
# ============================================================

import logging
import os
import threading
import time
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"

HEADERS = {
    "User-Agent": "HelloAI-LocationService/1.1"
}

TIMEOUT = 15
MAX_RESULTS = 5

# Public Nominatim service should not be queried faster
# than one request per second.
_request_lock = threading.Lock()
_last_request_time = 0.0


def _request_json(url, params):
    """Make a polite, validated Nominatim request."""

    global _last_request_time

    with _request_lock:
        wait = 1.05 - (time.monotonic() - _last_request_time)

        if wait > 0:
            time.sleep(wait)

        try:
            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=TIMEOUT,
            )
            _last_request_time = time.monotonic()
        except requests.RequestException:
            _last_request_time = time.monotonic()
            raise

    response.raise_for_status()
    return response.json()


def _text(value):
    """Convert optional fields to safe, trimmed text."""

    if value is None:
        return ""

    return str(value).strip()


def _google_maps_url(query):
    """Create a Maps search link, not a claim of verified coordinates."""

    query = _text(query)

    if not query:
        return ""

    return (
        "https://www.google.com/maps/search/?"
        + urlencode({"api": "1", "query": query})
    )


def _osm_url(item):
    """Return the real OSM object URL when its ID is available."""

    osm_type = _text(item.get("osm_type")).lower()
    osm_id = _text(item.get("osm_id"))

    type_names = {
        "node": "node",
        "way": "way",
        "relation": "relation",
        "n": "node",
        "w": "way",
        "r": "relation",
    }

    object_type = type_names.get(osm_type)

    if object_type and osm_id.isdigit():
        return f"https://www.openstreetmap.org/{object_type}/{osm_id}"

    return ""


def _normalise_result(item):
    """Convert a Nominatim result into a consistent structure."""

    if not isinstance(item, dict):
        return None

    address = item.get("address") or {}

    if not isinstance(address, dict):
        address = {}

    name = _text(item.get("name"))
    display_name = _text(item.get("display_name"))

    road = (
        address.get("road")
        or address.get("pedestrian")
        or address.get("residential")
        or address.get("footway")
        or address.get("path")
        or ""
    )

    city = (
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("municipality")
        or address.get("city_district")
        or ""
    )

    suburb = (
        address.get("suburb")
        or address.get("neighbourhood")
        or address.get("quarter")
        or address.get("hamlet")
        or ""
    )

    latitude = _text(item.get("lat"))
    longitude = _text(item.get("lon"))

    maps_query = display_name or name

    if latitude and longitude:
        maps_query = f"{latitude},{longitude}"

    return {
        "name": name,
        "display_name": display_name,
        "latitude": latitude,
        "longitude": longitude,
        "type": _text(item.get("type")),
        "category": _text(item.get("category")),
        "road": _text(road),
        "house_number": _text(
            address.get("house_number")
        ),
        "suburb": _text(suburb),
        "city": _text(city),
        "district": _text(
            address.get("state_district")
            or address.get("county")
            or address.get("district")
            or ""
        ),
        "state": _text(address.get("state")),
        "country": _text(address.get("country")),
        "postcode": _text(address.get("postcode")),
        "country_code": _text(address.get("country_code")),
        "source": "OpenStreetMap Nominatim",
        "source_url": _osm_url(item),
        "maps_url": _google_maps_url(maps_query),
        "importance": item.get("importance"),
        "address_details": {
            str(key): _text(value)
            for key, value in address.items()
            if value is not None
        },
    }


def search_location(query: str, limit: int = 5) -> list[dict]:
    """
    Search globally for a place, road, address or postcode.

    Missing address components remain empty; they are never guessed.
    """

    query = _text(query)

    if not query:
        return []

    try:
        limit = max(1, min(int(limit), MAX_RESULTS))
    except (TypeError, ValueError):
        limit = 5

    params = {
        "q": query[:500],
        "format": "jsonv2",
        "addressdetails": 1,
        "namedetails": 1,
        "extratags": 1,
        "limit": limit,
    }

    try:
        data = _request_json(
            NOMINATIM_SEARCH_URL,
            params,
        )

        if not isinstance(data, list):
            return []

        results = []
        seen = set()

        for item in data:
            result = _normalise_result(item)

            if not result:
                continue

            identity = (
                result["display_name"].casefold(),
                result["latitude"],
                result["longitude"],
            )

            if identity in seen:
                continue

            seen.add(identity)
            results.append(result)

        logger.info(
            "Location search completed: results=%d",
            len(results),
        )

        return results

    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ) as exc:
        logger.warning(
            "Location search failed: %s",
            type(exc).__name__,
        )
        return []


def reverse_geocode(latitude, longitude) -> dict:
    """Find the best available address for given coordinates."""

    try:
        lat = float(latitude)
        lon = float(longitude)

        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return {}

        data = _request_json(
            NOMINATIM_REVERSE_URL,
            {
                "lat": lat,
                "lon": lon,
                "format": "jsonv2",
                "addressdetails": 1,
                "namedetails": 1,
                "zoom": 18,
            },
        )

        if not isinstance(data, dict):
            return {}

        result = _normalise_result(data)

        if not result:
            return {}

        result["source"] = "OpenStreetMap Nominatim"
        result["maps_url"] = _google_maps_url(
            f"{lat},{lon}"
        )

        return result

    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ) as exc:
        logger.warning(
            "Reverse geocoding failed: %s",
            type(exc).__name__,
        )
        return {}