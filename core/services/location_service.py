# ============================================================
# HELLO AI — GENERIC LOCATION SERVICE
# India / Global Places, Addresses, Roads, PIN Codes
# ============================================================

import logging
import re
import threading
import time
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"

HEADERS = {
    "User-Agent": "HelloAI-LocationService/3.0"
}

TIMEOUT = 15
MAX_RESULTS = 5

_request_lock = threading.Lock()
_last_request_time = 0.0


# ============================================================
# BASIC HELPERS
# ============================================================

def _text(value):
    if value is None:
        return ""

    return str(value).strip()


def _request_json(url, params):
    """
    Polite Nominatim request.
    Public Nominatim should not be queried faster than
    approximately one request per second.
    """

    global _last_request_time

    with _request_lock:
        elapsed = time.monotonic() - _last_request_time
        wait = 1.05 - elapsed

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


def _google_maps_url(query):
    query = _text(query)

    if not query:
        return ""

    return (
        "https://www.google.com/maps/search/?"
        + urlencode({
            "api": "1",
            "query": query,
        })
    )


def _osm_url(item):
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
        return (
            f"https://www.openstreetmap.org/"
            f"{object_type}/{osm_id}"
        )

    return ""


# ============================================================
# CLEAN USER LOCATION QUESTION
# ============================================================

def _clean_location_query(query):
    """
    Convert a natural-language location question into a cleaner
    place/address search query.

    This is generic and is NOT tied to any particular place.
    """

    text = _text(query)

    if not text:
        return ""

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    # --------------------------------------------------------
    # English question phrases
    # --------------------------------------------------------

    english_patterns = [
        r"\bwhere is\b",
        r"\bwhere are\b",
        r"\bwhere can i find\b",
        r"\bwhere do i find\b",
        r"\bwhere located\b",
        r"\bwhere is located\b",
        r"\blocated where\b",
        r"\bwhat is the location of\b",
        r"\blocation of\b",
        r"\baddress of\b",
        r"\bwhat is the address of\b",
        r"\bfull address of\b",
        r"\bpin code of\b",
        r"\bpincode of\b",
        r"\bpostal code of\b",
        r"\bpostcode of\b",
        r"\bzip code of\b",
        r"\broad name of\b",
        r"\bstreet name of\b",
        r"\bwhich road\b",
        r"\bwhich street\b",
        r"\bwhich area\b",
        r"\bwhat area\b",
        r"\bwhat is the road\b",
        r"\bwhat is the street\b",
        r"\bwhat is the pin code\b",
        r"\bwhat is the pincode\b",
        r"\bwhat is the postal code\b",
        r"\bwhat is the postcode\b",
        r"\bwhat is the address\b",
        r"\bfind\b",
        r"\blocation\b",
        r"\baddress\b",
        r"\bpin code\b",
        r"\bpincode\b",
        r"\bpostal code\b",
        r"\bpostcode\b",
        r"\bzip code\b",
    ]

    for pattern in english_patterns:
        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE,
        )

    # --------------------------------------------------------
    # Roman Bengali question phrases
    # --------------------------------------------------------

    roman_bengali_patterns = [
        r"\bkothay ache\b",
        r"\bkothay\b",
        r"\bkothai ache\b",
        r"\bkothai\b",
        r"\bkon jaygay\b",
        r"\bkon elakay\b",
        r"\bkon elaka\b",
        r"\bkon rastay\b",
        r"\brastar nam ki\b",
        r"\broad er nam ki\b",
        r"\bstreet er nam ki\b",
        r"\bpin code koto\b",
        r"\bpincode koto\b",
        r"\bpostal code koto\b",
        r"\baddress ki\b",
        r"\bthikana ki\b",
        r"\bfull address ki\b",
        r"\blocation ki\b",
        r"\blocation bolo\b",
        r"\baddress bolo\b",
        r"\bpin code bolo\b",
        r"\bpostcode koto\b",
        r"\bkothay obosthito\b",
        r"\bkothay pabo\b",
    ]

    for pattern in roman_bengali_patterns:
        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE,
        )

    # --------------------------------------------------------
    # Bengali question phrases
    # --------------------------------------------------------

    bengali_patterns = [
        r"কোথায় আছে",
        r"কোথায় অবস্থিত",
        r"কোথায়",
        r"কোথাই আছে",
        r"কোন এলাকায়",
        r"কোন এলাকাতে",
        r"কোন এলাকায় আছে",
        r"কোন রাস্তায়",
        r"রাস্তার নাম কী",
        r"রাস্তার নাম কি",
        r"রোডের নাম কী",
        r"রোডের নাম কি",
        r"স্ট্রিটের নাম কী",
        r"স্ট্রিটের নাম কি",
        r"পিন কোড কত",
        r"পিনকোড কত",
        r"পোস্টাল কোড কত",
        r"পোস্টকোড কত",
        r"ঠিকানা কী",
        r"ঠিকানা কি",
        r"ঠিকানা বলো",
        r"সম্পূর্ণ ঠিকানা",
        r"পুরো ঠিকানা",
        r"লোকেশন কী",
        r"লোকেশন কি",
        r"লোকেশন বলো",
        r"কোথায় পাব",
    ]

    for pattern in bengali_patterns:
        text = re.sub(pattern, " ", text)

    # --------------------------------------------------------
    # Hindi question phrases
    # --------------------------------------------------------

    hindi_patterns = [
        r"कहाँ है",
        r"कहाँ स्थित है",
        r"कहाँ",
        r"किस जगह",
        r"किस इलाके में",
        r"किस सड़क पर",
        r"सड़क का नाम क्या है",
        r"रोड का नाम क्या है",
        r"स्ट्रीट का नाम क्या है",
        r"पिन कोड कितना है",
        r"पिनकोड कितना है",
        r"पोस्टल कोड कितना है",
        r"पता क्या है",
        r"पूरा पता",
        r"लोकेशन क्या है",
        r"लोकेशन बताओ",
        r"पता बताओ",
    ]

    for pattern in hindi_patterns:
        text = re.sub(
            pattern,
            " ",
            text,
        )

    # --------------------------------------------------------
    # Remove common trailing question words.
    # --------------------------------------------------------

    text = re.sub(
        r"\b(please|pls|tell me|bolo|batao|ki|kya|what|how)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"[?!.。,，।]+",
        " ",
        text,
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text[:500]


def _query_variants(query):
    """
    Generate generic search variants.

    The original query is always retained first.
    """

    original = _text(query)

    if not original:
        return []

    cleaned = _clean_location_query(original)

    variants = []

    def add(value):
        value = _text(value)

        if not value:
            return

        key = value.casefold()

        if key not in {
            item.casefold()
            for item in variants
        }:
            variants.append(value)

    add(original)
    add(cleaned)

    # If the cleaned query contains common question separators,
    # create a compact variant.
    compact = re.sub(
        r"\s+(?:the|of|in|at|near)\s+",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    add(compact)

    return variants[:3]


# ============================================================
# RESULT NORMALIZATION
# ============================================================

def _normalise_result(item):
    if not isinstance(item, dict):
        return None

    address = item.get("address") or {}

    if not isinstance(address, dict):
        address = {}

    name = _text(item.get("name"))
    display_name = _text(item.get("display_name"))

    road = (
        address.get("road")
        or address.get("street")
        or address.get("pedestrian")
        or address.get("residential")
        or address.get("footway")
        or address.get("path")
        or ""
    )

    suburb = (
        address.get("suburb")
        or address.get("neighbourhood")
        or address.get("neighborhood")
        or address.get("quarter")
        or address.get("residential")
        or address.get("hamlet")
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

    district = (
        address.get("state_district")
        or address.get("district")
        or address.get("county")
        or ""
    )

    state = address.get("state") or ""

    country = address.get("country") or ""

    postcode = (
        address.get("postcode")
        or address.get("postalcode")
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
        "district": _text(district),
        "state": _text(state),
        "country": _text(country),
        "postcode": _text(postcode),
        "country_code": _text(
            address.get("country_code")
        ),
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


# ============================================================
# RESULT SCORING
# ============================================================

def _tokens(text):
    text = _text(text).casefold()

    return {
        token
        for token in re.findall(
            r"[\w\u0980-\u09ff\u0900-\u097f]+",
            text,
            flags=re.UNICODE,
        )
        if len(token) > 1
    }


def _score_result(result, query):
    if not isinstance(result, dict):
        return -999

    query_tokens = _tokens(
        _clean_location_query(query)
    )

    if not query_tokens:
        query_tokens = _tokens(query)

    result_text = " ".join([
        result.get("name", ""),
        result.get("display_name", ""),
        result.get("road", ""),
        result.get("suburb", ""),
        result.get("city", ""),
        result.get("district", ""),
        result.get("state", ""),
        result.get("country", ""),
        result.get("postcode", ""),
    ])

    result_tokens = _tokens(result_text)

    if not result_tokens:
        return -999

    score = 0

    overlap = query_tokens.intersection(result_tokens)

    score += len(overlap) * 10

    name_tokens = _tokens(result.get("name", ""))

    name_overlap = query_tokens.intersection(name_tokens)

    score += len(name_overlap) * 20

    display = result.get(
        "display_name",
        "",
    ).casefold()

    cleaned = _clean_location_query(query).casefold()

    if cleaned and cleaned in display:
        score += 50

    if result.get("postcode"):
        score += 4

    if result.get("road"):
        score += 3

    if result.get("city"):
        score += 3

    if result.get("state"):
        score += 2

    if result.get("latitude") and result.get("longitude"):
        score += 2

    try:
        importance = float(
            result.get("importance") or 0
        )
        score += int(importance * 10)
    except (TypeError, ValueError):
        pass

    return score


def _sort_results(results, query):
    scored = []

    for result in results:
        score = _score_result(
            result,
            query,
        )

        scored.append(
            (
                score,
                result,
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        result
        for _, result in scored
    ]


# ============================================================
# LOCATION SEARCH
# ============================================================

def search_location(query: str, limit: int = 5) -> list[dict]:
    """
    Generic location/address search.

    Works with natural-language questions and place/address names.
    Does not hard-code any specific location.
    """

    query = _text(query)

    if not query:
        return []

    try:
        limit = max(
            1,
            min(
                int(limit),
                MAX_RESULTS,
            ),
        )
    except (TypeError, ValueError):
        limit = 5

    variants = _query_variants(query)

    if not variants:
        return []

    all_results = []
    seen = set()

    for search_query in variants:
        params = {
            "q": search_query[:500],
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

        except (
            requests.RequestException,
            ValueError,
            TypeError,
        ) as exc:
            logger.warning(
                "Nominatim search failed for variant: %s (%s)",
                search_query,
                type(exc).__name__,
            )
            continue

        if not isinstance(data, list):
            continue

        for item in data:
            result = _normalise_result(item)

            if not result:
                continue

            identity = (
                result.get("display_name", "").casefold(),
                result.get("latitude", ""),
                result.get("longitude", ""),
            )

            if identity in seen:
                continue

            seen.add(identity)

            all_results.append(result)

        # We already have good results from the cleaned query.
        # Continue variants only when result count is still low.
        if len(all_results) >= limit:
            break

    if not all_results:
        logger.info(
            "Location search returned no results for query: %s",
            query,
        )
        return []

    sorted_results = _sort_results(
        all_results,
        query,
    )

    results = sorted_results[:limit]

    logger.info(
        "Location search completed: query=%r results=%d",
        query,
        len(results),
    )

    return results


# ============================================================
# REVERSE GEOCODING
# ============================================================

def reverse_geocode(latitude, longitude) -> dict:
    """
    Convert coordinates into the best available address.
    """

    try:
        lat = float(latitude)
        lon = float(longitude)

        if not (-90 <= lat <= 90):
            return {}

        if not (-180 <= lon <= 180):
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

        result["source"] = (
            "OpenStreetMap Nominatim"
        )

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