# core/services/weather_service.py

from __future__ import annotations

import logging
import math
from typing import Any

import requests


logger = logging.getLogger(__name__)


# ============================================================
# OPEN-METEO ENDPOINTS
# ============================================================

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

REQUEST_TIMEOUT = 15


# ============================================================
# WEATHER CODES
# ============================================================

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",

    45: "Fog",
    48: "Rime fog",

    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",

    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",

    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",

    66: "Light freezing rain",
    67: "Heavy freezing rain",

    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",

    77: "Snow grains",

    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",

    85: "Slight snow showers",
    86: "Heavy snow showers",

    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


# ============================================================
# COMMON CITY ALIASES
# ============================================================

CITY_ALIASES = {
    # Bengali
    "কলকাতা": "Kolkata",
    "কলকাতার": "Kolkata",
    "কোলকাতা": "Kolkata",
    "কোলকাতার": "Kolkata",

    "দিল্লি": "Delhi",
    "দিল্লির": "Delhi",

    "নয়াদিল্লি": "New Delhi",
    "নয়াদিল্লি": "New Delhi",

    "মুম্বাই": "Mumbai",
    "মুম্বাইয়ের": "Mumbai",
    "মুম্বাইয়ের": "Mumbai",

    "চেন্নাই": "Chennai",
    "চেন্নাইয়ের": "Chennai",
    "চেন্নাইয়ের": "Chennai",

    "ব্যাঙ্গালোর": "Bengaluru",
    "ব্যাঙ্গালোরের": "Bengaluru",
    "বেঙ্গালুরু": "Bengaluru",
    "বেঙ্গালুরুর": "Bengaluru",

    "হায়দ্রাবাদ": "Hyderabad",
    "হায়দ্রাবাদ": "Hyderabad",
    "হায়দ্রাবাদের": "Hyderabad",
    "হায়দ্রাবাদের": "Hyderabad",

    "পুনে": "Pune",
    "পুনের": "Pune",

    "হালিশহর": "Halisahar",
    "হালিশহরের": "Halisahar",

    # English
    "calcutta": "Kolkata",
    "kolkata": "Kolkata",

    "delhi": "Delhi",
    "new delhi": "New Delhi",

    "mumbai": "Mumbai",
    "bombay": "Mumbai",

    "chennai": "Chennai",
    "madras": "Chennai",

    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",

    "hyderabad": "Hyderabad",
    "pune": "Pune",

    "halisahar": "Halisahar",
}


# ============================================================
# HTTP / JSON
# ============================================================

def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    Open-Meteo endpoint থেকে JSON আনে এবং response validate করে।
    """

    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.Timeout as exc:
        logger.warning(
            "Open-Meteo request timed out: %s",
            url,
        )
        raise RuntimeError(
            "Weather request timed out"
        ) from exc

    except requests.RequestException as exc:
        logger.warning(
            "Open-Meteo request failed: %s",
            type(exc).__name__,
        )
        raise RuntimeError(
            "Could not connect to the weather service"
        ) from exc

    try:
        data = response.json()

    except ValueError as exc:
        logger.warning(
            "Open-Meteo returned invalid JSON"
        )
        raise RuntimeError(
            "Weather service returned invalid JSON"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            "Weather service returned invalid data"
        )

    if data.get("error") is True:
        reason = data.get(
            "reason",
            "Unknown API error",
        )

        logger.warning(
            "Open-Meteo API error: %s",
            reason,
        )

        raise RuntimeError(
            f"Weather service error: {reason}"
        )

    return data


# ============================================================
# NUMERIC VALIDATION
# ============================================================

def _to_finite_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    """
    Value-কে finite float-এ convert করে।
    """

    if value is None:
        return default

    if isinstance(value, bool):
        return default

    try:
        number = float(value)

    except (TypeError, ValueError, OverflowError):
        return default

    if not math.isfinite(number):
        return default

    return number


def _valid_coordinate(
    value: Any,
    minimum: float,
    maximum: float,
) -> bool:
    """
    Latitude/longitude valid কি না পরীক্ষা করে।
    """

    number = _to_finite_float(value)

    if number is None:
        return False

    return minimum <= number <= maximum


# ============================================================
# CITY NORMALIZATION
# ============================================================

def normalize_weather_city(city: str) -> str:
    """
    Common Bengali / English city aliases normalize করে।

    গুরুত্বপূর্ণ:
    এই function natural-language পুরো question parse করে না।
    পরিষ্কার city/location name-ই এখানে পাঠানো উচিত।
    """

    if not isinstance(city, str):
        raise ValueError("City name must be text")

    city = " ".join(city.strip().split())

    if not city:
        raise ValueError(
            "Please provide a city name"
        )

    # Exact alias match first.
    alias = CITY_ALIASES.get(
        city.casefold()
    )

    if alias:
        return alias

    # Bengali exact match যেখানে casefold প্রাসঙ্গিক নয়।
    if city in CITY_ALIASES:
        return CITY_ALIASES[city]

    return city


# ============================================================
# LOCATION SELECTION
# ============================================================

def _location_score(
    item: dict[str, Any],
    requested_city: str,
) -> int:
    """
    Geocoding results-এর মধ্যে সবচেয়ে relevant result বাছাই করে।
    """

    score = 0

    name = item.get("name")

    if isinstance(name, str):
        if name.casefold() == requested_city.casefold():
            score += 100

        elif requested_city.casefold() in name.casefold():
            score += 30

    feature_code = item.get("feature_code")

    # PPL/PPLC/PPLA জাতীয় populated-place result-কে
    # সাধারণত অগ্রাধিকার দেওয়া যায়।
    if isinstance(feature_code, str):
        if feature_code.upper() == "PPLC":
            score += 20
        elif feature_code.upper().startswith("PPLA"):
            score += 15
        elif feature_code.upper().startswith("PPL"):
            score += 10

    country_code = item.get("country_code")

    # Hello AI-এর প্রধান target India হলে Indian result-কে
    # সামান্য preference।
    if isinstance(country_code, str):
        if country_code.upper() == "IN":
            score += 5

    population = item.get("population")

    try:
        population = int(population or 0)
    except (TypeError, ValueError, OverflowError):
        population = 0

    if population > 0:
        score += min(
            10,
            int(math.log10(population + 1)),
        )

    return score


def _select_location(
    locations: list[Any],
    requested_city: str,
) -> dict[str, Any]:
    """
    Valid geocoding result থেকে best location নির্বাচন করে।
    """

    valid_locations: list[dict[str, Any]] = []

    for item in locations:
        if not isinstance(item, dict):
            continue

        if not _valid_coordinate(
            item.get("latitude"),
            -90,
            90,
        ):
            continue

        if not _valid_coordinate(
            item.get("longitude"),
            -180,
            180,
        ):
            continue

        valid_locations.append(item)

    if not valid_locations:
        raise RuntimeError(
            "Location coordinates are unavailable"
        )

    return max(
        valid_locations,
        key=lambda item: _location_score(
            item,
            requested_city,
        ),
    )


# ============================================================
# WEATHER
# ============================================================

def get_weather(city: str) -> dict[str, Any]:
    """
    Open-Meteo ব্যবহার করে একটি location-এর current weather আনে।

    Return:
        {
            "city": ...,
            "country": ...,
            "temperature_c": ...,
            "feels_like_c": ...,
            "humidity_percent": ...,
            "wind_speed_kmh": ...,
            "precipitation_mm": ...,
            "condition": ...,
            "time": ...,
            "timezone": ...,
            "units": ...,
            "is_day": ...
        }
    """

    if not isinstance(city, str):
        raise ValueError(
            "City name must be text"
        )

    city = normalize_weather_city(city)

    if not city:
        raise ValueError(
            "Please provide a city name"
        )

    # --------------------------------------------------------
    # GEOCODING
    # --------------------------------------------------------

    geo_data = _get_json(
        GEOCODING_URL,
        {
            "name": city[:120],
            "count": 10,
            "language": "en",
            "format": "json",
        },
    )

    locations = geo_data.get(
        "results",
        [],
    )

    if (
        not isinstance(locations, list)
        or not locations
    ):
        raise ValueError(
            f"Could not find the location: {city}"
        )

    location = _select_location(
        locations,
        city,
    )

    latitude = _to_finite_float(
        location.get("latitude")
    )

    longitude = _to_finite_float(
        location.get("longitude")
    )

    if latitude is None or longitude is None:
        raise RuntimeError(
            "Location coordinates are unavailable"
        )

    # --------------------------------------------------------
    # CURRENT WEATHER
    # --------------------------------------------------------

    weather_data = _get_json(
        WEATHER_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "apparent_temperature,"
                "is_day,"
                "precipitation,"
                "weather_code,"
                "wind_speed_10m"
            ),
            "timezone": "auto",
            "forecast_days": 1,
        },
    )

    current = weather_data.get(
        "current"
    )

    if not isinstance(current, dict):
        raise RuntimeError(
            "Current weather data is unavailable"
        )

    if not current:
        raise RuntimeError(
            "Current weather data is unavailable"
        )

    # --------------------------------------------------------
    # WEATHER CODE
    # --------------------------------------------------------

    code_value = current.get(
        "weather_code"
    )

    try:
        weather_code = int(code_value)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        weather_code = None

    # --------------------------------------------------------
    # UNITS
    # --------------------------------------------------------

    units = weather_data.get(
        "current_units",
        {},
    )

    if not isinstance(units, dict):
        units = {}

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {
        "city": location.get(
            "name",
            city,
        ),

        "country": location.get(
            "country",
            "",
        ),

        "country_code": location.get(
            "country_code",
            "",
        ),

        "admin1": location.get(
            "admin1",
            "",
        ),

        "latitude": latitude,
        "longitude": longitude,

        "temperature_c": _to_finite_float(
            current.get("temperature_2m")
        ),

        "feels_like_c": _to_finite_float(
            current.get("apparent_temperature")
        ),

        "humidity_percent": _to_finite_float(
            current.get(
                "relative_humidity_2m"
            )
        ),

        "wind_speed_kmh": _to_finite_float(
            current.get(
                "wind_speed_10m"
            )
        ),

        "precipitation_mm": _to_finite_float(
            current.get(
                "precipitation"
            )
        ),

        "weather_code": weather_code,

        "condition": WEATHER_CODES.get(
            weather_code,
            "Unknown conditions",
        ),

        "time": current.get(
            "time"
        ),

        "timezone": weather_data.get(
            "timezone"
        ),

        "is_day": current.get(
            "is_day"
        ),

        "units": units,
    }


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "get_weather",
    "normalize_weather_city",
    "WEATHER_CODES",
]