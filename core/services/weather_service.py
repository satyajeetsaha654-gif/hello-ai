"""Weather service for Hello AI using Open-Meteo."""

import logging

import requests


logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

REQUEST_TIMEOUT = 15

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


def _get_json(url: str, params: dict) -> dict:
    """Request and validate JSON from an Open-Meteo endpoint."""

    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

    except requests.Timeout as exc:
        logger.warning("Open-Meteo request timed out")
        raise RuntimeError("Weather request timed out") from exc

    except requests.RequestException as exc:
        logger.warning(
            "Open-Meteo request failed (%s)",
            type(exc).__name__,
        )
        raise RuntimeError(
            "Could not connect to the weather service"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Weather service returned invalid JSON"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            "Weather service returned invalid data"
        )

    if data.get("error"):
        reason = data.get("reason", "Unknown API error")
        logger.warning("Open-Meteo reported an API error")
        raise RuntimeError(
            f"Weather service error: {reason}"
        )

    return data


def _valid_coordinate(value, minimum, maximum) -> bool:
    """Check whether a coordinate is numeric and in range."""

    if isinstance(value, bool):
        return False

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False

    return (
        minimum <= number <= maximum
        and number == number
        and abs(number) != float("inf")
    )


def get_weather(city: str) -> dict:
    """Get current weather for a city using Open-Meteo."""

    if not isinstance(city, str):
        raise ValueError("City name must be text")

    city = city.strip()

    if not city:
        raise ValueError("Please provide a city name")

    geo_data = _get_json(
        GEOCODING_URL,
        {
            "name": city[:120],
            "count": 10,
            "language": "en",
            "format": "json",
        },
    )

    locations = geo_data.get("results", [])

    if not isinstance(locations, list) or not locations:
        raise ValueError(f"Could not find the location: {city}")

    valid_locations = [
        item
        for item in locations
        if isinstance(item, dict)
        and _valid_coordinate(
            item.get("latitude"), -90, 90
        )
        and _valid_coordinate(
            item.get("longitude"), -180, 180
        )
    ]

    if not valid_locations:
        raise RuntimeError(
            "Location coordinates are unavailable"
        )

    # Prefer a case-insensitive exact city-name match.
    location = next(
        (
            item
            for item in valid_locations
            if isinstance(item.get("name"), str)
            and item["name"].casefold() == city.casefold()
        ),
        valid_locations[0],
    )

    latitude = float(location["latitude"])
    longitude = float(location["longitude"])

    weather_data = _get_json(
        WEATHER_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "apparent_temperature,is_day,precipitation,"
                "weather_code,wind_speed_10m"
            ),
            "timezone": "auto",
            "forecast_days": 1,
        },
    )

    current = weather_data.get("current")

    if not isinstance(current, dict) or not current:
        raise RuntimeError(
            "Current weather data is unavailable"
        )

    units = weather_data.get("current_units", {})

    if not isinstance(units, dict):
        units = {}

    code = current.get("weather_code")

    try:
        code = int(code)
    except (TypeError, ValueError, OverflowError):
        code = None

    return {
        "city": location.get("name", city),
        "country": location.get("country", ""),
        "temperature_c": current.get("temperature_2m"),
        "feels_like_c": current.get("apparent_temperature"),
        "humidity_percent": current.get("relative_humidity_2m"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "precipitation_mm": current.get("precipitation"),
        "condition": WEATHER_CODES.get(
            code, "Unknown conditions"
        ),
        "time": current.get("time"),
        "timezone": weather_data.get("timezone"),
        "units": units,
        "is_day": current.get("is_day"),
    }