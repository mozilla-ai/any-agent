import json

import requests


def get_area_lat_lon(area_name: str) -> tuple[float, float]:
    """Get the latitude and longitude of an area from Nominatim.

    Uses the [Nominatim API](https://nominatim.org/release-docs/develop/api/Search/).

    Args:
        area_name: The name of the area.

    Returns:
        The area found.

    """
    response = requests.get(
        f"https://nominatim.openstreetmap.org/search?q={area_name}&format=jsonv2",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    area = json.loads(response.content.decode())
    return area[0]["lat"], area[0]["lon"]


def driving_hours_to_meters(driving_hours: int) -> int:
    """Convert driving hours to meters assuming a 70 km/h average speed.

    Args:
        driving_hours: The driving hours.

    Returns:
        The distance in meters.

    """
    return driving_hours * 70 * 1000


def get_lat_lon_center(bounds: dict) -> tuple[float, float]:
    """Get the latitude and longitude of the center of a bounding box.

    Args:
        bounds: The bounding box.

            ```json
            {
                "minlat": float,
                "minlon": float,
                "maxlat": float,
                "maxlon": float,
            }
            ```

    Returns:
        The latitude and longitude of the center.

    """
    return (
        (bounds["minlat"] + bounds["maxlat"]) / 2,
        (bounds["minlon"] + bounds["maxlon"]) / 2,
    )
