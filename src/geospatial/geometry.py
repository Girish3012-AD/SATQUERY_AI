from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


def bounds_to_geometry(
    bounds: tuple[float, float, float, float],
) -> BaseGeometry:
    """Convert raster bounds into a Shapely polygon."""

    left, bottom, right, top = bounds

    if left >= right or bottom >= top:
        raise ValueError("Invalid bounds.")

    return box(left, bottom, right, top)


def geometry_area(geometry: BaseGeometry) -> float:
    """Return geometry area in the geometry's coordinate units."""

    return float(geometry.area)


def geometry_distance(
    first: BaseGeometry,
    second: BaseGeometry,
) -> float:
    """Return distance between two geometries."""

    return float(first.distance(second))


def geometry_intersection(
    first: BaseGeometry,
    second: BaseGeometry,
) -> BaseGeometry:
    """Return the geometric intersection."""

    return first.intersection(second)


def geometry_buffer(
    geometry: BaseGeometry,
    distance: float,
) -> BaseGeometry:
    """Create a buffer around a geometry."""

    if distance < 0:
        raise ValueError("Buffer distance cannot be negative.")

    return geometry.buffer(distance)
