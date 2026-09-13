from shapely.geometry.base import BaseGeometry

from .geometry import (
    geometry_area,
    geometry_buffer,
    geometry_distance,
    geometry_intersection,
)


def buffer_geometry(
    geometry: BaseGeometry,
    distance: float,
) -> BaseGeometry:
    """Create a deterministic spatial buffer."""
    return geometry_buffer(geometry, distance)


def intersect_geometries(
    first: BaseGeometry,
    second: BaseGeometry,
) -> BaseGeometry:
    """Calculate deterministic spatial intersection."""
    return geometry_intersection(first, second)


def calculate_area(
    geometry: BaseGeometry,
) -> float:
    """Calculate geometry area."""
    return geometry_area(geometry)


def calculate_distance(
    first: BaseGeometry,
    second: BaseGeometry,
) -> float:
    """Calculate distance between geometries."""
    return geometry_distance(first, second)
