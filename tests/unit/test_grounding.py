from affine import Affine
import pytest

from src.geospatial.grounding import pixel_bbox_to_geographic_polygon


def test_pixel_bbox_to_geographic_polygon():
    transform = Affine.translation(1000, 2000) * Affine.scale(10, -10)

    polygon = pixel_bbox_to_geographic_polygon(
        (10, 20, 30, 40),
        transform,
    )

    assert polygon.is_valid
    assert polygon.geom_type == "Polygon"

    assert polygon.bounds == (
        1100.0,
        1600.0,
        1300.0,
        1800.0,
    )


def test_invalid_bbox_length():
    transform = Affine.identity()

    with pytest.raises(ValueError):
        pixel_bbox_to_geographic_polygon(
            (1, 2, 3),
            transform,
        )


def test_invalid_bbox_order():
    transform = Affine.identity()

    with pytest.raises(ValueError):
        pixel_bbox_to_geographic_polygon(
            (30, 20, 10, 40),
            transform,
        )


def test_polygon_area_matches_pixel_resolution():
    transform = Affine.translation(0, 0) * Affine.scale(10, -10)

    polygon = pixel_bbox_to_geographic_polygon(
        (0, 0, 20, 10),
        transform,
    )

    # 20 pixels × 10 pixels × 10m × 10m = 20,000 m²
    assert polygon.area == pytest.approx(20_000.0)
