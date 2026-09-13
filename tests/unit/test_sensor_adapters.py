import numpy as np

from src.geospatial.sensor_adapters import SensorAdapter


def test_normalize_cartosat_optical():
    optical_data = np.array(
        [0, 100, 500, 1000, 5000],
        dtype=np.uint16,
    )

    normalized = SensorAdapter.normalize_cartosat_optical(
        optical_data
    )

    assert normalized.dtype == np.uint8
    assert normalized.max() <= 255
    assert normalized.min() >= 0
    assert normalized.shape == optical_data.shape


def test_filter_risat_sar():
    np.random.seed(42)

    sar_data = np.random.uniform(
        10,
        100,
        (20, 20),
    ).astype(np.float32)

    filtered = SensorAdapter.filter_risat_sar(
        sar_data,
        kernel_size=3,
    )

    assert filtered.shape == (20, 20)
    assert filtered.dtype == np.float32
    assert filtered.min() >= 0.0
