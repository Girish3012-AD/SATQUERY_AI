import numpy as np
from scipy.ndimage import uniform_filter


class SensorAdapter:
    """Normalizes raw satellite sensor rasters into model-ready arrays."""

    @staticmethod
    def normalize_cartosat_optical(band_data: np.ndarray) -> np.ndarray:
        """
        Normalize high-bit-depth Cartosat-2S optical bands.

        Applies a 2%-98% percentile stretch to valid pixels and
        converts the model-ready representation to uint8.
        """
        valid_pixels = band_data[band_data > 0]

        if valid_pixels.size == 0:
            return np.zeros_like(band_data, dtype=np.uint8)

        p2, p98 = np.percentile(valid_pixels, (2, 98))

        stretched = np.clip(
            (band_data - p2) / (p98 - p2 + 1e-5),
            0,
            1,
        )

        return (stretched * 255).astype(np.uint8)

    @staticmethod
    def filter_risat_sar(
        sar_amplitude: np.ndarray,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """
        Apply a Lee-style local-statistics speckle filter to
        RISAT SAR amplitude data.

        The operation reduces local speckle while retaining
        stronger local structures.
        """
        img = sar_amplitude.astype(np.float32)

        mean = uniform_filter(img, size=kernel_size)
        mean_sq = uniform_filter(img ** 2, size=kernel_size)

        variance = np.maximum(
            mean_sq - mean ** 2,
            0,
        )

        overall_variance = np.var(img) + 1e-5

        weights = variance / (
            variance + overall_variance
        )

        filtered = mean + weights * (img - mean)

        return np.clip(filtered, 0, None).astype(np.float32)
