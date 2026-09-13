from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


PATCH_SIZE = 64
STRIDE = 64


def normalize_patch(image: np.ndarray) -> np.ndarray:
    """
    Normalize each band independently using the same
    percentile preprocessing used by the building pipeline.
    """
    image = image.astype(np.float32, copy=False)

    output = np.zeros_like(image, dtype=np.float32)

    for band_index in range(image.shape[0]):
        band = image[band_index]

        low = float(np.percentile(band, 2))
        high = float(np.percentile(band, 98))

        if high <= low:
            output[band_index] = 0.0
        else:
            output[band_index] = np.clip(
                (band - low) / (high - low),
                0.0,
                1.0,
            )

    return output


def generate_change_pair(
    image_path: str,
    output_dir: str,
    *,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
) -> dict:
    """
    Generate controlled bi-temporal development data from a real
    optical remote-sensing scene.

    T1 is the original scene.

    T2 contains deterministic rectangular structural changes.

    The change mask exactly records those injected regions.

    This dataset is for pipeline/model development and is NOT a
    public benchmark.
    """
    source = Path(image_path)

    if not source.exists():
        raise FileNotFoundError(
            f"Source raster does not exist: {source}"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    with rasterio.open(source) as dataset:
        if dataset.count != 4:
            raise ValueError(
                f"Expected 4 bands, found {dataset.count}"
            )

        if dataset.crs is None:
            raise ValueError(
                "Source raster must have a valid CRS."
            )

        image = dataset.read()

        height = dataset.height
        width = dataset.width

        if height < patch_size or width < patch_size:
            raise ValueError(
                "Raster is smaller than requested patch size."
            )

        rng = np.random.default_rng(20260913)

        records = []
        patch_id = 0

        for top in range(0, height - patch_size + 1, stride):
            for left in range(0, width - patch_size + 1, stride):

                t1_raw = image[
                    :,
                    top:top + patch_size,
                    left:left + patch_size,
                ]

                t2_raw = t1_raw.copy()

                change_mask = np.zeros(
                    (patch_size, patch_size),
                    dtype=np.uint8,
                )

                # Roughly half of the patches contain controlled change.
                if rng.random() < 0.5:
                    rect_h = int(
                        rng.integers(
                            patch_size // 4,
                            patch_size // 2,
                        )
                    )

                    rect_w = int(
                        rng.integers(
                            patch_size // 4,
                            patch_size // 2,
                        )
                    )

                    max_top = patch_size - rect_h
                    max_left = patch_size - rect_w

                    rect_top = int(
                        rng.integers(0, max_top + 1)
                    )

                    rect_left = int(
                        rng.integers(0, max_left + 1)
                    )

                    rect_bottom = rect_top + rect_h
                    rect_right = rect_left + rect_w

                    # Controlled spectral change.
                    # Apply the same spatial footprint across bands
                    # with different amplitudes.
                    multipliers = np.array(
                        [1.35, 1.25, 1.15, 1.45],
                        dtype=np.float32,
                    )

                    for band_index in range(4):
                        region = t2_raw[
                            band_index,
                            rect_top:rect_bottom,
                            rect_left:rect_right,
                        ].astype(np.float32)

                        region = region * multipliers[band_index]

                        t2_raw[
                            band_index,
                            rect_top:rect_bottom,
                            rect_left:rect_right,
                        ] = np.clip(
                            region,
                            0,
                            np.iinfo(t2_raw.dtype).max,
                        ).astype(t2_raw.dtype)

                    change_mask[
                        rect_top:rect_bottom,
                        rect_left:rect_right,
                    ] = 1

                t1 = normalize_patch(t1_raw)
                t2 = normalize_patch(t2_raw)

                image_file = (
                    destination
                    / f"pair_{patch_id:04d}_t1.npy"
                )

                after_file = (
                    destination
                    / f"pair_{patch_id:04d}_t2.npy"
                )

                mask_file = (
                    destination
                    / f"pair_{patch_id:04d}_mask.npy"
                )

                np.save(image_file, t1.astype(np.float32))
                np.save(after_file, t2.astype(np.float32))
                np.save(
                    mask_file,
                    change_mask.astype(np.float32),
                )

                records.append(
                    {
                        "patch_id": patch_id,
                        "t1": str(image_file),
                        "t2": str(after_file),
                        "mask": str(mask_file),
                        "changed": bool(change_mask.any()),
                        "change_pixels": int(change_mask.sum()),
                    }
                )

                patch_id += 1

    manifest = destination / "manifest.npy"
    np.save(
        manifest,
        np.array(records, dtype=object),
    )

    changed = sum(
        1 for record in records
        if record["changed"]
    )

    return {
        "source": str(source.resolve()),
        "output_dir": str(destination.resolve()),
        "patch_count": len(records),
        "changed_patch_count": changed,
        "unchanged_patch_count": len(records) - changed,
        "patch_size": patch_size,
        "stride": stride,
        "development_only": True,
        "label_method": "controlled_synthetic_spectral_change",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    result = generate_change_pair(
        args.image,
        args.output,
    )

    print(result)
