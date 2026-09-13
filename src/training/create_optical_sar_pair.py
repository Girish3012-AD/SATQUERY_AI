from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


SOURCE = Path(
    "data/remote_sensing/spacenet4/"
    "Pan-Sharpen_Atlanta_nadir53_catid_"
    "1030010003CD4300_743501_3721539.tif"
)

OUTPUT_DIR = Path(
    "data/remote_sensing/sar_dev"
)

OPTICAL = (
    OUTPUT_DIR /
    "optical.tif"
)

SAR = (
    OUTPUT_DIR /
    "sar.tif"
)


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(SOURCE) as src:

        optical = src.read()

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

    if optical.shape[0] != 4:
        raise ValueError(
            "Expected four optical bands."
        )

    optical_profile = profile.copy()

    optical_profile.update(
        count=4,
        dtype="uint16",
        nodata=0,
    )

    with rasterio.open(
        OPTICAL,
        "w",
        **optical_profile,
    ) as dst:

        dst.write(
            optical.astype(
                np.uint16
            )
        )

    # Development-only SAR surrogate.
    #
    # We derive a single-band radar-like intensity
    # representation from the optical scene solely
    # to exercise the multimodal plumbing.
    #
    # This is NOT RISAT data.
    sar_float = (
        0.35 * optical[0].astype(
            np.float32
        )
        + 0.30 * optical[1].astype(
            np.float32
        )
        + 0.20 * optical[2].astype(
            np.float32
        )
        + 0.15 * optical[3].astype(
            np.float32
        )
    )

    sar_float = np.maximum(
        sar_float,
        1.0,
    )

    sar_db = (
        10.0
        * np.log10(
            sar_float
            / np.median(sar_float)
        )
    )

    sar_profile = profile.copy()

    sar_profile.update(
        count=1,
        dtype="float32",
        nodata=None,
    )

    with rasterio.open(
        SAR,
        "w",
        **sar_profile,
    ) as dst:

        dst.write(
            sar_db.astype(
                np.float32
            ),
            1,
        )

    print(
        "Optical:",
        OPTICAL.resolve(),
    )

    print(
        "SAR surrogate:",
        SAR.resolve(),
    )

    print(
        "Optical shape:",
        optical.shape,
    )

    print(
        "SAR shape:",
        sar_db.shape,
    )

    print(
        "Optical CRS:",
        crs,
    )

    print(
        "Optical resolution:",
        (
            abs(transform.a),
            abs(transform.e),
        ),
    )

    print(
        "SAR range:",
        float(sar_db.min()),
        float(sar_db.max()),
    )

    print(
        "\nWARNING: SAR file is synthetic "
        "development data, NOT RISAT."
    )

    print(
        "\nSTEP 22H.02: PASS"
    )


if __name__ == "__main__":
    main()
