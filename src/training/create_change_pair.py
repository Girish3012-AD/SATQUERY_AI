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
    "data/remote_sensing/change_dev"
)

BEFORE = OUTPUT_DIR / "realistic_before.tif"
AFTER = OUTPUT_DIR / "realistic_after.tif"
GT = OUTPUT_DIR / "realistic_change_gt.tif"


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(SOURCE) as src:
        image = src.read()
        profile = src.profile.copy()

    if image.shape[0] != 4:
        raise ValueError(
            f"Expected 4 bands, got {image.shape[0]}"
        )

    before = image.copy()

    after = image.copy()

    height, width = before.shape[1:]

    # Controlled rectangular development area.
    # This is synthetic development data, not benchmark data.
    y0 = height // 2 - 96
    y1 = height // 2 + 96
    x0 = width // 2 - 96
    x1 = width // 2 + 96

    after[:, y0:y1, x0:x1] = (
        after[:, y0:y1, x0:x1].astype(np.float32)
        * np.array(
            [1.35, 1.25, 1.15, 1.45],
            dtype=np.float32,
        )[:, None, None]
    ).clip(
        0,
        np.iinfo(image.dtype).max,
    ).astype(image.dtype)

    gt = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    gt[y0:y1, x0:x1] = 1

    profile.update(
        dtype=str(image.dtype),
        count=4,
        nodata=0,
    )

    with rasterio.open(
        BEFORE,
        "w",
        **profile,
    ) as dst:
        dst.write(before)

    with rasterio.open(
        AFTER,
        "w",
        **profile,
    ) as dst:
        dst.write(after)

    gt_profile = profile.copy()
    gt_profile.update(
        dtype="uint8",
        count=1,
        nodata=0,
    )

    with rasterio.open(
        GT,
        "w",
        **gt_profile,
    ) as dst:
        dst.write(gt, 1)

    print("=== SYNTHETIC CHANGE PAIR ===")
    print("Before:", BEFORE.resolve())
    print("After:", AFTER.resolve())
    print("Ground truth:", GT.resolve())
    print("Shape:", before.shape)
    print("Changed GT pixels:", int(gt.sum()))
    print(
        "GT percentage:",
        float(gt.mean() * 100.0),
    )
    print("CRS:", profile["crs"])

    print("\nSTEP 22G.11: PASS")


if __name__ == "__main__":
    main()
