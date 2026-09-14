from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("data/remote_sensing/rs_vqa_visual")
TRAIN_JSONL = ROOT / "train.jsonl"
VAL_JSONL = ROOT / "val.jsonl"

TRAIN_OUT = ROOT / "images_rgb" / "train"
VAL_OUT = ROOT / "images_rgb" / "val"


def stretch_band(band: np.ndarray) -> np.ndarray:
    band = band.astype(np.float32)

    valid = np.isfinite(band)

    if not valid.any():
        return np.zeros(band.shape, dtype=np.uint8)

    values = band[valid]

    low = np.percentile(values, 2.0)
    high = np.percentile(values, 98.0)

    if high <= low:
        return np.zeros(band.shape, dtype=np.uint8)

    scaled = (band - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)

    scaled[~valid] = 0.0

    return (scaled * 255.0).round().astype(np.uint8)


def convert_record(record: dict, output_dir: Path) -> dict:
    source_path = Path(record["image"])

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source image does not exist: {source_path}"
        )

    array = np.load(source_path)

    if array.ndim != 3:
        raise ValueError(
            f"Expected [bands,H,W], got {array.shape} "
            f"for {source_path}"
        )

    if array.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 bands for RGB conversion, "
            f"got {array.shape[0]} for {source_path}"
        )

    rgb = np.stack(
        [
            stretch_band(array[0]),
            stretch_band(array[1]),
            stretch_band(array[2]),
        ],
        axis=-1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{source_path.stem}.png"

    Image.fromarray(rgb, mode="RGB").save(
        output_path,
        format="PNG",
    )

    updated = dict(record)

    updated["source_raster_patch"] = str(source_path.resolve())
    updated["image"] = str(output_path.resolve())

    updated["image_representation"] = {
        "type": "rgb_png",
        "source_bands": [0, 1, 2],
        "source_band_count": int(array.shape[0]),
        "height": int(array.shape[1]),
        "width": int(array.shape[2]),
        "dtype_source": str(array.dtype),
        "normalization": "per-band 2nd-98th percentile stretch",
    }

    return updated


def process_split(input_jsonl: Path, output_jsonl: Path, output_dir: Path):
    records = [
        json.loads(line)
        for line in input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    updated_records = []

    for record in records:
        updated_records.append(
            convert_record(record, output_dir)
        )

    output_jsonl.write_text(
        "\n".join(
            json.dumps(record)
            for record in updated_records
        )
        + "\n",
        encoding="utf-8",
    )

    return records, updated_records


def main():
    print("=" * 60)
    print("PHASE 9.4H.9A.1 - VISUAL RGB REPRESENTATION")
    print("=" * 60)

    train_original, train_updated = process_split(
        TRAIN_JSONL,
        TRAIN_JSONL,
        TRAIN_OUT,
    )

    val_original, val_updated = process_split(
        VAL_JSONL,
        VAL_JSONL,
        VAL_OUT,
    )

    print()
    print("Train records:", len(train_updated))
    print("Validation records:", len(val_updated))

    print()
    print("Train RGB PNGs:", len(list(TRAIN_OUT.glob("*.png"))))
    print("Validation RGB PNGs:", len(list(VAL_OUT.glob("*.png"))))

    print()
    print("Sample train mapping:")
    print("  Original:", train_original[0]["image"])
    print("  RGB:", train_updated[0]["image"])
    print("  Source raster:", train_updated[0]["source_raster_patch"])

    print()
    print("Sample validation mapping:")
    print("  Original:", val_original[0]["image"])
    print("  RGB:", val_updated[0]["image"])
    print("  Source raster:", val_updated[0]["source_raster_patch"])

    print()
    print("PHASE 9.4H.9A.1 CONVERSION COMPLETE")


if __name__ == "__main__":
    main()
