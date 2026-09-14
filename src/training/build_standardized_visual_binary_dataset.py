from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


INPUT_ROOT = Path(
    "data/remote_sensing/"
    "rs_vqa_visual_binary_balanced"
)

OUTPUT_ROOT = Path(
    "data/remote_sensing/"
    "rs_vqa_visual_binary_standardized"
)

TRAIN_INPUT = INPUT_ROOT / "train.jsonl"
VAL_INPUT = INPUT_ROOT / "val.jsonl"

TRAIN_OUTPUT = OUTPUT_ROOT / "train.jsonl"
VAL_OUTPUT = OUTPUT_ROOT / "val.jsonl"

IMAGE_ROOT = OUTPUT_ROOT / "images"

TARGET_MEAN = 128.0
TARGET_STD = 50.0


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def save_jsonl(path, records):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in records
        )
        + "\n",
        encoding="utf-8",
    )


def standardize_image(path):

    image = Image.open(
        path
    ).convert("RGB")

    array = np.asarray(
        image
    ).astype(np.float32)

    output = np.zeros_like(
        array,
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # Normalize each RGB channel independently.
    #
    # This removes per-image global brightness and contrast
    # while preserving spatial arrangement.
    # ------------------------------------------------------------

    for channel in range(3):

        values = array[:, :, channel]

        mean = float(
            values.mean()
        )

        std = float(
            values.std()
        )

        if std < 1e-6:
            normalized = (
                np.zeros_like(values)
                + TARGET_MEAN
            )

        else:
            normalized = (
                (
                    values - mean
                )
                / std
            ) * TARGET_STD + TARGET_MEAN

        output[:, :, channel] = (
            normalized
        )

    output = np.clip(
        output,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        output,
        mode="RGB",
    )


def main():

    print("=" * 60)
    print(
        "PHASE 9.4H.9C.6D - "
        "PER-IMAGE STANDARDIZED DATASET"
    )
    print("=" * 60)

    train = load_jsonl(
        TRAIN_INPUT
    )

    val = load_jsonl(
        VAL_INPUT
    )

    print(
        "Original training records:",
        len(train),
    )

    print(
        "Validation records:",
        len(val),
    )

    if len(train) != 268:
        raise RuntimeError(
            "Training dataset must contain "
            "268 records."
        )

    if len(val) != 78:
        raise RuntimeError(
            "Validation dataset must contain "
            "78 records."
        )

    IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    standardized_train = []

    for index, record in enumerate(train):

        source_image = Path(
            record["image"]
        )

        standardized = standardize_image(
            source_image
        )

        output_name = (
            f"{index:04d}_"
            f"{source_image.stem}_"
            f"standardized.png"
        )

        output_path = (
            IMAGE_ROOT / output_name
        )

        standardized.save(
            output_path,
            format="PNG",
        )

        new_record = dict(record)

        new_record["image"] = str(
            output_path
        )

        new_record[
            "original_rgb_image"
        ] = str(
            source_image
        )

        new_record[
            "image_representation"
        ] = (
            "RGB PNG with "
            "per-image channel "
            "standardization"
        )

        new_record[
            "standardization"
        ] = {
            "per_channel": True,
            "target_mean": TARGET_MEAN,
            "target_std": TARGET_STD,
            "spatial_geometry_changed": False,
        }

        standardized_train.append(
            new_record
        )

    # ------------------------------------------------------------
    # Validation is intentionally unchanged.
    # ------------------------------------------------------------

    save_jsonl(
        TRAIN_OUTPUT,
        standardized_train,
    )

    save_jsonl(
        VAL_OUTPUT,
        val,
    )

    yes = sum(
        r["answer"] == "YES"
        for r in standardized_train
    )

    no = sum(
        r["answer"] == "NO"
        for r in standardized_train
    )

    print()
    print(
        "Standardized training records:",
        len(standardized_train),
    )

    print(
        "YES:",
        yes,
    )

    print(
        "NO :",
        no,
    )

    print()
    print(
        "Normalization:"
    )

    print(
        "  Per-channel: TRUE"
    )

    print(
        "  Target mean:",
        TARGET_MEAN,
    )

    print(
        "  Target std:",
        TARGET_STD,
    )

    print(
        "  Spatial geometry changed: FALSE"
    )

    print()
    print(
        "VLM evidence supplied: False"
    )

    print(
        "Ground truth supplied: False"
    )

    print(
        "Validation modified: False"
    )

    print()
    print(
        "Training dataset:",
        TRAIN_OUTPUT,
    )

    print(
        "Validation dataset:",
        VAL_OUTPUT,
    )

    print()
    print(
        "PHASE 9.4H.9C.6D BUILD COMPLETE"
    )


if __name__ == "__main__":
    main()
