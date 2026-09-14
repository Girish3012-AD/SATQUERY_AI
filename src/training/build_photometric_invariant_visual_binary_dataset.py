from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance


INPUT_ROOT = Path(
    "data/remote_sensing/"
    "rs_vqa_visual_binary_balanced"
)

OUTPUT_ROOT = Path(
    "data/remote_sensing/"
    "rs_vqa_visual_binary_photometric"
)

TRAIN_INPUT = INPUT_ROOT / "train.jsonl"
VAL_INPUT = INPUT_ROOT / "val.jsonl"

TRAIN_OUTPUT = OUTPUT_ROOT / "train.jsonl"
VAL_OUTPUT = OUTPUT_ROOT / "val.jsonl"

IMAGE_ROOT = OUTPUT_ROOT / "images"

SEED = 20260913


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


def transform_image(image, rng):
    """
    Apply photometric changes only.

    Spatial structure is untouched.

    The same transformation distribution
    is applied independently to YES and NO
    examples, preventing the transformation
    itself from encoding the class.
    """

    brightness = rng.uniform(
        0.65,
        1.35,
    )

    contrast = rng.uniform(
        0.70,
        1.30,
    )

    color = rng.uniform(
        0.80,
        1.20,
    )

    image = ImageEnhance.Brightness(
        image
    ).enhance(brightness)

    image = ImageEnhance.Contrast(
        image
    ).enhance(contrast)

    image = ImageEnhance.Color(
        image
    ).enhance(color)

    # Small gamma adjustment implemented
    # directly on the RGB array.
    gamma = rng.uniform(
        0.80,
        1.20,
    )

    array = np.asarray(
        image
    ).astype(np.float32) / 255.0

    array = np.power(
        np.clip(
            array,
            0.0,
            1.0,
        ),
        gamma,
    )

    array = (
        np.clip(
            array * 255.0,
            0.0,
            255.0,
        )
        .astype(np.uint8)
    )

    return Image.fromarray(
        array,
        mode="RGB",
    )


def main():
    rng = random.Random(SEED)

    print("=" * 60)
    print(
        "PHASE 9.4H.9C.6A - "
        "PHOTOMETRIC-INVARIANT DATASET"
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

    IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    transformed_train = []

    for index, record in enumerate(train):
        source_image = Path(
            record["image"]
        )

        image = Image.open(
            source_image
        ).convert("RGB")

        transformed = transform_image(
            image,
            rng,
        )

        output_name = (
            f"{index:04d}_"
            f"{source_image.stem}_"
            f"photometric.png"
        )

        output_path = (
            IMAGE_ROOT / output_name
        )

        transformed.save(
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
            "RGB PNG with random "
            "photometric transform"
        )

        new_record[
            "photometric_augmentation"
        ] = {
            "brightness": True,
            "contrast": True,
            "color": True,
            "gamma": True,
            "spatial_geometry_changed": False,
        }

        transformed_train.append(
            new_record
        )

    # ------------------------------------------------------------
    # Validation is deliberately copied unchanged.
    # ------------------------------------------------------------

    save_jsonl(
        TRAIN_OUTPUT,
        transformed_train,
    )

    save_jsonl(
        VAL_OUTPUT,
        val,
    )

    print()
    print(
        "Transformed training records:",
        len(transformed_train),
    )

    print(
        "YES:",
        sum(
            r["answer"] == "YES"
            for r in transformed_train
        ),
    )

    print(
        "NO :",
        sum(
            r["answer"] == "NO"
            for r in transformed_train
        ),
    )

    print()
    print(
        "Photometric transform:"
    )

    print(
        "  Brightness: RANDOM"
    )

    print(
        "  Contrast  : RANDOM"
    )

    print(
        "  Color     : RANDOM"
    )

    print(
        "  Gamma     : RANDOM"
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
        "PHASE 9.4H.9C.6A BUILD COMPLETE"
    )


if __name__ == "__main__":
    main()
