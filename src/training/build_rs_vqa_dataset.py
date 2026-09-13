from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MANIFEST = Path(
    "data/remote_sensing/spacenet4/patches/"
    "Atlanta_743501_3721539/manifest.npy"
)

DEFAULT_OUTPUT = Path("data/remote_sensing/rs_vqa")

SEED = 20260913
VAL_RATIO = 0.20


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest does not exist: {manifest_path}"
        )

    records = np.load(
        manifest_path,
        allow_pickle=True,
    ).tolist()

    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain a non-empty list.")

    return records


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert a [C,H,W] float image to an RGB uint8 image.

    The source patches are four-band SpaceNet optical data.
    We create a display representation only. The original
    four-band arrays remain unchanged.
    """
    if image.ndim != 3:
        raise ValueError(
            f"Expected [C,H,W], got {image.shape}"
        )

    channels = image.shape[0]

    if channels < 3:
        raise ValueError(
            "At least three channels are required for RGB conversion."
        )

    rgb = image[:3].astype(np.float32, copy=False)

    output = np.zeros_like(rgb, dtype=np.float32)

    for index in range(3):
        band = rgb[index]

        finite = np.isfinite(band)

        if not finite.any():
            continue

        values = band[finite]

        low = float(np.percentile(values, 2))
        high = float(np.percentile(values, 98))

        if high <= low:
            output[index] = np.clip(band, 0.0, 1.0)
        else:
            output[index] = (
                (band - low) / (high - low)
            )
            output[index] = np.clip(
                output[index],
                0.0,
                1.0,
            )

    return np.transpose(
        (output * 255.0).round().astype(np.uint8),
        (1, 2, 0),
    )


def density_label(building_fraction: float) -> str:
    if building_fraction == 0.0:
        return "no buildings"

    if building_fraction < 0.01:
        return "very low"

    if building_fraction < 0.05:
        return "low"

    if building_fraction < 0.15:
        return "medium"

    return "high"


def scene_label(building_fraction: float) -> str:
    if building_fraction == 0.0:
        return "non-built-up"

    if building_fraction < 0.05:
        return "mostly non-built-up"

    if building_fraction < 0.15:
        return "mixed built-up"

    return "mostly built-up"


def make_examples(
    record: dict[str, Any],
    image_path: str,
    split: str,
) -> list[dict[str, Any]]:
    patch_id = str(record["patch_id"])

    building_pixels = int(record["building_pixels"])
    building_fraction = float(record["building_fraction"])

    if building_pixels > 0:
        presence_answer = "Yes, buildings are present."
    else:
        presence_answer = "No, no buildings are present."

    density = density_label(building_fraction)

    coverage_answer = (
        f"Approximately "
        f"{building_fraction * 100.0:.1f}% "
        f"of the image contains buildings."
    )

    scene = scene_label(building_fraction)

    examples = [
        {
            "id": f"{patch_id}_presence",
            "image": image_path,
            "question": "Are there any buildings in this image?",
            "answer": presence_answer,
            "task": "building_presence",
            "source": "SpaceNet4",
            "split": split,
        },
        {
            "id": f"{patch_id}_density",
            "image": image_path,
            "question": "What is the building density in this image?",
            "answer": f"The building density is {density}.",
            "task": "building_density",
            "source": "SpaceNet4",
            "split": split,
        },
        {
            "id": f"{patch_id}_coverage",
            "image": image_path,
            "question": (
                "Approximately what fraction of the image "
                "contains buildings?"
            ),
            "answer": coverage_answer,
            "task": "building_coverage",
            "source": "SpaceNet4",
            "split": split,
        },
        {
            "id": f"{patch_id}_scene",
            "image": image_path,
            "question": (
                "Is this scene mostly built-up or "
                "mostly non-built-up?"
            ),
            "answer": f"The scene is {scene}.",
            "task": "scene_type",
            "source": "SpaceNet4",
            "split": split,
        },
    ]

    for example in examples:
        example["ground_truth"] = {
            "building_pixels": building_pixels,
            "building_fraction": building_fraction,
        }

    return examples


def build_dataset(
    manifest_path: Path = DEFAULT_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT,
    seed: int = SEED,
    val_ratio: float = VAL_RATIO,
) -> dict[str, Any]:
    records = load_manifest(manifest_path)

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_output_dir = output_dir / "images"
    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shuffled = list(records)

    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_count = max(
        1,
        int(round(len(shuffled) * val_ratio)),
    )

    validation_records = shuffled[:val_count]
    training_records = shuffled[val_count:]

    train_examples: list[dict[str, Any]] = []
    val_examples: list[dict[str, Any]] = []

    for record, split in [
        *[(r, "train") for r in training_records],
        *[(r, "val") for r in validation_records],
    ]:
        source_image = Path(record["image"])

        if not source_image.exists():
            raise FileNotFoundError(
                f"Patch image does not exist: {source_image}"
            )

        image = np.load(source_image)

        if image.shape != (4, 64, 64):
            raise ValueError(
                f"Unexpected image shape for "
                f"{source_image}: {image.shape}"
            )

        if image.dtype != np.float32:
            raise ValueError(
                f"Unexpected image dtype for "
                f"{source_image}: {image.dtype}"
            )

        if not np.isfinite(image).all():
            raise ValueError(
                f"Image contains non-finite values: "
                f"{source_image}"
            )

        rgb = normalize_to_uint8(image)

        output_image = image_output_dir / (
            f"{record['patch_id']}.png"
        )

        Image.fromarray(
            rgb,
            mode="RGB",
        ).save(output_image)

        relative_image = str(
            output_image
        )

        examples = make_examples(
            record,
            relative_image,
            split,
        )

        if split == "train":
            train_examples.extend(examples)
        else:
            val_examples.extend(examples)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with train_path.open("w", encoding="utf-8") as handle:
        for example in train_examples:
            handle.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with val_path.open("w", encoding="utf-8") as handle:
        for example in val_examples:
            handle.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

    train_patch_ids = {
        example["id"].split("_")[0]
        for example in train_examples
    }

    val_patch_ids = {
        example["id"].split("_")[0]
        for example in val_examples
    }

    overlap = train_patch_ids & val_patch_ids

    if overlap:
        raise RuntimeError(
            f"Train/validation patch leakage: {sorted(overlap)}"
        )

    metadata = {
        "dataset": "SpaceNet4-derived RS-VQA development dataset",
        "source_manifest": str(manifest_path),
        "seed": seed,
        "val_ratio": val_ratio,
        "total_patches": len(records),
        "training_patches": len(training_records),
        "validation_patches": len(validation_records),
        "training_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "questions_per_patch": 4,
        "question_tasks": [
            "building_presence",
            "building_density",
            "building_coverage",
            "scene_type",
        ],
        "grounding_source": "SpaceNet4 building masks",
        "image_representation": (
            "RGB visualization generated from first three "
            "optical channels; original four-band arrays "
            "remain unchanged"
        ),
        "train_validation_patch_overlap": len(overlap),
        "development_only": True,
    }

    with (output_dir / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    return metadata


if __name__ == "__main__":
    metadata = build_dataset()

    print("RS-VQA DATASET CREATED")
    print(json.dumps(metadata, indent=2))
