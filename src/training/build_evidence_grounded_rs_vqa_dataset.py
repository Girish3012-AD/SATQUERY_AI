from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "remote_sensing"
    / "spacenet4"
    / "patches"
    / "Atlanta_743501_3721539"
    / "manifest.npy"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "remote_sensing"
    / "rs_vqa_evidence_grounded"
)

SEED = 20260913
VAL_RATIO = 0.20


def load_manifest() -> list[dict[str, Any]]:
    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Source manifest not found: {SOURCE_MANIFEST}"
        )

    raw = np.load(
        SOURCE_MANIFEST,
        allow_pickle=True,
    )

    records: list[dict[str, Any]] = []

    for item in raw.tolist():
        records.append(dict(item))

    return records


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_mask(record: dict[str, Any]) -> np.ndarray:
    mask_value = (
        record.get("mask")
        or record.get("mask_path")
        or record.get("label")
        or record.get("label_path")
    )

    if mask_value is None:
        raise KeyError(
            "Manifest record does not contain a mask path."
        )

    path = resolve_path(mask_value)

    if not path.exists():
        raise FileNotFoundError(path)

    mask = np.load(path)

    if mask.ndim != 2:
        raise ValueError(
            f"Expected 2D mask, got {mask.shape}"
        )

    return (mask > 0).astype(np.uint8)


def resolve_image(record: dict[str, Any]) -> Path:
    image_value = (
        record.get("image")
        or record.get("image_path")
    )

    if image_value is None:
        raise KeyError(
            "Manifest record does not contain an image path."
        )

    path = resolve_path(image_value)

    if not path.exists():
        raise FileNotFoundError(path)

    return path


def classify_density(
    fraction: float,
) -> str:
    if fraction == 0:
        return "no buildings"
    if fraction <= 0.01:
        return "very low building density"
    if fraction <= 0.05:
        return "low building density"
    if fraction <= 0.15:
        return "medium building density"
    return "high building density"


def classify_scene(
    fraction: float,
) -> str:
    if fraction == 0:
        return "non-built-up"
    if fraction <= 0.05:
        return "mostly non-built-up"
    if fraction <= 0.15:
        return "mixed built-up"
    return "mostly built-up"


def classify_coverage(
    fraction: float,
) -> str:
    if fraction == 0:
        return "no building coverage"
    if fraction <= 0.05:
        return "low building coverage"
    if fraction <= 0.15:
        return "moderate building coverage"
    if fraction <= 0.30:
        return "high building coverage"
    return "very high building coverage"


def build_examples(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        image_path = resolve_image(record)
        mask = load_mask(record)

        total_pixels = int(mask.size)
        building_pixels = int(mask.sum())

        fraction = (
            building_pixels / total_pixels
            if total_pixels
            else 0.0
        )

        percentage = fraction * 100.0

        density = classify_density(fraction)
        scene = classify_scene(fraction)
        coverage = classify_coverage(fraction)

        evidence = (
            "Remote-sensing building-mask evidence: "
            f"{building_pixels} of {total_pixels} pixels "
            f"are classified as building pixels "
            f"({percentage:.2f}% coverage). "
            f"The derived density category is '{density}'. "
            f"The derived scene category is '{scene}'."
        )

        # Use the source image/patch identity rather than the
        # local enumeration index. Train and validation are
        # generated independently, so a local index would
        # otherwise create duplicate IDs across splits.
        source_id = image_path.stem
        prefix = f"evidence_{source_id}"

        examples.extend(
            [
                {
                    "id": f"{prefix}_summary",
                    "image": str(image_path),
                    "question": (
                        "Using the image and the provided "
                        "remote-sensing evidence, summarize "
                        "the building situation in this patch."
                    ),
                    "evidence": evidence,
                    "answer": (
                        f"The patch has {percentage:.2f}% "
                        f"building coverage, corresponding "
                        f"to {density} and a {scene} scene."
                    ),
                    "task": "grounded_building_summary",
                    "source": "SpaceNet4",
                    "ground_truth": {
                        "building_pixels": building_pixels,
                        "total_pixels": total_pixels,
                        "building_fraction": fraction,
                        "building_percentage": percentage,
                        "density": density,
                        "scene": scene,
                        "coverage": coverage,
                    },
                },
                {
                    "id": f"{prefix}_density",
                    "image": str(image_path),
                    "question": (
                        "Based on the image and the supplied "
                        "building evidence, what is the "
                        "building density?"
                    ),
                    "evidence": evidence,
                    "answer": (
                        f"The building density is {density}."
                    ),
                    "task": "grounded_building_density",
                    "source": "SpaceNet4",
                    "ground_truth": {
                        "building_pixels": building_pixels,
                        "total_pixels": total_pixels,
                        "building_fraction": fraction,
                        "density": density,
                    },
                },
                {
                    "id": f"{prefix}_coverage",
                    "image": str(image_path),
                    "question": (
                        "Using the image together with the "
                        "remote-sensing evidence, describe "
                        "the building coverage."
                    ),
                    "evidence": evidence,
                    "answer": (
                        f"The building coverage is {coverage}, "
                        f"with {percentage:.2f}% of the patch "
                        "classified as building pixels."
                    ),
                    "task": "grounded_building_coverage",
                    "source": "SpaceNet4",
                    "ground_truth": {
                        "building_pixels": building_pixels,
                        "total_pixels": total_pixels,
                        "building_fraction": fraction,
                        "building_percentage": percentage,
                        "coverage": coverage,
                    },
                },
                {
                    "id": f"{prefix}_scene",
                    "image": str(image_path),
                    "question": (
                        "Considering both the image and the "
                        "provided building evidence, how would "
                        "you characterize this scene?"
                    ),
                    "evidence": evidence,
                    "answer": (
                        f"The scene is {scene}, with "
                        f"{percentage:.2f}% building coverage."
                    ),
                    "task": "grounded_scene_description",
                    "source": "SpaceNet4",
                    "ground_truth": {
                        "building_pixels": building_pixels,
                        "total_pixels": total_pixels,
                        "building_fraction": fraction,
                        "building_percentage": percentage,
                        "scene": scene,
                    },
                },
            ]
        )

    return examples


def split_by_patch(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(SEED)

    shuffled = list(records)
    rng.shuffle(shuffled)

    val_count = max(
        1,
        int(round(len(shuffled) * VAL_RATIO)),
    )

    val = shuffled[:val_count]
    train = shuffled[val_count:]

    return train, val


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    records = load_manifest()

    if not records:
        raise RuntimeError(
            "Source manifest contains no records."
        )

    train_patches, val_patches = split_by_patch(
        records
    )

    train_examples = build_examples(
        train_patches
    )

    val_examples = build_examples(
        val_patches
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        OUTPUT_ROOT / "train.jsonl",
        train_examples,
    )

    write_jsonl(
        OUTPUT_ROOT / "val.jsonl",
        val_examples,
    )

    metadata = {
        "phase": "9.4H.8B.1",
        "status": "DEVELOPMENT_DATASET",
        "dataset_name": (
            "SpaceNet4 Evidence-Grounded RS-VQA"
        ),
        "source_manifest": str(
            SOURCE_MANIFEST
        ),
        "source": "SpaceNet4",
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "train_patches": len(train_patches),
        "val_patches": len(val_patches),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "examples_per_patch": 4,
        "tasks": [
            "grounded_building_summary",
            "grounded_building_density",
            "grounded_building_coverage",
            "grounded_scene_description",
        ],
        "evidence_source": (
            "SpaceNet4 building masks"
        ),
        "exact_measurements_retained": True,
        "gis_measurements_outside_vlm": True,
        "development_only": True,
        "scientific_warning": (
            "Evidence is derived from SpaceNet4 ground-truth "
            "building masks. This dataset is for controlled "
            "development of evidence-grounded language "
            "generation and is not an independent benchmark."
        ),
    }

    metadata_path = (
        OUTPUT_ROOT / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("SOURCE PATCHES:", len(records))
    print("TRAIN PATCHES:", len(train_patches))
    print("VAL PATCHES:", len(val_patches))
    print("TRAIN EXAMPLES:", len(train_examples))
    print("VAL EXAMPLES:", len(val_examples))
    print()
    print("TRAIN:", OUTPUT_ROOT / "train.jsonl")
    print("VAL:", OUTPUT_ROOT / "val.jsonl")
    print("METADATA:", metadata_path)


if __name__ == "__main__":
    main()
