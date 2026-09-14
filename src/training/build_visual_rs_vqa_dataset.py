from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


SEED = 20260913
VAL_RATIO = 0.20

SOURCE_MANIFEST = Path(
    "data/remote_sensing/spacenet4/patches/"
    "Atlanta_743501_3721539/manifest.npy"
)

OUTPUT_DIR = Path("data/remote_sensing/rs_vqa_visual")

TRAIN_JSONL = OUTPUT_DIR / "train.jsonl"
VAL_JSONL = OUTPUT_DIR / "val.jsonl"
METADATA_JSON = OUTPUT_DIR / "metadata.json"

QUESTIONS = [
    {
        "task": "visual_building_presence",
        "question": "Are buildings visible in this remote-sensing image patch?",
    },
    {
        "task": "visual_built_environment",
        "question": "Does this remote-sensing image patch contain a built environment?",
    },
]


def load_manifest() -> list[dict[str, Any]]:
    records = np.load(SOURCE_MANIFEST, allow_pickle=True)

    result: list[dict[str, Any]] = []

    for record in records:
        if hasattr(record, "item"):
            record = record.item()

        result.append(dict(record))

    return result


def building_fraction(mask_path: Path) -> float:
    mask = np.load(mask_path)

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")

    return float((mask > 0).mean())


def answer_from_fraction(fraction: float) -> tuple[str, str]:
    if fraction > 0.0:
        return (
            "Buildings are visible.",
            "positive",
        )

    return (
        "No buildings are visible.",
        "negative",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main() -> None:
    random.seed(SEED)

    records = load_manifest()

    if len(records) != 196:
        raise ValueError(
            f"Expected 196 source patches, found {len(records)}"
        )

    enriched = []

    for record in records:
        image_path = Path(record["image"])
        mask_path = Path(record["mask"])

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        fraction = building_fraction(mask_path)

        answer, visual_class = answer_from_fraction(fraction)

        enriched.append(
            {
                "source_id": image_path.stem,
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
                "building_fraction": fraction,
                "visual_class": visual_class,
                "answer": answer,
            }
        )

    random.shuffle(enriched)

    val_count = round(len(enriched) * VAL_RATIO)

    validation = enriched[:val_count]
    training = enriched[val_count:]

    train_examples = []
    val_examples = []

    for split_records, output in [
        (training, train_examples),
        (validation, val_examples),
    ]:
        for source in split_records:
            for question in QUESTIONS:
                output.append(
                    {
                        "id": (
                            f"visual_{source['source_id']}_"
                            f"{question['task']}"
                        ),
                        "image": source["image"],
                        "question": question["question"],
                        "answer": source["answer"],
                        "task": question["task"],
                        "source": "SpaceNet4",
                        "split": (
                            "train"
                            if output is train_examples
                            else "validation"
                        ),
                        "visual_class": source["visual_class"],
                        "ground_truth": {
                            "building_fraction": source[
                                "building_fraction"
                            ],
                            "visual_class": source["visual_class"],
                        },
                        "evidence": None,
                        "development_only": True,
                        "image_only_target": True,
                    }
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(TRAIN_JSONL, train_examples)
    write_jsonl(VAL_JSONL, val_examples)

    metadata = {
        "phase": "9.4H.9A",
        "purpose": "genuine_visual_discrimination",
        "source": "SpaceNet4",
        "seed": SEED,
        "source_patches": len(enriched),
        "train_patches": len(training),
        "validation_patches": len(validation),
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "questions_per_patch": len(QUESTIONS),
        "evidence_supplied_to_vlm": False,
        "image_only_target": True,
        "development_only": True,
        "independent_benchmark": False,
        "warning": (
            "Development experiment from a single SpaceNet4 scene. "
            "This dataset is intended to test visual dependence and "
            "does not constitute an independent benchmark."
        ),
    }

    with METADATA_JSON.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Train patches:", len(training))
    print("Validation patches:", len(validation))
    print("Train examples:", len(train_examples))
    print("Validation examples:", len(val_examples))
    print("Evidence supplied to VLM:", False)
    print("Image-only target:", True)
    print("Dataset build: PASS")


if __name__ == "__main__":
    main()
