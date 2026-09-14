from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


INPUT_ROOT = Path("data/remote_sensing/rs_vqa_visual_binary")
OUTPUT_ROOT = Path("data/remote_sensing/rs_vqa_visual_binary_balanced")

TRAIN_INPUT = INPUT_ROOT / "train.jsonl"
VAL_INPUT = INPUT_ROOT / "val.jsonl"

TRAIN_OUTPUT = OUTPUT_ROOT / "train.jsonl"
VAL_OUTPUT = OUTPUT_ROOT / "val.jsonl"

SEED = 20260913


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def group_by_image(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)

    for record in records:
        groups[record["image"]].append(record)

    return dict(groups)


def main():
    rng = random.Random(SEED)

    print("=" * 60)
    print("PHASE 9.4H.9C.3A - BALANCED VISUAL TRAINING DATASET")
    print("=" * 60)

    train = load_jsonl(TRAIN_INPUT)
    val = load_jsonl(VAL_INPUT)

    train_groups = group_by_image(train)
    val_groups = group_by_image(val)

    positive_groups = []
    negative_groups = []

    for image, records in train_groups.items():
        labels = {r["answer"] for r in records}

        if labels == {"YES"}:
            positive_groups.append(image)
        elif labels == {"NO"}:
            negative_groups.append(image)
        else:
            raise ValueError(
                f"Mixed labels found for image: {image}"
            )

    print("Original training patches:")
    print("  YES patches:", len(positive_groups))
    print("  NO patches :", len(negative_groups))

    target_count = min(
        len(positive_groups),
        len(negative_groups),
    )

    rng.shuffle(positive_groups)
    rng.shuffle(negative_groups)

    selected_positive = positive_groups[:target_count]
    selected_negative = negative_groups[:target_count]

    selected_images = set(
        selected_positive + selected_negative
    )

    balanced_train = [
        record
        for record in train
        if record["image"] in selected_images
    ]

    rng.shuffle(balanced_train)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    TRAIN_OUTPUT.write_text(
        "\n".join(
            json.dumps(record)
            for record in balanced_train
        )
        + "\n",
        encoding="utf-8",
    )

    # Validation is copied unchanged.
    VAL_OUTPUT.write_text(
        "\n".join(
            json.dumps(record)
            for record in val
        )
        + "\n",
        encoding="utf-8",
    )

    yes_train = sum(
        r["answer"] == "YES"
        for r in balanced_train
    )

    no_train = sum(
        r["answer"] == "NO"
        for r in balanced_train
    )

    yes_val = sum(
        r["answer"] == "YES"
        for r in val
    )

    no_val = sum(
        r["answer"] == "NO"
        for r in val
    )

    print()
    print("Selected training patches:")
    print("  YES:", len(selected_positive))
    print("  NO :", len(selected_negative))

    print()
    print("Balanced training examples:")
    print("  YES:", yes_train)
    print("  NO :", no_train)
    print("  Total:", len(balanced_train))

    print()
    print("Validation examples:")
    print("  YES:", yes_val)
    print("  NO :", no_val)
    print("  Total:", len(val))

    print()
    print("Training image count:", len(selected_images))
    print("Validation image count:", len(val_groups))

    overlap = selected_images & set(val_groups)

    print("Train/validation image overlap:", len(overlap))

    print()
    print("VLM evidence supplied: False")
    print("Ground truth supplied to VLM: False")
    print("Training balance: 50/50")
    print("Validation distribution: unchanged")
    print()
    print("PHASE 9.4H.9C.3A BUILD COMPLETE")


if __name__ == "__main__":
    main()
