from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


INPUT_ROOT = Path(
    "data/remote_sensing/rs_vqa_visual_binary"
)

OUTPUT_ROOT = Path(
    "data/remote_sensing/rs_vqa_visual_binary_brightness_matched"
)

TRAIN_INPUT = INPUT_ROOT / "train.jsonl"
VAL_INPUT = INPUT_ROOT / "val.jsonl"

TRAIN_OUTPUT = OUTPUT_ROOT / "train.jsonl"
VAL_OUTPUT = OUTPUT_ROOT / "val.jsonl"

SEED = 20260913


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def image_features(path):
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype(np.float32)

    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def group_by_image(records):
    groups = {}

    for record in records:
        groups.setdefault(
            record["image"],
            [],
        ).append(record)

    return groups


def choose_matched_pairs(
    yes_images,
    no_images,
):
    """
    Greedy nearest-neighbour matching.

    Each YES patch is matched to one unused NO patch
    using normalized distance in mean/std space.
    """

    yes_features = {
        image: image_features(image)
        for image in yes_images
    }

    no_features = {
        image: image_features(image)
        for image in no_images
    }

    # Estimate global scales so mean and std contribute
    # comparably to the distance.
    all_features = list(
        yes_features.values()
    ) + list(
        no_features.values()
    )

    mean_values = [
        x["mean"]
        for x in all_features
    ]

    std_values = [
        x["std"]
        for x in all_features
    ]

    mean_scale = max(
        np.std(mean_values),
        1.0,
    )

    std_scale = max(
        np.std(std_values),
        1.0,
    )

    remaining_no = set(no_images)

    pairs = []

    # Match the most unusual YES examples first.
    ordered_yes = sorted(
        yes_images,
        key=lambda image: (
            abs(
                yes_features[image]["mean"]
                - np.mean(mean_values)
            )
            / mean_scale
            +
            abs(
                yes_features[image]["std"]
                - np.mean(std_values)
            )
            / std_scale
        ),
        reverse=True,
    )

    for yes_image in ordered_yes:

        if not remaining_no:
            break

        y = yes_features[yes_image]

        best_no = min(
            remaining_no,
            key=lambda no_image: (
                (
                    (
                        no_features[no_image]["mean"]
                        - y["mean"]
                    )
                    / mean_scale
                ) ** 2
                +
                (
                    (
                        no_features[no_image]["std"]
                        - y["std"]
                    )
                    / std_scale
                ) ** 2
            ),
        )

        pairs.append(
            (
                yes_image,
                best_no,
            )
        )

        remaining_no.remove(best_no)

    return pairs


def main():
    random.seed(SEED)

    print("=" * 60)
    print(
        "PHASE 9.4H.9C.5B - "
        "BRIGHTNESS-MATCHED DATASET"
    )
    print("=" * 60)

    train = load_jsonl(TRAIN_INPUT)
    val = load_jsonl(VAL_INPUT)

    train_groups = group_by_image(train)
    val_groups = group_by_image(val)

    yes_images = []
    no_images = []

    for image, records in train_groups.items():
        labels = {
            record["answer"]
            for record in records
        }

        if labels == {"YES"}:
            yes_images.append(image)

        elif labels == {"NO"}:
            no_images.append(image)

        else:
            raise ValueError(
                f"Unexpected labels for {image}: "
                f"{labels}"
            )

    print(
        "Original YES images:",
        len(yes_images),
    )

    print(
        "Original NO images:",
        len(no_images),
    )

    pair_count = min(
        len(yes_images),
        len(no_images),
    )

    # Randomly reduce the larger class before matching.
    random.shuffle(yes_images)
    random.shuffle(no_images)

    yes_images = yes_images[:pair_count]
    no_images = no_images[:pair_count]

    pairs = choose_matched_pairs(
        yes_images,
        no_images,
    )

    if len(pairs) != pair_count:
        raise RuntimeError(
            "Failed to construct complete "
            "YES/NO matching."
        )

    selected_images = {
        image
        for pair in pairs
        for image in pair
    }

    matched_train = [
        record
        for record in train
        if record["image"] in selected_images
    ]

    random.shuffle(matched_train)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRAIN_OUTPUT.write_text(
        "\n".join(
            json.dumps(record)
            for record in matched_train
        )
        + "\n",
        encoding="utf-8",
    )

    # Validation remains completely untouched.
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
        for r in matched_train
    )

    no_train = sum(
        r["answer"] == "NO"
        for r in matched_train
    )

    # Measure pair quality.
    pair_distances = []

    for yes_image, no_image in pairs:
        yf = image_features(yes_image)
        nf = image_features(no_image)

        distance = (
            abs(yf["mean"] - nf["mean"])
            + abs(yf["std"] - nf["std"])
        )

        pair_distances.append(distance)

    print()
    print("Matched pairs:", len(pairs))

    print(
        "Training YES examples:",
        yes_train,
    )

    print(
        "Training NO examples:",
        no_train,
    )

    print(
        "Training total:",
        len(matched_train),
    )

    print(
        "Mean pair brightness/std distance:",
        float(np.mean(pair_distances)),
    )

    print(
        "Median pair brightness/std distance:",
        float(np.median(pair_distances)),
    )

    print(
        "Maximum pair brightness/std distance:",
        float(np.max(pair_distances)),
    )

    print()
    print(
        "Validation copied unchanged:",
        len(val),
        "examples",
    )

    print(
        "VLM evidence supplied: False"
    )

    print(
        "Ground truth supplied: False"
    )

    print(
        "Training representation: original RGB PNG"
    )

    print(
        "Matching variables: mean brightness + image std"
    )

    print()
    print(
        "PHASE 9.4H.9C.5B BUILD COMPLETE"
    )


if __name__ == "__main__":
    main()
