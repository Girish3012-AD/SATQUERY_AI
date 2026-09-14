from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ORIGINAL_ROOT = Path(
    "data/remote_sensing/"
    "rs_vqa_visual_binary_balanced"
)

STANDARDIZED_ROOT = Path(
    "data/remote_sensing/"
    "rs_vqa_visual_binary_standardized"
)

ORIGINAL_TRAIN = (
    ORIGINAL_ROOT / "train.jsonl"
)

STANDARDIZED_TRAIN = (
    STANDARDIZED_ROOT / "train.jsonl"
)


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def image_stats(path):
    image = Image.open(
        path
    ).convert("RGB")

    arr = np.asarray(
        image
    ).astype(np.float32)

    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def extract(records):
    result = []

    for record in records:

        stats = image_stats(
            record["image"]
        )

        result.append(
            {
                "answer": record["answer"],
                "mean": stats["mean"],
                "std": stats["std"],
            }
        )

    return result


def threshold_accuracy(
    items,
    threshold,
    yes_when_below,
):
    correct = 0

    for item in items:

        if yes_when_below:

            prediction = (
                "YES"
                if item["mean"] < threshold
                else "NO"
            )

        else:

            prediction = (
                "YES"
                if item["mean"] >= threshold
                else "NO"
            )

        if prediction == item["answer"]:
            correct += 1

    return correct / len(items)


def best_brightness_threshold(items):

    values = sorted(
        {
            item["mean"]
            for item in items
        }
    )

    candidates = []

    if values:

        candidates.append(
            values[0] - 1e-6
        )

    for a, b in zip(
        values,
        values[1:],
    ):

        candidates.append(
            (a + b) / 2.0
        )

    if values:

        candidates.append(
            values[-1] + 1e-6
        )

    best = None

    for threshold in candidates:

        for yes_when_below in [
            True,
            False,
        ]:

            accuracy = threshold_accuracy(
                items,
                threshold,
                yes_when_below,
            )

            candidate = {
                "threshold": threshold,
                "yes_when_below": (
                    yes_when_below
                ),
                "accuracy": accuracy,
            }

            if (
                best is None
                or accuracy
                > best["accuracy"]
            ):

                best = candidate

    return best


def summarize(items, label):

    selected = [
        item
        for item in items
        if item["answer"] == label
    ]

    means = [
        item["mean"]
        for item in selected
    ]

    stds = [
        item["std"]
        for item in selected
    ]

    return {
        "count": len(selected),
        "mean_pixel": float(
            np.mean(means)
        ),
        "pixel_mean_std": float(
            np.std(means)
        ),
        "mean_image_std": float(
            np.mean(stds)
        ),
    }


def evaluate(name, items):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    yes = summarize(
        items,
        "YES",
    )

    no = summarize(
        items,
        "NO",
    )

    print()
    print("YES")

    print(
        "  count:",
        yes["count"],
    )

    print(
        "  mean pixel:",
        yes["mean_pixel"],
    )

    print(
        "  pixel-mean std:",
        yes["pixel_mean_std"],
    )

    print(
        "  mean image std:",
        yes["mean_image_std"],
    )

    print()
    print("NO")

    print(
        "  count:",
        no["count"],
    )

    print(
        "  mean pixel:",
        no["mean_pixel"],
    )

    print(
        "  pixel-mean std:",
        no["pixel_mean_std"],
    )

    print(
        "  mean image std:",
        no["mean_image_std"],
    )

    brightness_gap = abs(
        yes["mean_pixel"]
        -
        no["mean_pixel"]
    )

    image_std_gap = abs(
        yes["mean_image_std"]
        -
        no["mean_image_std"]
    )

    best = best_brightness_threshold(
        items
    )

    print()
    print(
        "Brightness gap:",
        brightness_gap,
    )

    print(
        "Image-std gap:",
        image_std_gap,
    )

    print(
        "Best threshold:",
        best["threshold"],
    )

    print(
        "YES when below:",
        best["yes_when_below"],
    )

    print(
        "Brightness-only accuracy:",
        best["accuracy"],
    )

    return {
        "brightness_gap": brightness_gap,
        "image_std_gap": image_std_gap,
        "brightness_only_accuracy": (
            best["accuracy"]
        ),
    }


def main():

    print("=" * 60)
    print(
        "PHASE 9.4H.9C.6E - "
        "STANDARDIZATION SHORTCUT TEST"
    )
    print("=" * 60)

    original_records = load_jsonl(
        ORIGINAL_TRAIN
    )

    standardized_records = load_jsonl(
        STANDARDIZED_TRAIN
    )

    assert len(
        original_records
    ) == 268

    assert len(
        standardized_records
    ) == 268

    original = extract(
        original_records
    )

    standardized = extract(
        standardized_records
    )

    original_result = evaluate(
        "ORIGINAL BALANCED DATASET",
        original,
    )

    standardized_result = evaluate(
        "STANDARDIZED DATASET",
        standardized,
    )

    original_accuracy = (
        original_result[
            "brightness_only_accuracy"
        ]
    )

    standardized_accuracy = (
        standardized_result[
            "brightness_only_accuracy"
        ]
    )

    reduction = (
        original_accuracy
        -
        standardized_accuracy
    )

    print()
    print("=" * 60)
    print("SHORTCUT COMPARISON")
    print("=" * 60)

    print()
    print(
        "Original brightness-only accuracy:",
        original_accuracy,
    )

    print(
        "Standardized brightness-only accuracy:",
        standardized_accuracy,
    )

    print(
        "Shortcut accuracy reduction:",
        reduction,
    )

    print()
    print(
        "Original brightness gap:",
        original_result[
            "brightness_gap"
        ],
    )

    print(
        "Standardized brightness gap:",
        standardized_result[
            "brightness_gap"
        ],
    )

    print()
    print(
        "Original image-std gap:",
        original_result[
            "image_std_gap"
        ],
    )

    print(
        "Standardized image-std gap:",
        standardized_result[
            "image_std_gap"
        ],
    )

    # ------------------------------------------------------------
    # Scientific gate
    #
    # Standardization should make brightness
    # substantially less predictive.
    #
    # We require:
    #
    #   standardized accuracy <= 55%
    #   AND
    #   shortcut reduction >= 15 points
    #
    # This is intentionally stricter than C.6C.
    # ------------------------------------------------------------

    accuracy_pass = (
        standardized_accuracy
        <= 0.55
    )

    reduction_pass = (
        reduction
        >= 0.15
    )

    print()
    print(
        "Standardized brightness accuracy <= 55%:",
        accuracy_pass,
    )

    print(
        "Shortcut reduction >= 15 points:",
        reduction_pass,
    )

    if (
        accuracy_pass
        and reduction_pass
    ):

        print()
        print(
            "STATUS: "
            "STANDARDIZATION_CONTROL_PASS"
        )

        print()
        print(
            "PHASE 9.4H.9C.6E: PASS"
        )

    else:

        print()
        print(
            "STATUS: "
            "STANDARDIZATION_CONTROL_FAIL"
        )

        print()
        print(
            "DO NOT TRAIN QWEN."
        )

        raise SystemExit(2)


if __name__ == "__main__":
    main()
