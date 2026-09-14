from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ORIGINAL_ROOT = Path(
    "data/remote_sensing/"
    "rs_vqa_visual_binary_balanced"
)

PHOTOMETRIC_ROOT = Path(
    "data/remote_sensing/"
    "rs_vqa_visual_binary_photometric"
)

ORIGINAL_TRAIN = (
    ORIGINAL_ROOT / "train.jsonl"
)

PHOTOMETRIC_TRAIN = (
    PHOTOMETRIC_ROOT / "train.jsonl"
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
        "mean_pixel_std": float(
            np.std(means)
        ),
        "mean_image_std": float(
            np.mean(stds)
        ),
    }


def evaluate_dataset(name, records):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    for label in [
        "YES",
        "NO",
    ]:

        s = summarize(
            records,
            label,
        )

        print()
        print(label)

        print(
            "  count:",
            s["count"],
        )

        print(
            "  mean pixel:",
            s["mean_pixel"],
        )

        print(
            "  pixel-mean std:",
            s["mean_pixel_std"],
        )

        print(
            "  mean image std:",
            s["mean_image_std"],
        )

    yes_summary = summarize(
        records,
        "YES",
    )

    no_summary = summarize(
        records,
        "NO",
    )

    brightness_gap = abs(
        yes_summary["mean_pixel"]
        -
        no_summary["mean_pixel"]
    )

    image_std_gap = abs(
        yes_summary["mean_image_std"]
        -
        no_summary["mean_image_std"]
    )

    shortcut = (
        best_brightness_threshold(
            records
        )
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
        "Best brightness threshold:",
        shortcut["threshold"],
    )

    print(
        "YES when below:",
        shortcut["yes_when_below"],
    )

    print(
        "Brightness-only accuracy:",
        shortcut["accuracy"],
    )

    return {
        "brightness_gap": brightness_gap,
        "image_std_gap": image_std_gap,
        "brightness_only_accuracy": (
            shortcut["accuracy"]
        ),
        "threshold": shortcut[
            "threshold"
        ],
        "yes_when_below": shortcut[
            "yes_when_below"
        ],
    }


def main():

    print("=" * 60)
    print(
        "PHASE 9.4H.9C.6C - "
        "PHOTOMETRIC SHORTCUT CONTROL"
    )
    print("=" * 60)

    original_records = load_jsonl(
        ORIGINAL_TRAIN
    )

    photometric_records = load_jsonl(
        PHOTOMETRIC_TRAIN
    )

    assert len(
        original_records
    ) == 268

    assert len(
        photometric_records
    ) == 268

    original = extract(
        original_records
    )

    photometric = extract(
        photometric_records
    )

    original_result = evaluate_dataset(
        "ORIGINAL BALANCED DATASET",
        original,
    )

    photometric_result = evaluate_dataset(
        "PHOTOMETRIC DATASET",
        photometric,
    )

    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print()
    print(
        "Original brightness-only accuracy:",
        original_result[
            "brightness_only_accuracy"
        ],
    )

    print(
        "Photometric brightness-only accuracy:",
        photometric_result[
            "brightness_only_accuracy"
        ],
    )

    reduction = (
        original_result[
            "brightness_only_accuracy"
        ]
        -
        photometric_result[
            "brightness_only_accuracy"
        ]
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
        "Photometric brightness gap:",
        photometric_result[
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
        "Photometric image-std gap:",
        photometric_result[
            "image_std_gap"
        ],
    )

    # ------------------------------------------------------------
    # Scientific gate
    #
    # We don't demand perfection.
    #
    # We require the photometric transformation to
    # materially reduce the brightness shortcut.
    # ------------------------------------------------------------

    accuracy_pass = (
        photometric_result[
            "brightness_only_accuracy"
        ]
        <= 0.60
    )

    reduction_pass = (
        reduction >= 0.10
    )

    if (
        accuracy_pass
        and reduction_pass
    ):

        status = (
            "PHOTOMETRIC_CONTROL_PASS"
        )

    else:

        status = (
            "PHOTOMETRIC_CONTROL_FAIL"
        )

    print()
    print(
        "Brightness-only accuracy <= 60%:",
        accuracy_pass,
    )

    print(
        "Shortcut reduction >= 10 points:",
        reduction_pass,
    )

    print()
    print(
        "STATUS:",
        status,
    )

    if status != (
        "PHOTOMETRIC_CONTROL_PASS"
    ):

        print()
        print(
            "DO NOT TRAIN QWEN."
        )

        raise SystemExit(2)

    print()
    print(
        "PHASE 9.4H.9C.6C: PASS"
    )


if __name__ == "__main__":
    main()
