from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(
    "data/remote_sensing/rs_vqa_visual_binary_balanced"
)

TRAIN = ROOT / "train.jsonl"
VAL = ROOT / "val.jsonl"


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def mean_pixel(path):
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype(np.float32)

    return float(arr.mean())


def extract(records):
    return [
        {
            "answer": r["answer"],
            "mean": mean_pixel(r["image"]),
            "image": r["image"],
        }
        for r in records
    ]


def threshold_accuracy(items, threshold, yes_when_below):
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


def find_best_threshold(train):
    values = sorted(
        {item["mean"] for item in train}
    )

    candidates = []

    if values:
        candidates.append(values[0] - 1e-6)

    for a, b in zip(values, values[1:]):
        candidates.append((a + b) / 2.0)

    if values:
        candidates.append(values[-1] + 1e-6)

    best = None

    for threshold in candidates:
        for yes_when_below in [True, False]:

            accuracy = threshold_accuracy(
                train,
                threshold,
                yes_when_below,
            )

            candidate = {
                "threshold": threshold,
                "yes_when_below": yes_when_below,
                "accuracy": accuracy,
            }

            if (
                best is None
                or candidate["accuracy"]
                > best["accuracy"]
            ):
                best = candidate

    return best


def summarize(items, label):
    values = [
        item["mean"]
        for item in items
        if item["answer"] == label
    ]

    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def main():
    print("=" * 60)
    print("PHASE 9.4H.9C.5A - BRIGHTNESS SHORTCUT TEST")
    print("=" * 60)

    train_records = load_jsonl(TRAIN)
    val_records = load_jsonl(VAL)

    train = extract(train_records)
    val = extract(val_records)

    print("Training examples:", len(train))
    print("Validation examples:", len(val))

    print()
    print("TRAINING BRIGHTNESS")
    print("-" * 60)

    for label in ["YES", "NO"]:
        s = summarize(train, label)

        print(label)
        print("  count :", s["count"])
        print("  mean  :", s["mean"])
        print("  std   :", s["std"])
        print("  min   :", s["min"])
        print("  max   :", s["max"])
        print("  median:", s["median"])

    print()
    print("VALIDATION BRIGHTNESS")
    print("-" * 60)

    for label in ["YES", "NO"]:
        s = summarize(val, label)

        print(label)
        print("  count :", s["count"])
        print("  mean  :", s["mean"])
        print("  std   :", s["std"])
        print("  min   :", s["min"])
        print("  max   :", s["max"])
        print("  median:", s["median"])

    print()
    print("BEST TRAINING BRIGHTNESS THRESHOLD")
    print("-" * 60)

    best = find_best_threshold(train)

    print("Threshold:", best["threshold"])
    print(
        "YES when mean is below threshold:",
        best["yes_when_below"],
    )

    train_accuracy = best["accuracy"]

    val_accuracy = threshold_accuracy(
        val,
        best["threshold"],
        best["yes_when_below"],
    )

    print(
        "Training accuracy:",
        train_accuracy,
    )

    print(
        "Validation accuracy:",
        val_accuracy,
    )

    print()
    print("INTERPRETATION")

    if train_accuracy >= 0.70:
        print(
            "Brightness-only training shortcut: STRONG"
        )
    elif train_accuracy >= 0.60:
        print(
            "Brightness-only training shortcut: MODERATE"
        )
    else:
        print(
            "Brightness-only training shortcut: WEAK"
        )

    if (
        train_accuracy - val_accuracy
        >= 0.15
    ):
        print(
            "Train/validation brightness shift: STRONG"
        )
    else:
        print(
            "Train/validation brightness shift: LIMITED"
        )

    print()
    print("PHASE 9.4H.9C.5A COMPLETE")


if __name__ == "__main__":
    main()
