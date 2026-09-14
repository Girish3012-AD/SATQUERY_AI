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

OUTPUT = Path(
    "outputs/checkpoints/"
    "qwen2vl_rs_vqa_visual_binary_dev/"
    "visual_dataset_diagnostic.json"
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
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype(np.float32)

    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "channel_mean": [
            float(arr[:, :, i].mean())
            for i in range(3)
        ],
        "channel_std": [
            float(arr[:, :, i].std())
            for i in range(3)
        ],
    }


def summarize(records):
    grouped = {
        "YES": [],
        "NO": [],
    }

    for record in records:
        stats = image_stats(record["image"])

        grouped[record["answer"]].append(
            {
                "id": record["id"],
                **stats,
            }
        )

    result = {}

    for label, values in grouped.items():
        result[label] = {
            "count": len(values),
            "mean_of_means": float(
                np.mean(
                    [v["mean"] for v in values]
                )
            ),
            "mean_of_stds": float(
                np.mean(
                    [v["std"] for v in values]
                )
            ),
            "mean_channel_means": [
                float(
                    np.mean(
                        [
                            v["channel_mean"][i]
                            for v in values
                        ]
                    )
                )
                for i in range(3)
            ],
            "mean_channel_stds": [
                float(
                    np.mean(
                        [
                            v["channel_std"][i]
                            for v in values
                        ]
                    )
                )
                for i in range(3)
            ],
        }

    return result


def main():
    print("=" * 60)
    print("PHASE 9.4H.9C.5 - VISUAL DATASET DIAGNOSTIC")
    print("=" * 60)

    train = load_jsonl(TRAIN)
    val = load_jsonl(VAL)

    print("Training records:", len(train))
    print("Validation records:", len(val))

    train_summary = summarize(train)
    val_summary = summarize(val)

    print()
    print("TRAINING IMAGE STATISTICS")
    print("-" * 60)

    for label in ["YES", "NO"]:
        print(label)
        print(
            "  count:",
            train_summary[label]["count"],
        )
        print(
            "  mean pixel:",
            train_summary[label]["mean_of_means"],
        )
        print(
            "  mean image std:",
            train_summary[label]["mean_of_stds"],
        )
        print(
            "  channel means:",
            train_summary[label]["mean_channel_means"],
        )
        print(
            "  channel stds:",
            train_summary[label]["mean_channel_stds"],
        )

    print()
    print("VALIDATION IMAGE STATISTICS")
    print("-" * 60)

    for label in ["YES", "NO"]:
        print(label)
        print(
            "  count:",
            val_summary[label]["count"],
        )
        print(
            "  mean pixel:",
            val_summary[label]["mean_of_means"],
        )
        print(
            "  mean image std:",
            val_summary[label]["mean_of_stds"],
        )
        print(
            "  channel means:",
            val_summary[label]["mean_channel_means"],
        )
        print(
            "  channel stds:",
            val_summary[label]["mean_channel_stds"],
        )

    # ------------------------------------------------------------
    # Paired-image consistency
    # ------------------------------------------------------------

    print()
    print("QUESTION / IMAGE CONSISTENCY")
    print("-" * 60)

    by_image = {}

    for record in train + val:
        by_image.setdefault(
            record["image"],
            [],
        ).append(record)

    image_label_counts = {
        "one_record": 0,
        "two_records": 0,
        "other": 0,
    }

    same_label_pairs = 0
    different_label_pairs = 0

    for image, records in by_image.items():
        if len(records) == 1:
            image_label_counts["one_record"] += 1

        elif len(records) == 2:
            image_label_counts["two_records"] += 1

            labels = {
                r["answer"]
                for r in records
            }

            if len(labels) == 1:
                same_label_pairs += 1
            else:
                different_label_pairs += 1

        else:
            image_label_counts["other"] += 1

    print(
        "Images with one record:",
        image_label_counts["one_record"],
    )

    print(
        "Images with two records:",
        image_label_counts["two_records"],
    )

    print(
        "Images with other record counts:",
        image_label_counts["other"],
    )

    print(
        "Two-record images with same label:",
        same_label_pairs,
    )

    print(
        "Two-record images with different labels:",
        different_label_pairs,
    )

    # ------------------------------------------------------------
    # Source patch metadata
    # ------------------------------------------------------------

    print()
    print("SOURCE PATCH DISTRIBUTION")
    print("-" * 60)

    source_groups = {
        "YES": set(),
        "NO": set(),
    }

    for record in train:
        source_groups[
            record["answer"]
        ].add(record["source_raster_patch"])

    print(
        "Unique YES source patches:",
        len(source_groups["YES"]),
    )

    print(
        "Unique NO source patches:",
        len(source_groups["NO"]),
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    result = {
        "phase": "9.4H.9C.5",
        "train_records": len(train),
        "validation_records": len(val),
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "image_record_distribution": image_label_counts,
        "same_label_pairs": same_label_pairs,
        "different_label_pairs": different_label_pairs,
        "unique_yes_source_patches": len(
            source_groups["YES"]
        ),
        "unique_no_source_patches": len(
            source_groups["NO"]
        ),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Diagnostic saved:", OUTPUT)

    print()
    print(
        "PHASE 9.4H.9C.5 DIAGNOSTIC COMPLETE"
    )


if __name__ == "__main__":
    main()
