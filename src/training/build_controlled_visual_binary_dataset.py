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
    "data/remote_sensing/rs_vqa_visual_binary_controlled"
)

TRAIN_INPUT = INPUT_ROOT / "train.jsonl"
VAL_INPUT = INPUT_ROOT / "val.jsonl"

TRAIN_OUTPUT = OUTPUT_ROOT / "train.jsonl"
VAL_OUTPUT = OUTPUT_ROOT / "val.jsonl"

SEED = 20260913

# Maximum acceptable class-level differences.
MAX_MEAN_GAP = 4.0
MAX_STD_GAP = 4.0

# Maximum average paired distance.
MAX_PAIR_DISTANCE = 10.0


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


def pair_distance(a, b):
    return abs(
        a["mean"] - b["mean"]
    ) + abs(
        a["std"] - b["std"]
    )


def build_pairs(yes_images, no_images):
    yes_features = {
        image: image_features(image)
        for image in yes_images
    }

    no_features = {
        image: image_features(image)
        for image in no_images
    }

    # ------------------------------------------------------------
    # Build coarse bins.
    #
    # 8-pixel brightness bins and 8-pixel std bins.
    # Matching inside nearby bins prevents the algorithm from
    # simply selecting arbitrary NO examples.
    # ------------------------------------------------------------

    def bin_key(features):
        return (
            int(features["mean"] // 8),
            int(features["std"] // 8),
        )

    no_bins = {}

    for image, features in no_features.items():
        no_bins.setdefault(
            bin_key(features),
            set(),
        ).add(image)

    remaining_no = set(no_images)
    pairs = []

    # Hard examples first.
    ordered_yes = sorted(
        yes_images,
        key=lambda image: (
            yes_features[image]["mean"],
            yes_features[image]["std"],
        ),
    )

    for yes_image in ordered_yes:
        if not remaining_no:
            break

        yf = yes_features[yes_image]

        ybin = bin_key(yf)

        candidates = []

        # Search same bin first, then neighbouring bins.
        for radius in range(0, 4):
            for mean_offset in range(
                -radius,
                radius + 1,
            ):
                for std_offset in range(
                    -radius,
                    radius + 1,
                ):
                    key = (
                        ybin[0] + mean_offset,
                        ybin[1] + std_offset,
                    )

                    candidates.extend(
                        no_bins.get(key, set())
                        & remaining_no
                    )

            if candidates:
                break

        if not candidates:
            candidates = list(
                remaining_no
            )

        best_no = min(
            candidates,
            key=lambda image: pair_distance(
                yf,
                no_features[image],
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


def collect_stats(records):
    values = {
        "YES": [],
        "NO": [],
    }

    for record in records:
        values[
            record["answer"]
        ].append(
            image_features(record["image"])
        )

    result = {}

    for label in ["YES", "NO"]:
        means = [
            x["mean"]
            for x in values[label]
        ]

        stds = [
            x["std"]
            for x in values[label]
        ]

        result[label] = {
            "count": len(means),
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

    return result


def main():
    random.seed(SEED)

    print("=" * 60)
    print(
        "PHASE 9.4H.9C.5C - "
        "CONTROLLED VISUAL DATASET"
    )
    print("=" * 60)

    train = load_jsonl(TRAIN_INPUT)
    val = load_jsonl(VAL_INPUT)

    groups = group_by_image(train)

    yes_images = []
    no_images = []

    for image, records in groups.items():
        labels = {
            r["answer"]
            for r in records
        }

        if labels == {"YES"}:
            yes_images.append(image)

        elif labels == {"NO"}:
            no_images.append(image)

        else:
            raise ValueError(
                f"Mixed labels for image: {image}"
            )

    print(
        "Available YES images:",
        len(yes_images),
    )

    print(
        "Available NO images:",
        len(no_images),
    )

    pair_count = min(
        len(yes_images),
        len(no_images),
    )

    if len(yes_images) > pair_count:
        yes_images = yes_images[:pair_count]

    random.shuffle(no_images)

    pairs = build_pairs(
        yes_images,
        no_images,
    )

    if len(pairs) != pair_count:
        raise RuntimeError(
            "Could not construct complete matching."
        )

    selected_images = {
        image
        for pair in pairs
        for image in pair
    }

    controlled_train = [
        record
        for record in train
        if record["image"] in selected_images
    ]

    random.shuffle(controlled_train)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRAIN_OUTPUT.write_text(
        "\n".join(
            json.dumps(record)
            for record in controlled_train
        )
        + "\n",
        encoding="utf-8",
    )

    # Validation remains untouched.
    VAL_OUTPUT.write_text(
        "\n".join(
            json.dumps(record)
            for record in val
        )
        + "\n",
        encoding="utf-8",
    )

    stats = collect_stats(
        controlled_train
    )

    yes_mean = stats["YES"]["mean_pixel"]
    no_mean = stats["NO"]["mean_pixel"]

    yes_std = stats["YES"]["mean_image_std"]
    no_std = stats["NO"]["mean_image_std"]

    mean_gap = abs(
        yes_mean - no_mean
    )

    std_gap = abs(
        yes_std - no_std
    )

    pair_distances = []

    for yes_image, no_image in pairs:
        yf = image_features(yes_image)
        nf = image_features(no_image)

        pair_distances.append(
            pair_distance(yf, nf)
        )

    mean_pair_distance = float(
        np.mean(pair_distances)
    )

    median_pair_distance = float(
        np.median(pair_distances)
    )

    maximum_pair_distance = float(
        np.max(pair_distances)
    )

    print()
    print("CONTROLLED TRAINING DATA")
    print("-" * 60)

    print(
        "Examples:",
        len(controlled_train),
    )

    print(
        "YES:",
        sum(
            r["answer"] == "YES"
            for r in controlled_train
        ),
    )

    print(
        "NO :",
        sum(
            r["answer"] == "NO"
            for r in controlled_train
        ),
    )

    print()
    print("CLASS STATISTICS")
    print("-" * 60)

    print(
        "YES mean pixel:",
        yes_mean,
    )

    print(
        "NO mean pixel :",
        no_mean,
    )

    print(
        "Mean brightness gap:",
        mean_gap,
    )

    print(
        "YES mean image std:",
        yes_std,
    )

    print(
        "NO mean image std :",
        no_std,
    )

    print(
        "Image-std gap:",
        std_gap,
    )

    print()
    print("PAIR QUALITY")
    print("-" * 60)

    print(
        "Matched pairs:",
        len(pairs),
    )

    print(
        "Mean pair distance:",
        mean_pair_distance,
    )

    print(
        "Median pair distance:",
        median_pair_distance,
    )

    print(
        "Maximum pair distance:",
        maximum_pair_distance,
    )

    print()
    print("CONTROL CRITERIA")
    print("-" * 60)

    mean_pass = (
        mean_gap <= MAX_MEAN_GAP
    )

    std_pass = (
        std_gap <= MAX_STD_GAP
    )

    pair_pass = (
        mean_pair_distance
        <= MAX_PAIR_DISTANCE
    )

    print(
        "Mean brightness gap <= 4:",
        mean_pass,
    )

    print(
        "Image std gap <= 4:",
        std_pass,
    )

    print(
        "Mean pair distance <= 10:",
        pair_pass,
    )

    if not (
        mean_pass
        and std_pass
        and pair_pass
    ):
        print()
        print(
            "CONTROLLED MATCHING: FAIL"
        )
        print(
            "The resulting dataset still "
            "contains excessive low-level "
            "visual differences."
        )
        print()
        print(
            "Do NOT train Qwen on this dataset."
        )
        raise SystemExit(2)

    print()
    print(
        "CONTROLLED MATCHING: PASS"
    )

    print()
    print(
        "Validation examples:",
        len(val),
    )

    print(
        "Validation remains untouched."
    )

    print()
    print(
        "PHASE 9.4H.9C.5C COMPLETE"
    )


if __name__ == "__main__":
    main()
