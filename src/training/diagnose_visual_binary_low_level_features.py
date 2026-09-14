from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


ROOT = Path(
    "data/remote_sensing/"
    "rs_vqa_visual_binary_balanced"
)

TRAIN = ROOT / "train.jsonl"


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def extract_features(path):

    image = Image.open(
        path
    ).convert("RGB")

    rgb = np.asarray(
        image
    ).astype(np.float32) / 255.0

    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    # ------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------

    mean = float(
        rgb.mean()
    )

    std = float(
        rgb.std()
    )

    channel_means = [
        float(rgb[:, :, i].mean())
        for i in range(3)
    ]

    channel_stds = [
        float(rgb[:, :, i].std())
        for i in range(3)
    ]

    # ------------------------------------------------------------
    # Gradient magnitude
    # ------------------------------------------------------------

    gx = np.diff(
        gray,
        axis=1,
    )

    gy = np.diff(
        gray,
        axis=0,
    )

    gradient = np.sqrt(
        gx[:-1, :] ** 2
        +
        gy[:, :-1] ** 2
    )

    gradient_mean = float(
        gradient.mean()
    )

    gradient_std = float(
        gradient.std()
    )

    edge_density = float(
        np.mean(
            gradient > 0.08
        )
    )

    strong_edge_density = float(
        np.mean(
            gradient > 0.15
        )
    )

    # ------------------------------------------------------------
    # Laplacian-like second derivative
    # ------------------------------------------------------------

    dxx = (
        gray[:, 2:]
        -
        2.0 * gray[:, 1:-1]
        +
        gray[:, :-2]
    )

    dyy = (
        gray[2:, :]
        -
        2.0 * gray[1:-1, :]
        +
        gray[:-2, :]
    )

    laplacian = np.zeros_like(
        gray
    )

    laplacian[:, 1:-1] += dxx
    laplacian[1:-1, :] += dyy

    laplacian_variance = float(
        laplacian.var()
    )

    # ------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------

    gray_uint8 = (
        np.clip(
            gray * 255.0,
            0,
            255,
        )
        .astype(np.uint8)
    )

    histogram = np.bincount(
        gray_uint8.ravel(),
        minlength=256,
    ).astype(np.float64)

    probabilities = (
        histogram
        /
        histogram.sum()
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = float(
        -np.sum(
            probabilities
            *
            np.log2(probabilities)
        )
    )

    # ------------------------------------------------------------
    # Saturation
    # ------------------------------------------------------------

    max_rgb = rgb.max(
        axis=2
    )

    min_rgb = rgb.min(
        axis=2
    )

    saturation = np.where(
        max_rgb > 1e-6,
        (max_rgb - min_rgb)
        / max_rgb,
        0.0,
    )

    saturation_mean = float(
        saturation.mean()
    )

    saturation_std = float(
        saturation.std()
    )

    # ------------------------------------------------------------
    # Local contrast
    # ------------------------------------------------------------

    image_gray = Image.fromarray(
        gray_uint8,
        mode="L",
    )

    blurred = image_gray.filter(
        ImageFilter.GaussianBlur(
            radius=2.0
        )
    )

    blurred_arr = (
        np.asarray(
            blurred
        ).astype(np.float32)
        / 255.0
    )

    local_contrast = float(
        np.abs(
            gray - blurred_arr
        ).mean()
    )

    # ------------------------------------------------------------
    # Spatial occupancy
    #
    # Count pixels substantially different from
    # the image's median intensity.
    # ------------------------------------------------------------

    median = float(
        np.median(gray)
    )

    occupancy = float(
        np.mean(
            np.abs(
                gray - median
            ) > 0.10
        )
    )

    return {
        "mean": mean,
        "std": std,
        "channel_mean_r": channel_means[0],
        "channel_mean_g": channel_means[1],
        "channel_mean_b": channel_means[2],
        "channel_std_r": channel_stds[0],
        "channel_std_g": channel_stds[1],
        "channel_std_b": channel_stds[2],
        "gradient_mean": gradient_mean,
        "gradient_std": gradient_std,
        "edge_density": edge_density,
        "strong_edge_density": strong_edge_density,
        "laplacian_variance": laplacian_variance,
        "entropy": entropy,
        "saturation_mean": saturation_mean,
        "saturation_std": saturation_std,
        "local_contrast": local_contrast,
        "occupancy": occupancy,
    }


def main():

    print("=" * 60)
    print(
        "PHASE 9.4H.9C.7A - "
        "LOW-LEVEL VISUAL FEATURE DIAGNOSTIC"
    )
    print("=" * 60)

    records = load_jsonl(
        TRAIN
    )

    assert len(records) == 268

    rows = []

    for record in records:

        features = extract_features(
            record["image"]
        )

        rows.append(
            {
                "answer": record["answer"],
                **features,
            }
        )

    feature_names = [
        key
        for key in rows[0]
        if key != "answer"
    ]

    print()
    print(
        "Records:",
        len(rows),
    )

    print(
        "Features:",
        len(feature_names),
    )

    print()
    print("=" * 60)
    print("CLASS-WISE FEATURE MEANS")
    print("=" * 60)

    results = {}

    for feature in feature_names:

        yes_values = np.array(
            [
                row[feature]
                for row in rows
                if row["answer"] == "YES"
            ],
            dtype=np.float64,
        )

        no_values = np.array(
            [
                row[feature]
                for row in rows
                if row["answer"] == "NO"
            ],
            dtype=np.float64,
        )

        yes_mean = float(
            yes_values.mean()
        )

        no_mean = float(
            no_values.mean()
        )

        pooled_std = float(
            np.sqrt(
                (
                    yes_values.var()
                    +
                    no_values.var()
                )
                / 2.0
            )
        )

        if pooled_std > 1e-12:

            effect_size = (
                abs(
                    yes_mean
                    -
                    no_mean
                )
                /
                pooled_std
            )

        else:

            effect_size = 0.0

        results[feature] = {
            "yes_mean": yes_mean,
            "no_mean": no_mean,
            "absolute_gap": abs(
                yes_mean
                -
                no_mean
            ),
            "effect_size": float(
                effect_size
            ),
        }

    ranked = sorted(
        results.items(),
        key=lambda item:
            item[1]["effect_size"],
        reverse=True,
    )

    print()

    for feature, stats in ranked:

        print(
            f"{feature:24s} "
            f"YES={stats['yes_mean']:.6f} "
            f"NO={stats['no_mean']:.6f} "
            f"gap={stats['absolute_gap']:.6f} "
            f"effect={stats['effect_size']:.4f}"
        )

    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    print()
    print(
        "Large effect sizes indicate that "
        "simple image statistics differ "
        "between YES and NO."
    )

    print(
        "This does NOT automatically mean "
        "the dataset is invalid."
    )

    print(
        "It tells us whether low-level "
        "signals can potentially support "
        "the classification."
    )

    print()
    print(
        "PHASE 9.4H.9C.7A COMPLETE"
    )


if __name__ == "__main__":
    main()
