from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(
    "data/remote_sensing/"
    "rs_vqa_visual_binary_balanced"
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
    # Global statistics
    # ------------------------------------------------------------

    channel_means = [
        float(
            rgb[:, :, i].mean()
        )
        for i in range(3)
    ]

    channel_stds = [
        float(
            rgb[:, :, i].std()
        )
        for i in range(3)
    ]

    mean = float(
        rgb.mean()
    )

    std = float(
        rgb.std()
    )

    # ------------------------------------------------------------
    # Gradient
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
    # Laplacian
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
            np.log2(
                probabilities
            )
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

    safe_max = np.maximum(
        max_rgb,
        1e-6,
    )

    saturation = (
        max_rgb - min_rgb
    ) / safe_max

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
    # Occupancy
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

    return [
        mean,
        std,
        channel_means[0],
        channel_means[1],
        channel_means[2],
        channel_stds[0],
        channel_stds[1],
        channel_stds[2],
        gradient_mean,
        gradient_std,
        edge_density,
        strong_edge_density,
        laplacian_variance,
        entropy,
        saturation_mean,
        saturation_std,
        local_contrast,
        occupancy,
    ]


def build_dataset(records):

    X = []
    y = []

    for record in records:

        X.append(
            extract_features(
                record["image"]
            )
        )

        y.append(
            1
            if record["answer"] == "YES"
            else 0
        )

    return (
        np.asarray(
            X,
            dtype=np.float64,
        ),
        np.asarray(
            y,
            dtype=np.int64,
        ),
    )


def confusion_matrix(y_true, y_pred):

    tp = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 1)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            &
            (y_pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            &
            (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 0)
        )
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def main():

    print("=" * 60)
    print(
        "PHASE 9.4H.9C.7B - "
        "LOW-LEVEL FEATURE CLASSIFIER"
    )
    print("=" * 60)

    train_records = load_jsonl(
        TRAIN
    )

    val_records = load_jsonl(
        VAL
    )

    assert len(
        train_records
    ) == 268

    assert len(
        val_records
    ) == 78

    X_train, y_train = build_dataset(
        train_records
    )

    X_val, y_val = build_dataset(
        val_records
    )

    print()
    print(
        "Training shape:",
        X_train.shape,
    )

    print(
        "Validation shape:",
        X_val.shape,
    )

    print(
        "Training YES:",
        int(
            np.sum(y_train == 1)
        ),
    )

    print(
        "Training NO:",
        int(
            np.sum(y_train == 0)
        ),
    )

    print(
        "Validation YES:",
        int(
            np.sum(y_val == 1)
        ),
    )

    print(
        "Validation NO:",
        int(
            np.sum(y_val == 0)
        ),
    )

    # ------------------------------------------------------------
    # Standardized logistic regression.
    # ------------------------------------------------------------

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=20260913,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    train_pred = model.predict(
        X_train
    )

    val_pred = model.predict(
        X_val
    )

    train_accuracy = float(
        np.mean(
            train_pred == y_train
        )
    )

    val_accuracy = float(
        np.mean(
            val_pred == y_val
        )
    )

    train_cm = confusion_matrix(
        y_train,
        train_pred,
    )

    val_cm = confusion_matrix(
        y_val,
        val_pred,
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print()
    print(
        "Training accuracy:",
        train_accuracy,
    )

    print(
        "Validation accuracy:",
        val_accuracy,
    )

    print()
    print("TRAIN CONFUSION MATRIX")
    print(
        "  TP:",
        train_cm["TP"],
    )

    print(
        "  TN:",
        train_cm["TN"],
    )

    print(
        "  FP:",
        train_cm["FP"],
    )

    print(
        "  FN:",
        train_cm["FN"],
    )

    print()
    print("VALIDATION CONFUSION MATRIX")
    print(
        "  TP:",
        val_cm["TP"],
    )

    print(
        "  TN:",
        val_cm["TN"],
    )

    print(
        "  FP:",
        val_cm["FP"],
    )

    print(
        "  FN:",
        val_cm["FN"],
    )

    # ------------------------------------------------------------
    # Per-class validation recall.
    # ------------------------------------------------------------

    yes_total = int(
        np.sum(y_val == 1)
    )

    no_total = int(
        np.sum(y_val == 0)
    )

    yes_recall = (
        val_cm["TP"]
        /
        yes_total
    )

    no_recall = (
        val_cm["TN"]
        /
        no_total
    )

    balanced_accuracy = (
        yes_recall
        +
        no_recall
    ) / 2.0

    print()
    print(
        "Validation YES recall:",
        yes_recall,
    )

    print(
        "Validation NO recall:",
        no_recall,
    )

    print(
        "Validation balanced accuracy:",
        balanced_accuracy,
    )

    # ------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------

    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if val_accuracy >= 0.70:

        print(
            "STRONG LOW-LEVEL "
            "GENERALIZATION"
        )

        print(
            "Simple image statistics "
            "predict the class on held-out "
            "validation data."
        )

    elif val_accuracy >= 0.60:

        print(
            "MODERATE LOW-LEVEL "
            "GENERALIZATION"
        )

        print(
            "Low-level image statistics "
            "carry substantial held-out "
            "class information."
        )

    else:

        print(
            "WEAK LOW-LEVEL "
            "GENERALIZATION"
        )

        print(
            "Simple statistics alone "
            "do not strongly generalize."
        )

    print()
    print(
        "This test does NOT prove that "
        "the dataset is invalid."
    )

    print(
        "It measures how much class "
        "information is available from "
        "simple low-level image statistics."
    )

    print()
    print(
        "PHASE 9.4H.9C.7B COMPLETE"
    )


if __name__ == "__main__":
    main()
