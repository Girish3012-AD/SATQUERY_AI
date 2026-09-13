from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260913

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = PROJECT_ROOT / "data" / "remote_sensing" / "rs_vqa"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "remote_sensing" / "rs_vqa_balanced"

TRAIN_SOURCE = SOURCE_ROOT / "train.jsonl"
VAL_SOURCE = SOURCE_ROOT / "val.jsonl"

TRAIN_OUTPUT = OUTPUT_ROOT / "train.jsonl"
VAL_OUTPUT = OUTPUT_ROOT / "val.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def patch_id(record: dict) -> str:
    return str(record["id"]).rsplit("_", 1)[0]


def coverage(record: dict) -> float:
    gt = record["ground_truth"]

    if not isinstance(gt, dict):
        raise ValueError("ground_truth must be a dictionary")

    value = gt.get("building_fraction")

    if value is None:
        raise ValueError(
            f"Missing building_fraction in {record['id']}"
        )

    return float(value)


def density_class(fraction: float) -> str:
    if fraction == 0:
        return "No buildings"
    if fraction <= 0.01:
        return "Very low building density"
    if fraction <= 0.05:
        return "Low building density"
    if fraction <= 0.15:
        return "Medium building density"
    return "High building density"


def scene_class(fraction: float) -> str:
    if fraction == 0:
        return "Non-built-up"
    if fraction <= 0.05:
        return "Mostly non-built-up"
    if fraction <= 0.15:
        return "Mixed built-up"
    return "Mostly built-up"


def coverage_class(fraction: float) -> str:
    if fraction == 0:
        return "No building coverage"
    if fraction <= 0.05:
        return "Low building coverage"
    if fraction <= 0.15:
        return "Moderate building coverage"
    if fraction <= 0.30:
        return "High building coverage"
    return "Very high building coverage"


def make_questions(record: dict) -> list[tuple[str, str, str]]:
    fraction = coverage(record)

    presence = (
        "Buildings are present."
        if fraction > 0
        else
        "No buildings are present."
    )

    density = density_class(fraction)
    scene = scene_class(fraction)
    coverage_label = coverage_class(fraction)

    return [
        (
            "building_presence",
            "Are buildings visible in this image?",
            presence,
        ),
        (
            "building_density",
            "How would you classify the building density?",
            density,
        ),
        (
            "building_coverage",
            "How would you classify the amount of image covered by buildings?",
            coverage_label,
        ),
        (
            "scene_type",
            "How would you classify the overall scene?",
            scene,
        ),
    ]


def rebuild_record(
    source_record: dict,
    task: str,
    question: str,
    answer: str,
) -> dict:
    fraction = coverage(source_record)

    return {
        "id": f"{patch_id(source_record)}_{task}",
        "image": source_record["image"],
        "question": question,
        "answer": answer,
        "task": task,
        "source": source_record["source"],
        "split": source_record["split"],
        "ground_truth": {
            "building_pixels": source_record["ground_truth"]["building_pixels"],
            "building_fraction": fraction,
            "density_class": density_class(fraction),
            "scene_class": scene_class(fraction),
            "coverage_class": coverage_class(fraction),
        },
        "metadata": {
            "development_only": True,
            "target_design": "balanced_categorical_rs_vqa_v2",
            "exact_coverage_retained_for_gis": True,
        },
    }


def choose_balanced_train_patches(
    records: list[dict],
) -> list[dict]:
    patches = {}

    for record in records:
        pid = patch_id(record)

        if pid in patches:
            continue

        patches[pid] = record

    groups = defaultdict(list)

    for record in patches.values():
        fraction = coverage(record)
        groups[density_class(fraction)].append(record)

    print("\nAvailable training patches by density:")
    for key in (
        "No buildings",
        "Very low building density",
        "Low building density",
        "Medium building density",
        "High building density",
    ):
        print(f"  {key}: {len(groups[key])}")

    # We retain every useful positive patch and balance the
    # dominant empty class down to the number of positive patches.
    positive = [
        record
        for record in patches.values()
        if coverage(record) > 0
    ]

    negative = [
        record
        for record in patches.values()
        if coverage(record) == 0
    ]

    rng = random.Random(SEED)

    target_negative = min(
        len(negative),
        len(positive),
    )

    selected_negative = rng.sample(
        negative,
        target_negative,
    )

    selected = positive + selected_negative

    rng.shuffle(selected)

    print("\nBalanced patch selection:")
    print("  Positive patches:", len(positive))
    print("  Selected negative patches:", len(selected_negative))
    print("  Total training patches:", len(selected))

    return selected


def build_split(
    source_records: list[dict],
    selected_patches: set[str] | None = None,
) -> list[dict]:
    patch_records = {}

    for record in source_records:
        pid = patch_id(record)

        if selected_patches is not None and pid not in selected_patches:
            continue

        if pid not in patch_records:
            patch_records[pid] = record

    output = []

    for source_record in patch_records.values():
        for task, question, answer in make_questions(source_record):
            output.append(
                rebuild_record(
                    source_record,
                    task,
                    question,
                    answer,
                )
            )

    return output


def validate_split(records: list[dict], name: str) -> None:
    print(f"\n{name} validation:")
    print("  examples:", len(records))

    required = {
        "id",
        "image",
        "question",
        "answer",
        "task",
        "source",
        "split",
        "ground_truth",
    }

    for record in records:
        missing = required - set(record)

        if missing:
            raise AssertionError(
                f"{name}: missing fields {missing}"
            )

    ids = [record["id"] for record in records]

    if len(ids) != len(set(ids)):
        raise AssertionError(
            f"{name}: duplicate IDs detected"
        )

    images = {record["image"] for record in records}

    print("  unique images:", len(images))

    task_counts = Counter(
        record["task"]
        for record in records
    )

    print("  task counts:", dict(task_counts))

    for task in (
        "building_presence",
        "building_density",
        "building_coverage",
        "scene_type",
    ):
        if task_counts[task] == 0:
            raise AssertionError(
                f"{name}: missing task {task}"
            )


def print_answer_distribution(records: list[dict], name: str) -> None:
    print(f"\n{name} answer distribution:")

    by_task = defaultdict(list)

    for record in records:
        by_task[record["task"]].append(record["answer"])

    for task in sorted(by_task):
        print(f"\n  [{task}]")

        counts = Counter(by_task[task])

        for answer, count in counts.most_common():
            print(f"    {count:4d} | {answer}")


def main() -> None:
    if not TRAIN_SOURCE.exists():
        raise FileNotFoundError(TRAIN_SOURCE)

    if not VAL_SOURCE.exists():
        raise FileNotFoundError(VAL_SOURCE)

    train_source = load_jsonl(TRAIN_SOURCE)
    val_source = load_jsonl(VAL_SOURCE)

    print("=" * 70)
    print("PHASE 9.4H.2 - BALANCED RS-VQA DATASET")
    print("=" * 70)

    print("\nSource:")
    print("  Train examples:", len(train_source))
    print("  Val examples:", len(val_source))

    selected_train = choose_balanced_train_patches(
        train_source
    )

    selected_ids = {
        patch_id(record)
        for record in selected_train
    }

    train_records = build_split(
        train_source,
        selected_ids,
    )

    # Validation is deliberately rebuilt only in target format.
    # The original validation patch membership is preserved.
    val_records = build_split(
        val_source,
        None,
    )

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRAIN_OUTPUT.write_text(
        "\n".join(
            json.dumps(record, sort_keys=True)
            for record in train_records
        )
        + "\n",
        encoding="utf-8",
    )

    VAL_OUTPUT.write_text(
        "\n".join(
            json.dumps(record, sort_keys=True)
            for record in val_records
        )
        + "\n",
        encoding="utf-8",
    )

    validate_split(train_records, "TRAIN")
    validate_split(val_records, "VAL")

    print_answer_distribution(
        train_records,
        "TRAIN",
    )

    print_answer_distribution(
        val_records,
        "VAL",
    )

    train_images = {
        record["image"]
        for record in train_records
    }

    val_images = {
        record["image"]
        for record in val_records
    }

    overlap = train_images & val_images

    print("\nCross-split image overlap:", len(overlap))

    if overlap:
        raise AssertionError(
            "Image leakage detected between train and val."
        )

    print("IMAGE SPLIT: PASS")

    train_patches = {
        patch_id(record)
        for record in train_records
    }

    val_patches = {
        patch_id(record)
        for record in val_records
    }

    patch_overlap = train_patches & val_patches

    print("Cross-split patch overlap:", len(patch_overlap))

    if patch_overlap:
        raise AssertionError(
            "Patch leakage detected between train and val."
        )

    print("PATCH SPLIT: PASS")

    print("\nOutput:")
    print(" ", TRAIN_OUTPUT)
    print(" ", VAL_OUTPUT)

    print("\n" + "=" * 70)
    print("PHASE 9.4H.2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
