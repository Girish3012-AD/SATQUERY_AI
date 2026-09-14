from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path("data/remote_sensing/rs_vqa_visual")
TRAIN_INPUT = ROOT / "train.jsonl"
VAL_INPUT = ROOT / "val.jsonl"

OUTPUT_ROOT = Path("data/remote_sensing/rs_vqa_visual_binary")
TRAIN_OUTPUT = OUTPUT_ROOT / "train.jsonl"
VAL_OUTPUT = OUTPUT_ROOT / "val.jsonl"

SEED = 20260913


def convert_record(record: dict) -> dict:
    task = record["task"]

    if task == "visual_building_presence":
        question = (
            "Are buildings visible in this remote-sensing image patch?"
        )
    elif task == "visual_built_environment":
        question = (
            "Does this remote-sensing image patch contain a built environment?"
        )
    else:
        raise ValueError(f"Unexpected visual task: {task}")

    original_answer = record["answer"].strip().lower()

    if original_answer.startswith("no"):
        answer = "NO"
        label = 0
    elif original_answer.startswith("yes") or original_answer.startswith("buildings"):
        answer = "YES"
        label = 1
    else:
        raise ValueError(
            f"Cannot normalize answer '{record['answer']}' "
            f"for task '{task}'"
        )

    updated = {
        "id": record["id"],
        "image": record["image"],
        "source_raster_patch": record.get(
            "source_raster_patch",
            record["image"],
        ),
        "question": question,
        "answer": answer,
        "task": task,
        "source": record["source"],
        "ground_truth": {
            **record["ground_truth"],
            "binary_label": label,
        },
        "image_representation": record.get(
            "image_representation",
            {
                "type": "rgb_png",
                "source_bands": [0, 1, 2],
                "source_band_count": 4,
            },
        ),
        "vlm_input_policy": {
            "image_only": True,
            "evidence_supplied": False,
            "ground_truth_supplied": False,
            "exact_measurements_supplied": False,
        },
        "target_design": {
            "type": "binary",
            "classes": ["NO", "YES"],
            "purpose": "controlled visual discrimination",
        },
        "original_answer": record["answer"],
    }

    return updated


def process(input_path: Path, output_path: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    updated = [convert_record(record) for record in records]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        "\n".join(json.dumps(record) for record in updated) + "\n",
        encoding="utf-8",
    )

    return updated


def main():
    random.seed(SEED)

    print("=" * 60)
    print("PHASE 9.4H.9C.1 - CONTROLLED BINARY VISUAL DATASET")
    print("=" * 60)

    train = process(TRAIN_INPUT, TRAIN_OUTPUT)
    val = process(VAL_INPUT, VAL_OUTPUT)

    print("Train examples:", len(train))
    print("Validation examples:", len(val))
    print("Total:", len(train) + len(val))

    train_images = {r["image"] for r in train}
    val_images = {r["image"] for r in val}

    print()
    print("Train unique images:", len(train_images))
    print("Validation unique images:", len(val_images))
    print("Train/validation overlap:", len(train_images & val_images))

    for name, records in [
        ("TRAIN", train),
        ("VALIDATION", val),
    ]:
        yes = sum(r["answer"] == "YES" for r in records)
        no = sum(r["answer"] == "NO" for r in records)

        print()
        print(name)
        print("  YES:", yes)
        print("  NO :", no)

    print()
    print("VLM evidence supplied: False")
    print("Ground truth supplied to VLM: False")
    print("Target type: YES/NO")
    print()
    print("PHASE 9.4H.9C.1 BUILD COMPLETE")


if __name__ == "__main__":
    main()
