from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


BASE_MODEL = (
    "/home/lenovo/.cache/huggingface/hub/"
    "models--Qwen--Qwen2-VL-2B-Instruct/"
    "snapshots/895c3a49bc3fa70a340399125c650a463535e71c"
)

ADAPTER = Path(
    "outputs/checkpoints/"
    "qwen2vl_rs_vqa_visual_binary_dev"
)

DATA_ROOT = Path(
    "data/remote_sensing/rs_vqa_visual_binary_balanced"
)

VAL_JSONL = DATA_ROOT / "val.jsonl"

OUTPUT = Path(
    "outputs/checkpoints/"
    "qwen2vl_rs_vqa_visual_binary_dev/"
    "visual_acceptance_results.json"
)

SEED = 20260913

# Evaluate the complete held-out validation set.
# Counterfactual tests are performed on the first 20 records
# to keep the experiment practical on a 6 GB GPU.
COUNTERFACTUAL_LIMIT = 20


def load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_prediction(text: str):
    text = text.strip().upper()

    # Extract an isolated YES or NO.
    matches = re.findall(r"\b(YES|NO)\b", text)

    if matches:
        return matches[0]

    return "UNKNOWN"


def predict(model, processor, image, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": question
                    + "\nAnswer only YES or NO.",
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    device = next(model.parameters()).device

    inputs = {
        key: value.to(device)
        if torch.is_tensor(value)
        else value
        for key, value in inputs.items()
    }

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = generated[:, input_length:]

    text = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0]

    return text.strip(), normalize_prediction(text)


def blank_image(image):
    return Image.new(
        "RGB",
        image.size,
        (255, 255, 255),
    )


def black_image(image):
    return Image.new(
        "RGB",
        image.size,
        (0, 0, 0),
    )


def noise_image(image, rng):
    import numpy as np

    array = np.asarray(image).copy()

    noise = np.array(
        [
            [
                [
                    rng.randrange(256)
                    for _ in range(3)
                ]
                for _ in range(array.shape[1])
            ]
            for _ in range(array.shape[0])
        ],
        dtype=np.uint8,
    )

    return Image.fromarray(noise, mode="RGB")


def accuracy(records):
    valid = [
        r for r in records
        if r["prediction"] in {"YES", "NO"}
    ]

    if not valid:
        return 0.0

    correct = sum(
        r["prediction"] == r["answer"]
        for r in valid
    )

    return correct / len(valid)


def confusion(records):
    result = {
        "YES->YES": 0,
        "YES->NO": 0,
        "NO->YES": 0,
        "NO->NO": 0,
        "UNKNOWN": 0,
    }

    for r in records:
        target = r["answer"]
        prediction = r["prediction"]

        if prediction == "UNKNOWN":
            result["UNKNOWN"] += 1
        elif target == "YES" and prediction == "YES":
            result["YES->YES"] += 1
        elif target == "YES" and prediction == "NO":
            result["YES->NO"] += 1
        elif target == "NO" and prediction == "YES":
            result["NO->YES"] += 1
        elif target == "NO" and prediction == "NO":
            result["NO->NO"] += 1

    return result


def main():
    random.seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print("=" * 60)
    print("PHASE 9.4H.9C.4 - VISUAL ACCEPTANCE TEST")
    print("=" * 60)

    print("GPU:", torch.cuda.get_device_name(0))

    records = load_jsonl(VAL_JSONL)

    print("Validation records:", len(records))

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    )

    print("Loading base Qwen2-VL-2B...")

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER,
        local_files_only=True,
    )

    model.eval()

    print("Adapter loaded: PASS")

    # ------------------------------------------------------------
    # 1. COMPLETE HELD-OUT VALIDATION
    # ------------------------------------------------------------

    print()
    print("-" * 60)
    print("1. HELD-OUT VALIDATION")
    print("-" * 60)

    heldout = []

    for index, record in enumerate(records):
        image = Image.open(record["image"]).convert("RGB")

        raw, prediction = predict(
            model,
            processor,
            image,
            record["question"],
        )

        result = {
            "id": record["id"],
            "answer": record["answer"],
            "prediction": prediction,
            "raw_prediction": raw,
        }

        heldout.append(result)

        print(
            f"[{index + 1:02d}/{len(records)}] "
            f"target={record['answer']} "
            f"pred={prediction} "
            f"raw={raw!r}"
        )

    heldout_accuracy = accuracy(heldout)
    heldout_confusion = confusion(heldout)

    print()
    print("Held-out accuracy:", heldout_accuracy)

    print("Confusion matrix:")
    for key, value in heldout_confusion.items():
        print(f"  {key}: {value}")

    # ------------------------------------------------------------
    # 2. COUNTERFACTUAL IMAGE SENSITIVITY
    # ------------------------------------------------------------

    print()
    print("-" * 60)
    print("2. IMAGE SENSITIVITY / COUNTERFACTUAL TEST")
    print("-" * 60)

    sensitivity_records = []

    selected = records[:COUNTERFACTUAL_LIMIT]

    for index, record in enumerate(selected):
        image = Image.open(record["image"]).convert("RGB")

        real_raw, real_pred = predict(
            model,
            processor,
            image,
            record["question"],
        )

        white_raw, white_pred = predict(
            model,
            processor,
            blank_image(image),
            record["question"],
        )

        black_raw, black_pred = predict(
            model,
            processor,
            black_image(image),
            record["question"],
        )

        noise_raw, noise_pred = predict(
            model,
            processor,
            noise_image(
                image,
                random.Random(SEED + index),
            ),
            record["question"],
        )

        changed_white = real_pred != white_pred
        changed_black = real_pred != black_pred
        changed_noise = real_pred != noise_pred

        result = {
            "id": record["id"],
            "target": record["answer"],
            "real": {
                "prediction": real_pred,
                "raw": real_raw,
            },
            "white": {
                "prediction": white_pred,
                "raw": white_raw,
            },
            "black": {
                "prediction": black_pred,
                "raw": black_raw,
            },
            "noise": {
                "prediction": noise_pred,
                "raw": noise_raw,
            },
            "changed_real_to_white": changed_white,
            "changed_real_to_black": changed_black,
            "changed_real_to_noise": changed_noise,
        }

        sensitivity_records.append(result)

        print(
            f"[{index + 1:02d}/{len(selected)}] "
            f"target={record['answer']} "
            f"real={real_pred} "
            f"white={white_pred} "
            f"black={black_pred} "
            f"noise={noise_pred}"
        )

    changed_white_count = sum(
        r["changed_real_to_white"]
        for r in sensitivity_records
    )

    changed_black_count = sum(
        r["changed_real_to_black"]
        for r in sensitivity_records
    )

    changed_noise_count = sum(
        r["changed_real_to_noise"]
        for r in sensitivity_records
    )

    sensitivity_n = len(sensitivity_records)

    white_rate = changed_white_count / sensitivity_n
    black_rate = changed_black_count / sensitivity_n
    noise_rate = changed_noise_count / sensitivity_n

    print()
    print(
        "Real -> white changed:",
        changed_white_count,
        "/",
        sensitivity_n,
        f"({white_rate:.3f})",
    )

    print(
        "Real -> black changed:",
        changed_black_count,
        "/",
        sensitivity_n,
        f"({black_rate:.3f})",
    )

    print(
        "Real -> noise changed:",
        changed_noise_count,
        "/",
        sensitivity_n,
        f"({noise_rate:.3f})",
    )

    # ------------------------------------------------------------
    # 3. SIMPLE SCIENTIFIC GATE
    # ------------------------------------------------------------

    # This is deliberately NOT the final production gate.
    #
    # We require:
    #   - accuracy substantially above random
    #   - evidence of image dependence
    #
    # The exact threshold is conservative for this tiny dev set.
    #
    # Passing this gate means:
    #   "continue scientific evaluation"
    #
    # It does NOT mean:
    #   "production ready"

    accuracy_gate = heldout_accuracy >= 0.60

    image_dependence_gate = (
        white_rate >= 0.30
        or black_rate >= 0.30
        or noise_rate >= 0.30
    )

    if accuracy_gate and image_dependence_gate:
        preliminary_status = (
            "VISUAL_SIGNAL_DETECTED_CONTINUE_EVALUATION"
        )
    elif accuracy_gate:
        preliminary_status = (
            "ACCURACY_OK_BUT_IMAGE_DEPENDENCE_UNPROVEN"
        )
    else:
        preliminary_status = (
            "FAILED_PRELIMINARY_VISUAL_ACCEPTANCE"
        )

    results = {
        "phase": "9.4H.9C.4",
        "status": preliminary_status,
        "base_model": "Qwen2-VL-2B-Instruct",
        "adapter": str(ADAPTER),
        "validation_examples": len(records),
        "counterfactual_examples": sensitivity_n,
        "heldout_accuracy": heldout_accuracy,
        "heldout_confusion": heldout_confusion,
        "counterfactual": {
            "real_to_white_changed": changed_white_count,
            "real_to_black_changed": changed_black_count,
            "real_to_noise_changed": changed_noise_count,
            "white_rate": white_rate,
            "black_rate": black_rate,
            "noise_rate": noise_rate,
        },
        "heldout_predictions": heldout,
        "sensitivity_records": sensitivity_records,
        "production_ready": False,
        "note": (
            "This is a preliminary visual acceptance test. "
            "Passing does not establish production RS-VLM "
            "validity. Further held-out, task-diverse and "
            "remote-sensing acceptance tests are required."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("PHASE 9.4H.9C.4 SUMMARY")
    print("=" * 60)

    print("Held-out accuracy:", heldout_accuracy)

    print(
        "Real -> white change rate:",
        white_rate,
    )

    print(
        "Real -> black change rate:",
        black_rate,
    )

    print(
        "Real -> noise change rate:",
        noise_rate,
    )

    print("Preliminary status:", preliminary_status)

    print()
    print("Production registration: FALSE")
    print("Production acceptance: FALSE")

    print()
    print("Results saved:", OUTPUT)

    del model
    del base_model
    del processor

    import gc

    gc.collect()
    torch.cuda.empty_cache()

    print(
        "GPU after cleanup:",
        torch.cuda.memory_allocated()
        / (1024 ** 3),
        "GB allocated",
    )

    print()
    print("PHASE 9.4H.9C.4 COMPLETE")


if __name__ == "__main__":
    main()
