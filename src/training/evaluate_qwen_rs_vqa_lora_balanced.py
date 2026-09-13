from __future__ import annotations

import gc
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration

from src.training.rs_vqa_lora_collator import (
    DEFAULT_QWEN_SNAPSHOT,
    RSVQACollator,
)
from src.training.rs_vqa_lora_data_balanced import (
    DEFAULT_DATASET_ROOT,
    RSVQADataset,
    build_qwen_prompt_messages,
)


ADAPTER_DIR = Path(
    "outputs/checkpoints/qwen2vl_rs_vqa_lora_balanced_dev"
)

MAX_EXAMPLES = 156
MAX_NEW_TOKENS = 32
MAX_LENGTH = 512


def normalize_answer(text: str) -> str:
    """
    Conservative normalization for development evaluation.

    This does not use semantic similarity or an LLM judge.
    """

    text = text.strip().lower()

    text = re.sub(
        r"[^a-z0-9.%\- ]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def exact_match(
    prediction: str,
    target: str,
) -> bool:
    return (
        normalize_answer(prediction)
        == normalize_answer(target)
    )


def load_base_model():
    return Qwen2VLForConditionalGeneration.from_pretrained(
        DEFAULT_QWEN_SNAPSHOT,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )


def load_adapted_model():
    base_model = load_base_model()

    return PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
    )


def model_device(model) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    raise RuntimeError(
        "Could not determine model device."
    )


def generate_answer(
    model,
    processor,
    example,
) -> tuple[str, float]:
    """
    Generate one answer using the Qwen multimodal processor.

    Returns:
        answer, latency_seconds
    """

    messages = build_qwen_prompt_messages(
        example
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    from PIL import Image

    image = Image.open(
        example.image
    ).convert("RGB")

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    device = model_device(model)

    model_inputs = {
        key: value.to(device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in inputs.items()
    }

    input_length = (
        model_inputs["input_ids"].shape[1]
    )

    start = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    latency = time.perf_counter() - start

    generated_tokens = generated[
        :,
        input_length:,
    ]

    answer = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    return answer, latency


def evaluate_model(
    model,
    processor,
    dataset,
    label: str,
):
    print()
    print(
        "=================================================="
    )
    print(f"EVALUATING: {label}")
    print(
        "=================================================="
    )

    model.eval()

    records = []

    total_latency = 0.0

    for index in range(
        min(MAX_EXAMPLES, len(dataset))
    ):
        example = dataset[index]

        prediction, latency = generate_answer(
            model,
            processor,
            example,
        )

        target = example.answer

        exact = exact_match(
            prediction,
            target,
        )

        total_latency += latency

        record = {
            "index": index,
            "id": example.example_id,
            "task": example.task,
            "question": example.question,
            "ground_truth": target,
            "prediction": prediction,
            "exact_match": exact,
            "normalized_prediction": normalize_answer(
                prediction
            ),
            "normalized_ground_truth": normalize_answer(
                target
            ),
            "latency_seconds": latency,
        }

        records.append(record)

        print(
            f"[{index + 1:02d}/{MAX_EXAMPLES}] "
            f"{example.task:<22} "
            f"EM={'PASS' if exact else 'FAIL'} "
            f"| latency={latency:.2f}s"
        )

        print(
            f"  GT:   {target}"
        )

        print(
            f"  Pred: {prediction}"
        )

    accuracy = (
        sum(
            record["exact_match"]
            for record in records
        )
        / len(records)
    )

    mean_latency = (
        total_latency
        / len(records)
    )

    per_task = defaultdict(list)

    for record in records:
        per_task[
            record["task"]
        ].append(
            record["exact_match"]
        )

    task_accuracy = {
        task: float(
            np.mean(values)
        )
        for task, values in per_task.items()
    }

    print()
    print(
        f"{label} exact-match accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"{label} mean latency: "
        f"{mean_latency:.3f}s"
    )

    print(
        f"{label} per-task accuracy:"
    )

    for task in sorted(task_accuracy):
        print(
            f"  {task}: "
            f"{task_accuracy[task]:.4f}"
        )

    return {
        "label": label,
        "examples": len(records),
        "accuracy": accuracy,
        "mean_latency_seconds": mean_latency,
        "per_task_accuracy": task_accuracy,
        "records": records,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Adapter not found: {ADAPTER_DIR}"
        )

    adapter_file = (
        ADAPTER_DIR
        / "adapter_model.safetensors"
    )

    adapter_config = (
        ADAPTER_DIR
        / "adapter_config.json"
    )

    if not adapter_file.exists():
        raise FileNotFoundError(
            adapter_file
        )

    if not adapter_config.exists():
        raise FileNotFoundError(
            adapter_config
        )

    print("GPU:")
    print(torch.cuda.get_device_name(0))

    print()
    print("Loading validation dataset...")

    dataset = RSVQADataset(
        DEFAULT_DATASET_ROOT / "val.jsonl"
    )

    print(
        "Validation examples available:",
        len(dataset),
    )

    print(
        "Development evaluation examples:",
        min(MAX_EXAMPLES, len(dataset)),
    )

    print()
    print("Loading processor...")

    collator = RSVQACollator(
        model_path=DEFAULT_QWEN_SNAPSHOT,
        max_length=MAX_LENGTH,
    )

    processor = collator.processor

    # ------------------------------------------------
    # BASE MODEL
    # ------------------------------------------------

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print()
    print(
        "=================================================="
    )
    print("LOADING BASE QWEN2-VL-2B")
    print(
        "=================================================="
    )

    base_model = load_base_model()

    base_result = evaluate_model(
        base_model,
        processor,
        dataset,
        "BASE",
    )

    base_peak_memory = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        f"Base peak GPU memory: "
        f"{base_peak_memory:.3f} GB"
    )

    del base_model

    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------
    # ADAPTED MODEL
    # ------------------------------------------------

    torch.cuda.reset_peak_memory_stats()

    print()
    print(
        "=================================================="
    )
    print("LOADING ADAPTED QWEN2-VL-2B + LORA")
    print(
        "=================================================="
    )

    adapted_model = load_adapted_model()

    print()
    print("Adapter loaded:", ADAPTER_DIR)

    trainable = [
        name
        for name, parameter
        in adapted_model.named_parameters()
        if parameter.requires_grad
    ]

    print(
        "Trainable tensors:",
        len(trainable),
    )

    if trainable:
        raise AssertionError(
            "Evaluation model unexpectedly has "
            "trainable parameters."
        )

    print(
        "EVALUATION MODEL FROZEN: PASS"
    )

    adapted_result = evaluate_model(
        adapted_model,
        processor,
        dataset,
        "ADAPTED",
    )

    adapted_peak_memory = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        f"Adapted peak GPU memory: "
        f"{adapted_peak_memory:.3f} GB"
    )

    # ------------------------------------------------
    # COMPARISON
    # ------------------------------------------------

    accuracy_delta = (
        adapted_result["accuracy"]
        - base_result["accuracy"]
    )

    print()
    print(
        "=================================================="
    )
    print("BASE VS ADAPTED COMPARISON")
    print(
        "=================================================="
    )

    print(
        f"Base accuracy:       "
        f"{base_result['accuracy']:.4f}"
    )

    print(
        f"Adapted accuracy:    "
        f"{adapted_result['accuracy']:.4f}"
    )

    print(
        f"Accuracy delta:      "
        f"{accuracy_delta:+.4f}"
    )

    print()
    print("Per-task comparison:")

    all_tasks = sorted(
        set(
            base_result["per_task_accuracy"]
        )
        | set(
            adapted_result["per_task_accuracy"]
        )
    )

    for task in all_tasks:
        base_acc = (
            base_result["per_task_accuracy"]
            .get(task, 0.0)
        )

        adapted_acc = (
            adapted_result[
                "per_task_accuracy"
            ]
            .get(task, 0.0)
        )

        print(
            f"  {task:<22} "
            f"base={base_acc:.4f} "
            f"adapted={adapted_acc:.4f} "
            f"delta={adapted_acc - base_acc:+.4f}"
        )

    # ------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------

    output_path = (
        ADAPTER_DIR
        / "base_vs_adapted_eval.json"
    )

    evaluation = {
        "phase": "9.4F",
        "status": "DEVELOPMENT_EVALUATION",
        "base_model": "Qwen2-VL-2B-Instruct",
        "adapter": str(ADAPTER_DIR),
        "dataset": "SpaceNet4 RS-VQA development dataset",
        "validation_examples_available": len(dataset),
        "evaluation_examples": min(
            MAX_EXAMPLES,
            len(dataset),
        ),
        "max_new_tokens": MAX_NEW_TOKENS,
        "base": {
            "accuracy": base_result["accuracy"],
            "mean_latency_seconds": (
                base_result[
                    "mean_latency_seconds"
                ]
            ),
            "per_task_accuracy": (
                base_result[
                    "per_task_accuracy"
                ]
            ),
            "peak_gpu_memory_gb": base_peak_memory,
            "records": base_result["records"],
        },
        "adapted": {
            "accuracy": adapted_result["accuracy"],
            "mean_latency_seconds": (
                adapted_result[
                    "mean_latency_seconds"
                ]
            ),
            "per_task_accuracy": (
                adapted_result[
                    "per_task_accuracy"
                ]
            ),
            "peak_gpu_memory_gb": (
                adapted_peak_memory
            ),
            "records": adapted_result["records"],
        },
        "accuracy_delta": accuracy_delta,
        "development_only": True,
        "scientific_warning": (
            "This validation set originates from the "
            "single-scene SpaceNet4 development dataset. "
            "This result is not an independent benchmark "
            "of remote-sensing generalization."
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            evaluation,
            handle,
            indent=2,
        )

    print()
    print(
        "Saved evaluation:"
    )
    print(output_path)

    del adapted_model
    del collator
    del processor
    del dataset

    gc.collect()
    torch.cuda.empty_cache()

    print()
    print(
        "=================================================="
    )
    print("FINAL GPU CLEANUP")
    print(
        "=================================================="
    )

    print(
        f"Allocated: "
        f"{torch.cuda.memory_allocated() / 1024**3:.3f} GB"
    )

    print(
        f"Reserved: "
        f"{torch.cuda.memory_reserved() / 1024**3:.3f} GB"
    )

    print()
    print(
        "=================================================="
    )
    print("PHASE 9.4F COMPLETE")
    print(
        "=================================================="
    )


if __name__ == "__main__":
    main()
