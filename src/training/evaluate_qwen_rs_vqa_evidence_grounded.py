from __future__ import annotations

import gc
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration

from src.training.rs_vqa_evidence_grounded_data import (
    load_validation_dataset,
)

from src.training.rs_vqa_evidence_grounded_collator import (
    EvidenceGroundedQwenCollator,
)


MODEL_PATH = (
    "/home/lenovo/.cache/huggingface/hub/"
    "models--Qwen--Qwen2-VL-2B-Instruct/"
    "snapshots/"
    "895c3a49bc3fa70a340399125c650a463535e71c"
)

ADAPTER_PATH = Path(
    "outputs/checkpoints/"
    "qwen2vl_rs_vqa_evidence_grounded_dev"
)

DATASET_ROOT = Path(
    "data/remote_sensing/"
    "rs_vqa_evidence_grounded"
)

OUTPUT_DIR = ADAPTER_PATH

MAX_NEW_TOKENS = 64
MAX_LENGTH = 512

PERTURBATION_LIMIT = 24
IMAGE_SENSITIVITY_LIMIT = 24

SEED = 20260913


def cleanup() -> None:

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def normalize_text(text: str) -> str:

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^\w.%\- ]",
        "",
        text,
    )

    return text


def extract_percentage(text: str):
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*%",
        text,
    )

    if match is None:
        return None

    return float(
        match.group(1)
    )


def extract_density(text: str):
    text = text.lower()

    # Specific multi-word categories must be checked before
    # shorter substrings such as "high" and "low".
    categories = [
        "very high building density",
        "very low building density",
        "high building density",
        "medium building density",
        "low building density",
        "no building density",
    ]

    for category in categories:
        if category in text:
            return category

    if "no buildings" in text:
        return "no building density"

    return None

def extract_scene(text: str):

    text = text.lower()

    categories = [
        "mostly built-up",
        "mixed built-up",
        "mostly non-built-up",
        "non-built-up",
    ]

    for category in categories:

        if category in text:
            return category

    return None


def extract_coverage(text: str):
    text = text.lower()

    # Specific categories first so "very high" is not
    # incorrectly classified as merely "high".
    categories = [
        "very high building coverage",
        "very low building coverage",
        "high building coverage",
        "moderate building coverage",
        "low building coverage",
        "no building coverage",
    ]

    for category in categories:
        if category in text:
            return category

    return None

def semantic_task_match(
    example,
    prediction: str,
) -> bool:

    prediction_norm = normalize_text(
        prediction
    )

    gt = example.ground_truth

    task = example.task

    if task == "grounded_building_summary":

        expected_percentage = gt.get(
            "building_percentage"
        )

        expected_density = gt.get(
            "density"
        )

        expected_scene = gt.get(
            "scene"
        )

        percentage = extract_percentage(
            prediction
        )

        density = extract_density(
            prediction
        )

        scene = extract_scene(
            prediction
        )

        percentage_ok = (
            percentage is not None
            and expected_percentage is not None
            and abs(
                percentage
                - float(
                    expected_percentage
                )
            ) <= 0.01
        )

        density_ok = (
            density is not None
            and expected_density is not None
            and density
            == str(
                expected_density
            ).lower()
        )

        scene_ok = (
            scene is not None
            and expected_scene is not None
            and scene
            == str(
                expected_scene
            ).lower()
        )

        return (
            percentage_ok
            and density_ok
            and scene_ok
        )

    if task == "grounded_building_density":

        expected = str(
            gt.get(
                "density",
                ""
            )
        ).lower()

        predicted = extract_density(
            prediction
        )

        return (
            predicted is not None
            and predicted == expected
        )

    if task == "grounded_building_coverage":

        expected = str(
            gt.get(
                "coverage",
                ""
            )
        ).lower()

        predicted = extract_coverage(
            prediction
        )

        return (
            predicted is not None
            and predicted == expected
        )

    if task == "grounded_scene_description":

        expected = str(
            gt.get(
                "scene",
                ""
            )
        ).lower()

        predicted = extract_scene(
            prediction
        )

        return (
            predicted is not None
            and predicted == expected
        )

    return False


def generate_answer(
    model,
    processor,
    example,
    image,
    device,
):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": (
                        "Remote-sensing evidence:\n"
                        f"{example.evidence}\n\n"
                        "Question:\n"
                        f"{example.question}"
                    ),
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
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {
        key: (
            value.to(device)
            if torch.is_tensor(value)
            else value
        )
        for key, value
        in inputs.items()
    }

    input_length = (
        inputs["input_ids"].shape[1]
    )

    with torch.no_grad():

        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_tokens = (
        generated[
            :,
            input_length:
        ]
    )

    answer = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    del generated
    del generated_tokens
    del inputs

    return answer


def build_perturbed_evidence(
    example,
) -> str:
    """
    Create a task-aware, internally consistent
    evidence perturbation.

    The image and question remain unchanged.
    Only the structured evidence is changed.

    This is deliberately different from the
    original implementation because not every
    task contains building_percentage in its
    ground_truth.
    """

    gt = dict(
        example.ground_truth
    )

    task = example.task

    if task == "grounded_building_density":

        original = str(
            gt.get(
                "density",
                "",
            )
        ).lower()

        if (
            original
            == "high building density"
        ):

            altered = (
                "no building density"
            )

        else:

            altered = (
                "high building density"
            )

        return (
            "Remote-sensing building-mask evidence: "
            "The derived density category is "
            f"'{altered}'. "
            "[Evidence perturbation test: "
            f"the original density category was "
            f"'{original}'.]"
        )

    if task == "grounded_building_coverage":

        original = str(
            gt.get(
                "coverage",
                "",
            )
        ).lower()

        if (
            original
            == "very high building coverage"
        ):

            altered = (
                "no building coverage"
            )

        else:

            altered = (
                "very high building coverage"
            )

        return (
            "Remote-sensing building-mask evidence: "
            f"The derived coverage category is "
            f"'{altered}'. "
            "[Evidence perturbation test: "
            f"the original coverage category was "
            f"'{original}'.]"
        )

    if task == "grounded_scene_description":

        original = str(
            gt.get(
                "scene",
                "",
            )
        ).lower()

        if (
            original
            == "mostly built-up"
        ):

            altered = (
                "mostly non-built-up"
            )

        else:

            altered = (
                "mostly built-up"
            )

        return (
            "Remote-sensing building-mask evidence: "
            f"The derived scene category is "
            f"'{altered}'. "
            "[Evidence perturbation test: "
            f"the original scene category was "
            f"'{original}'.]"
        )

    if task == "grounded_building_summary":

        original_percentage = float(
            gt.get(
                "building_percentage",
                0.0,
            )
        )

        original_density = str(
            gt.get(
                "density",
                "",
            )
        ).lower()

        original_scene = str(
            gt.get(
                "scene",
                "",
            )
        ).lower()

        if original_percentage == 0:

            altered_percentage = 25.0
            altered_density = (
                "high building density"
            )
            altered_scene = (
                "mostly built-up"
            )

        else:

            altered_percentage = 0.0
            altered_density = (
                "no building density"
            )
            altered_scene = (
                "mostly non-built-up"
            )

        altered_pixels = round(
            4096
            * altered_percentage
            / 100.0
        )

        return (
            "Remote-sensing building-mask evidence: "
            f"{altered_pixels} of 4096 pixels are "
            f"classified as building pixels "
            f"({altered_percentage:.2f}% coverage). "
            f"The derived density category is "
            f"'{altered_density}'. "
            f"The derived scene category is "
            f"'{altered_scene}'. "
            "[Evidence perturbation test: "
            f"the original values were "
            f"{original_percentage:.2f}% coverage, "
            f"'{original_density}', "
            f"'{original_scene}'.]"
        )

    raise ValueError(
        "Unsupported perturbation task: "
        f"{task}"
    )


def generate_with_evidence(
    model,
    processor,
    example,
    evidence,
    image,
    device,
):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": (
                        "Remote-sensing evidence:\n"
                        f"{evidence}\n\n"
                        "Question:\n"
                        f"{example.question}"
                    ),
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
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {
        key: (
            value.to(device)
            if torch.is_tensor(value)
            else value
        )
        for key, value
        in inputs.items()
    }

    input_length = (
        inputs["input_ids"].shape[1]
    )

    with torch.no_grad():

        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_tokens = (
        generated[
            :,
            input_length:
        ]
    )

    answer = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    del generated
    del generated_tokens
    del inputs

    return answer


def make_blank_image(
    image: Image.Image,
) -> Image.Image:

    return Image.new(
        "RGB",
        image.size,
        0,
    )


def make_noise_image(
    image: Image.Image,
) -> Image.Image:

    rng = np.random.default_rng(
        SEED
    )

    array = rng.integers(
        0,
        256,
        size=(
            image.height,
            image.width,
            3,
        ),
        dtype=np.uint8,
    )

    return Image.fromarray(
        array,
        mode="RGB",
    )


print()
print("==================================================")
print("PHASE 9.4H.8B.7")
print("EVIDENCE-GROUNDING EVALUATION")
print("==================================================")


print()
print("[A] ENVIRONMENT")

assert torch.cuda.is_available()

device = torch.device(
    "cuda:0"
)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "CUDA:",
    torch.version.cuda
)

assert (
    ADAPTER_PATH
    / "adapter_model.safetensors"
).exists()

print(
    "Adapter:",
    ADAPTER_PATH.resolve()
)


print()
print("[B] LOAD VALIDATION DATA")

dataset = load_validation_dataset(
    DATASET_ROOT
)

assert len(dataset) == 156

print(
    "Validation examples:",
    len(dataset)
)


print()
print("[C] LOAD BASE MODEL + ADAPTER")

cleanup()

base_model = (
    Qwen2VLForConditionalGeneration
    .from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    local_files_only=True,
)

model.eval()

processor_collator = (
    EvidenceGroundedQwenCollator(
        MODEL_PATH,
        max_length=MAX_LENGTH,
    )
)

processor = (
    processor_collator.processor
)

print(
    "Base model loaded: PASS"
)

print(
    "Evidence-grounded adapter loaded: PASS"
)

print(
    "Model device:",
    next(
        model.parameters()
    ).device
)


print()
print("==================================================")
print("[D] FULL 156-EXAMPLE VALIDATION")
print("==================================================")

normal_results = []

start = time.perf_counter()

for index in range(
    len(dataset)
):

    example = dataset[index]

    image = dataset.load_image(
        example
    )

    answer_start = time.perf_counter()

    prediction = generate_answer(
        model,
        processor,
        example,
        image,
        device,
    )

    latency = (
        time.perf_counter()
        - answer_start
    )

    exact = (
        normalize_text(
            prediction
        )
        == normalize_text(
            example.answer
        )
    )

    semantic = (
        semantic_task_match(
            example,
            prediction,
        )
    )

    normal_results.append(
        {
            "id": example.example_id,
            "task": example.task,
            "prediction": prediction,
            "reference": example.answer,
            "exact_match": exact,
            "semantic_match": semantic,
            "latency_seconds": latency,
        }
    )

    if (
        index == 0
        or (index + 1) % 25 == 0
        or index + 1 == len(dataset)
    ):

        print(
            f"Validated {index + 1}/"
            f"{len(dataset)}"
        )

normal_time = (
    time.perf_counter()
    - start
)

normal_exact = sum(
    item["exact_match"]
    for item in normal_results
)

normal_semantic = sum(
    item["semantic_match"]
    for item in normal_results
)

print()
print(
    "Normal exact matches:",
    normal_exact,
    "/",
    len(normal_results)
)

print(
    "Normal exact-match accuracy:",
    normal_exact
    / len(normal_results)
)

print(
    "Normal semantic matches:",
    normal_semantic,
    "/",
    len(normal_results)
)

print(
    "Normal semantic accuracy:",
    normal_semantic
    / len(normal_results)
)

print(
    "Mean latency:",
    np.mean(
        [
            item["latency_seconds"]
            for item in normal_results
        ]
    )
)

print(
    "Total validation time:",
    round(
        normal_time,
        2,
    ),
    "seconds"
)


task_stats = defaultdict(
    lambda: {
        "count": 0,
        "exact": 0,
        "semantic": 0,
    }
)

for item in normal_results:

    stats = task_stats[
        item["task"]
    ]

    stats["count"] += 1
    stats["exact"] += int(
        item["exact_match"]
    )
    stats["semantic"] += int(
        item["semantic_match"]
    )


print()
print("TASK-WISE RESULTS")

for task in sorted(
    task_stats
):

    stats = task_stats[
        task
    ]

    print(
        task,
        "count=",
        stats["count"],
        "exact=",
        stats["exact"],
        "semantic=",
        stats["semantic"],
        "semantic_accuracy=",
        round(
            stats["semantic"]
            / stats["count"],
            4,
        ),
    )


print()
print("==================================================")
print("[E] SHORTCUT AUDIT")
print("==================================================")

prediction_counter = Counter(
    normalize_text(
        item["prediction"]
    )
    for item in normal_results
)

print(
    "Unique normalized predictions:",
    len(prediction_counter)
)

print(
    "Most common predictions:"
)

for prediction, count in (
    prediction_counter
    .most_common(10)
):

    print(
        count,
        "->",
        prediction
    )


task_prediction_counts = {}

for task in sorted(
    task_stats
):

    task_prediction_counts[
        task
    ] = Counter(
        normalize_text(
            item["prediction"]
        )
        for item in normal_results
        if item["task"] == task
    )

for task, counts in (
    task_prediction_counts.items()
):

    print()
    print(
        "Task:",
        task
    )

    for prediction, count in (
        counts.most_common(5)
    ):

        print(
            " ",
            count,
            "->",
            prediction
        )


print()
print("==================================================")
print("[F] EVIDENCE PERTURBATION AUDIT")
print("==================================================")

perturbation_indices = list(
    range(
        min(
            PERTURBATION_LIMIT,
            len(dataset),
        )
    )
)

perturbation_results = []

changed_predictions = 0
changed_semantic_status = 0

for index in perturbation_indices:

    example = dataset[index]

    image = dataset.load_image(
        example
    )

    original_prediction = (
        normal_results[index][
            "prediction"
        ]
    )

    altered_evidence = (
        build_perturbed_evidence(
            example
        )
    )

    altered_prediction = (
        generate_with_evidence(
            model,
            processor,
            example,
            altered_evidence,
            image,
            device,
        )
    )

    changed = (
        normalize_text(
            original_prediction
        )
        != normalize_text(
            altered_prediction
        )
    )

    original_semantic = (
        normal_results[index][
            "semantic_match"
        ]
    )

    altered_semantic = (
        semantic_task_match(
            example,
            altered_prediction,
        )
    )

    semantic_changed = (
        original_semantic
        != altered_semantic
    )

    changed_predictions += int(
        changed
    )

    changed_semantic_status += int(
        semantic_changed
    )

    perturbation_results.append(
        {
            "id": example.example_id,
            "task": example.task,
            "original_evidence": example.evidence,
            "altered_evidence": altered_evidence,
            "original_prediction": (
                original_prediction
            ),
            "altered_prediction": (
                altered_prediction
            ),
            "prediction_changed": changed,
            "original_semantic_match": (
                original_semantic
            ),
            "altered_semantic_match": (
                altered_semantic
            ),
            "semantic_status_changed": (
                semantic_changed
            ),
        }
    )

    print()
    print(
        "Example:",
        example.example_id
    )

    print(
        "Original:",
        original_prediction
    )

    print(
        "Altered evidence:",
        altered_prediction
    )

    print(
        "Prediction changed:",
        changed
    )


perturbation_rate = (
    changed_predictions
    / len(
        perturbation_results
    )
)

semantic_change_rate = (
    changed_semantic_status
    / len(
        perturbation_results
    )
)

print()
print(
    "Perturbation examples:",
    len(
        perturbation_results
    )
)

print(
    "Predictions changed:",
    changed_predictions
)

print(
    "Prediction change rate:",
    perturbation_rate
)

print(
    "Semantic status changes:",
    changed_semantic_status
)

print(
    "Semantic change rate:",
    semantic_change_rate
)


print()
print("==================================================")
print("[G] IMAGE SENSITIVITY AUDIT")
print("==================================================")

sensitivity_indices = list(
    range(
        min(
            IMAGE_SENSITIVITY_LIMIT,
            len(dataset),
        )
    )
)

sensitivity_results = []

real_vs_blank = 0
real_vs_noise = 0

for index in sensitivity_indices:

    example = dataset[index]

    real_image = dataset.load_image(
        example
    )

    blank_image = make_blank_image(
        real_image
    )

    noise_image = make_noise_image(
        real_image
    )

    real_prediction = (
        normal_results[index][
            "prediction"
        ]
    )

    blank_prediction = (
        generate_answer(
            model,
            processor,
            example,
            blank_image,
            device,
        )
    )

    noise_prediction = (
        generate_answer(
            model,
            processor,
            example,
            noise_image,
            device,
        )
    )

    real_blank_changed = (
        normalize_text(
            real_prediction
        )
        != normalize_text(
            blank_prediction
        )
    )

    real_noise_changed = (
        normalize_text(
            real_prediction
        )
        != normalize_text(
            noise_prediction
        )
    )

    real_vs_blank += int(
        real_blank_changed
    )

    real_vs_noise += int(
        real_noise_changed
    )

    sensitivity_results.append(
        {
            "id": example.example_id,
            "task": example.task,
            "real_prediction": (
                real_prediction
            ),
            "blank_prediction": (
                blank_prediction
            ),
            "noise_prediction": (
                noise_prediction
            ),
            "real_vs_blank_changed": (
                real_blank_changed
            ),
            "real_vs_noise_changed": (
                real_noise_changed
            ),
        }
    )

    print()
    print(
        "Example:",
        example.example_id
    )

    print(
        "Real:",
        real_prediction
    )

    print(
        "Blank:",
        blank_prediction
    )

    print(
        "Noise:",
        noise_prediction
    )


blank_change_rate = (
    real_vs_blank
    / len(
        sensitivity_results
    )
)

noise_change_rate = (
    real_vs_noise
    / len(
        sensitivity_results
    )
)

print()
print(
    "Sensitivity examples:",
    len(
        sensitivity_results
    )
)

print(
    "Real vs blank changed:",
    real_vs_blank
)

print(
    "Real vs blank change rate:",
    blank_change_rate
)

print(
    "Real vs noise changed:",
    real_vs_noise
)

print(
    "Real vs noise change rate:",
    noise_change_rate
)


print()
print("==================================================")
print("[H] SAVE EVALUATION REPORT")
print("==================================================")

report = {
    "phase": "9.4H.8B.7",
    "status": "EVALUATED",
    "adapter": str(
        ADAPTER_PATH.resolve()
    ),
    "base_model": (
        "Qwen2-VL-2B-Instruct"
    ),
    "dataset": "SpaceNet4",
    "validation_examples": len(
        dataset
    ),
    "normal_evaluation": {
        "exact_matches": normal_exact,
        "exact_accuracy": (
            normal_exact
            / len(normal_results)
        ),
        "semantic_matches": (
            normal_semantic
        ),
        "semantic_accuracy": (
            normal_semantic
            / len(normal_results)
        ),
        "mean_latency_seconds": (
            float(
                np.mean(
                    [
                        item[
                            "latency_seconds"
                        ]
                        for item
                        in normal_results
                    ]
                )
            )
        ),
        "task_wise": {
            task: {
                "count": stats[
                    "count"
                ],
                "exact": stats[
                    "exact"
                ],
                "semantic": stats[
                    "semantic"
                ],
                "semantic_accuracy": (
                    stats["semantic"]
                    / stats["count"]
                ),
            }
            for task, stats
            in task_stats.items()
        },
    },
    "shortcut_audit": {
        "unique_normalized_predictions": (
            len(prediction_counter)
        ),
        "most_common_predictions": [
            {
                "prediction": prediction,
                "count": count,
            }
            for prediction, count
            in prediction_counter.most_common(
                10
            )
        ],
    },
    "evidence_perturbation": {
        "examples": len(
            perturbation_results
        ),
        "predictions_changed": (
            changed_predictions
        ),
        "prediction_change_rate": (
            perturbation_rate
        ),
        "semantic_status_changes": (
            changed_semantic_status
        ),
        "semantic_change_rate": (
            semantic_change_rate
        ),
        "results": perturbation_results,
    },
    "image_sensitivity": {
        "examples": len(
            sensitivity_results
        ),
        "real_vs_blank_changed": (
            real_vs_blank
        ),
        "real_vs_blank_change_rate": (
            blank_change_rate
        ),
        "real_vs_noise_changed": (
            real_vs_noise
        ),
        "real_vs_noise_change_rate": (
            noise_change_rate
        ),
        "results": sensitivity_results,
    },
    "scientific_interpretation": {
        "evidence_grounding": (
            "Must be assessed from "
            "evidence perturbation."
        ),
        "visual_reasoning": (
            "Cannot be established "
            "from validation loss alone."
        ),
        "production_readiness": (
            "Not established by this "
            "evaluation alone."
        ),
        "development_only": True,
    },
}

report_path = (
    OUTPUT_DIR
    / "grounding_evaluation_9_4H_8B_7.json"
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    "Report:",
    report_path.resolve()
)

assert report_path.exists()

print(
    "EVALUATION REPORT SAVE: PASS"
)


print()
print("==================================================")
print("[I] CLEANUP")
print("==================================================")

del model
del base_model
del processor
del processor_collator
del dataset

cleanup()

used_mib = (
    torch.cuda.memory_allocated()
    / (1024 ** 2)
)

print(
    "GPU allocated after cleanup:",
    round(
        used_mib,
        2,
    ),
    "MiB"
)

assert used_mib < 256

print(
    "GPU CLEANUP: PASS"
)


print()
print("==================================================")
print("PHASE 9.4H.8B.7 EVALUATION: COMPLETE")
print("==================================================")
