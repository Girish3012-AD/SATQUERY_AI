from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from src.training.rs_vqa_lora_data import RSVQADataset


BASE_MODEL = (
    "/home/lenovo/.cache/huggingface/hub/"
    "models--Qwen--Qwen2-VL-2B-Instruct/"
    "snapshots/895c3a49bc3fa70a340399125c650a463535e71c"
)

DATASET_ROOT = Path("data/remote_sensing/rs_vqa_visual")
VAL_JSONL = DATASET_ROOT / "val.jsonl"

OUTPUT_DIR = Path(
    "outputs/checkpoints/qwen2vl_rs_vqa_visual_dev"
)

MAX_EXAMPLES = 20
MAX_NEW_TOKENS = 32
SEED = 20260913


def normalize_answer(text: str) -> str:
    return " ".join(text.lower().strip().split())


def load_model():
    print("Loading base Qwen2-VL-2B...")

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    model.eval()

    return processor, model


def run_single(
    processor,
    model,
    image: Image.Image,
    question: str,
) -> tuple[str, float]:

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": question,
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

    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(device)

    start = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    latency = time.perf_counter() - start

    prompt_len = inputs["input_ids"].shape[1]

    generated_tokens = generated[:, prompt_len:]

    answer = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    return answer, latency


def make_blank(image: Image.Image) -> Image.Image:
    return Image.new("RGB", image.size, 255)


def make_black(image: Image.Image) -> Image.Image:
    return Image.new("RGB", image.size, 0)


def make_noise(image: Image.Image, seed: int) -> Image.Image:
    generator = random.Random(seed)

    pixels = []

    for _ in range(image.width * image.height):
        pixels.append(
            (
                generator.randrange(256),
                generator.randrange(256),
                generator.randrange(256),
            )
        )

    noisy = Image.new("RGB", image.size)
    noisy.putdata(pixels)

    return noisy


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("=" * 60)
    print("PHASE 9.4H.9B - BASE QWEN VISUAL BASELINE")
    print("=" * 60)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Dataset: {VAL_JSONL}")

    dataset = RSVQADataset(VAL_JSONL)

    print(f"Validation examples: {len(dataset)}")

    processor, model = load_model()

    print("Model device:", next(model.parameters()).device)

    examples = []

    for index in range(min(MAX_EXAMPLES, len(dataset))):
        examples.append(dataset[index])

    results = []

    print()
    print("=" * 60)
    print("REAL IMAGE BASELINE")
    print("=" * 60)

    for index, example in enumerate(examples):
        image = Image.open(example.image).convert('RGB')

        answer, latency = run_single(
            processor,
            model,
            image,
            example.question,
        )

        normalized_prediction = normalize_answer(answer)
        normalized_target = normalize_answer(example.answer)

        exact = normalized_prediction == normalized_target

        results.append(
            {
                "index": index,
                "example_id": example.example_id,
                "task": example.task,
                "question": example.question,
                "target": example.answer,
                "prediction": answer,
                "exact_match": exact,
                "latency_seconds": latency,
                "image": str(example.image),
            }
        )

        print()
        print(f"[{index + 1}/{len(examples)}]")
        print("ID:", example.example_id)
        print("Task:", example.task)
        print("Question:", example.question)
        print("Target:", example.answer)
        print("Prediction:", answer)
        print("Exact:", exact)
        print(f"Latency: {latency:.3f}s")

    print()
    print("=" * 60)
    print("IMAGE SENSITIVITY TEST")
    print("=" * 60)

    sensitivity_results = []

    for index, example in enumerate(examples[:10]):
        image = Image.open(example.image).convert('RGB')

        real_answer, _ = run_single(
            processor,
            model,
            image,
            example.question,
        )

        blank_answer, _ = run_single(
            processor,
            model,
            make_blank(image),
            example.question,
        )

        black_answer, _ = run_single(
            processor,
            model,
            make_black(image),
            example.question,
        )

        noise_answer, _ = run_single(
            processor,
            model,
            make_noise(image, SEED + index),
            example.question,
        )

        changed_blank = (
            normalize_answer(real_answer)
            != normalize_answer(blank_answer)
        )

        changed_black = (
            normalize_answer(real_answer)
            != normalize_answer(black_answer)
        )

        changed_noise = (
            normalize_answer(real_answer)
            != normalize_answer(noise_answer)
        )

        record = {
            "example_id": example.example_id,
            "question": example.question,
            "real": real_answer,
            "blank": blank_answer,
            "black": black_answer,
            "noise": noise_answer,
            "real_vs_blank_changed": changed_blank,
            "real_vs_black_changed": changed_black,
            "real_vs_noise_changed": changed_noise,
        }

        sensitivity_results.append(record)

        print()
        print(f"[{index + 1}/{min(10, len(examples))}]")
        print("ID:", example.example_id)
        print("REAL :", real_answer)
        print("BLANK:", blank_answer)
        print("BLACK:", black_answer)
        print("NOISE:", noise_answer)
        print("Real vs blank :", changed_blank)
        print("Real vs black :", changed_black)
        print("Real vs noise :", changed_noise)

    real_blank_changed = sum(
        r["real_vs_blank_changed"]
        for r in sensitivity_results
    )

    real_black_changed = sum(
        r["real_vs_black_changed"]
        for r in sensitivity_results
    )

    real_noise_changed = sum(
        r["real_vs_noise_changed"]
        for r in sensitivity_results
    )

    total_sensitivity = len(sensitivity_results)

    summary = {
        "phase": "9.4H.9B",
        "status": "COMPLETED",
        "base_model": "Qwen2-VL-2B-Instruct",
        "dataset": str(VAL_JSONL),
        "examples_evaluated": len(examples),
        "real_image_exact_match": (
            sum(r["exact_match"] for r in results) / len(results)
            if results else 0.0
        ),
        "image_sensitivity": {
            "examples": total_sensitivity,
            "real_vs_blank_changed": real_blank_changed,
            "real_vs_black_changed": real_black_changed,
            "real_vs_noise_changed": real_noise_changed,
            "real_vs_blank_rate": (
                real_blank_changed / total_sensitivity
                if total_sensitivity
                else 0.0
            ),
            "real_vs_black_rate": (
                real_black_changed / total_sensitivity
                if total_sensitivity
                else 0.0
            ),
            "real_vs_noise_rate": (
                real_noise_changed / total_sensitivity
                if total_sensitivity
                else 0.0
            ),
        },
        "development_only": True,
        "warning": (
            "This is a base-model visual sensitivity baseline. "
            "It is not an RS-VLM benchmark."
        ),
        "results": results,
        "sensitivity_results": sensitivity_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "base_visual_baseline_9_4H_9B.json"

    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)

    print(
        f"Real-image exact match: "
        f"{summary['real_image_exact_match']:.4f}"
    )

    print(
        "Real vs blank changed:",
        f"{real_blank_changed}/{total_sensitivity}",
    )

    print(
        "Real vs black changed:",
        f"{real_black_changed}/{total_sensitivity}",
    )

    print(
        "Real vs noise changed:",
        f"{real_noise_changed}/{total_sensitivity}",
    )

    print()
    print("Saved:", output_path)

    del model
    del processor

    gc.collect()

    torch.cuda.empty_cache()

    print()
    print(
        "GPU after cleanup:",
        torch.cuda.memory_allocated() / (1024**3),
        "GB allocated",
    )

    print()
    print("PHASE 9.4H.9B BASELINE COMPLETE")


if __name__ == "__main__":
    main()
