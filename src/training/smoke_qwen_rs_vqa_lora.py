from __future__ import annotations

import gc
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2VLForConditionalGeneration

from src.training.rs_vqa_lora_collator import (
    DEFAULT_QWEN_SNAPSHOT,
    RSVQACollator,
)
from src.training.rs_vqa_lora_data import (
    DEFAULT_DATASET_ROOT,
    RSVQADataset,
)


LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)


def find_model_device(
    model: torch.nn.Module,
) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    raise RuntimeError(
        "Could not determine model device."
    )


def load_model() -> Qwen2VLForConditionalGeneration:
    if not Path(DEFAULT_QWEN_SNAPSHOT).exists():
        raise FileNotFoundError(
            DEFAULT_QWEN_SNAPSHOT
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        DEFAULT_QWEN_SNAPSHOT,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model = get_peft_model(
        model,
        LORA_CONFIG,
    )

    return model


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the Qwen LoRA smoke test."
        )

    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    print()
    print("Loading RS-VQA dataset...")

    dataset = RSVQADataset(
        DEFAULT_DATASET_ROOT / "train.jsonl"
    )

    example = dataset[0]

    print("Example:", example.example_id)
    print("Task:", example.task)
    print("Question:", example.question)
    print("Answer:", example.answer)

    print()
    print("Preparing Qwen processor...")

    collator = RSVQACollator(
        model_path=DEFAULT_QWEN_SNAPSHOT,
        max_length=512,
    )

    batch = collator([example])

    print(
        "Input IDs:",
        tuple(batch["input_ids"].shape),
    )

    print(
        "Labels:",
        tuple(batch["labels"].shape),
    )

    print(
        "Pixel values:",
        tuple(batch["pixel_values"].shape)
        if "pixel_values" in batch
        else None,
    )

    print()
    print("Loading Qwen2-VL-2B...")

    model = load_model()

    model.print_trainable_parameters()

    print()
    print("Checking trainable parameters...")

    trainable_names = []
    unexpected_trainable = []

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_names.append(name)

            if "lora_" not in name:
                unexpected_trainable.append(name)

    print(
        "Trainable parameter tensors:",
        len(trainable_names),
    )

    print(
        "Unexpected trainable tensors:",
        len(unexpected_trainable),
    )

    if unexpected_trainable:
        print("Unexpected examples:")
        for name in unexpected_trainable[:10]:
            print(" ", name)

        raise AssertionError(
            "Non-LoRA parameters are trainable."
        )

    print("TRAINABLE PARAMETER CHECK: PASS")

    visual_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (
            "visual" in name.lower()
            or "vision" in name.lower()
        )
    ]

    print(
        "Trainable visual tensors:",
        len(visual_trainable),
    )

    if visual_trainable:
        raise AssertionError(
            "Visual encoder has trainable parameters."
        )

    print("VISUAL ENCODER FROZEN: PASS")

    model_device = find_model_device(model)

    print()
    print("Model device:", model_device)

    model_inputs = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value.to(
                model_device
            )

    labels = model_inputs.pop("labels")

    print()
    print("Running forward pass...")

    model.train()

    outputs = model(
        **model_inputs,
        labels=labels,
    )

    loss = outputs.loss

    print("Loss:", float(loss.detach().cpu()))

    if loss is None:
        raise AssertionError(
            "Model returned no loss."
        )

    if not torch.isfinite(loss):
        raise AssertionError(
            f"Non-finite loss: {loss}"
        )

    print("FINITE LOSS: PASS")

    print()
    print("Running backward pass...")

    model.zero_grad(set_to_none=True)

    loss.backward()

    print("Backward pass completed.")

    lora_with_grad = []
    lora_nonzero_grad = []

    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue

        if parameter.grad is not None:
            lora_with_grad.append(name)

            if torch.isfinite(
                parameter.grad
            ).all():
                if (
                    parameter.grad.detach().abs().sum()
                    > 0
                ):
                    lora_nonzero_grad.append(name)

    print(
        "LoRA tensors with gradients:",
        len(lora_with_grad),
    )

    print(
        "LoRA tensors with non-zero gradients:",
        len(lora_nonzero_grad),
    )

    if not lora_with_grad:
        raise AssertionError(
            "No LoRA parameters received gradients."
        )

    if not lora_nonzero_grad:
        raise AssertionError(
            "LoRA gradients are all zero."
        )

    print("LORA GRADIENT CHECK: PASS")

    print()
    print("Running one optimizer step...")

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=1e-4,
    )

    optimizer.step()

    print("Optimizer step completed.")

    print()
    print("CUDA MEMORY AFTER TRAINING STEP:")

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    peak_allocated = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        f"Allocated: {allocated:.3f} GB"
    )

    print(
        f"Reserved: {reserved:.3f} GB"
    )

    print(
        f"Peak allocated: {peak_allocated:.3f} GB"
    )

    if peak_allocated > 5.8:
        print(
            "WARNING: Peak GPU allocation is close "
            "to the 6 GB hardware limit."
        )

    print()
    print("==================================================")
    print("ALL 9.4D.3 CHECKS PASSED")
    print("==================================================")

    del optimizer
    del outputs
    del model
    del collator
    del dataset

    gc.collect()

    torch.cuda.empty_cache()

    print()
    print("GPU memory after cleanup:")

    print(
        f"Allocated: "
        f"{torch.cuda.memory_allocated() / 1024**3:.3f} GB"
    )

    print(
        f"Reserved: "
        f"{torch.cuda.memory_reserved() / 1024**3:.3f} GB"
    )


if __name__ == "__main__":
    main()
