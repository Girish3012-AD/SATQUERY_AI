from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2VLForConditionalGeneration

from src.training.rs_vqa_lora_collator import (
    DEFAULT_QWEN_SNAPSHOT,
    RSVQACollator,
)
from src.training.rs_vqa_lora_data_balanced import (
    DEFAULT_DATASET_ROOT,
    RSVQADataset,
)


SEED = 20260913

OUTPUT_DIR = (
    Path("outputs/checkpoints")
    / "qwen2vl_rs_vqa_lora_balanced_dev"
)

TRAIN_JSONL = DEFAULT_DATASET_ROOT / "train.jsonl"
VAL_JSONL = DEFAULT_DATASET_ROOT / "val.jsonl"

MAX_LENGTH = 512
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 4

EPOCHS = 1

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model():
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


def model_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    raise RuntimeError(
        "Unable to determine model device."
    )


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def validate_trainable_parameters(model) -> None:
    unexpected = []
    visual = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if "lora_" not in name:
            unexpected.append(name)

        if (
            "visual" in name.lower()
            or "vision" in name.lower()
        ):
            visual.append(name)

    print(
        "Trainable parameter tensors:",
        sum(
            1
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )

    print(
        "Unexpected trainable tensors:",
        len(unexpected),
    )

    print(
        "Trainable visual tensors:",
        len(visual),
    )

    if unexpected:
        raise AssertionError(
            "Unexpected non-LoRA trainable parameters."
        )

    if visual:
        raise AssertionError(
            "Visual encoder is unexpectedly trainable."
        )

    print("TRAINABLE PARAMETER CHECK: PASS")
    print("VISUAL ENCODER FROZEN: PASS")


def evaluate(
    model,
    dataset,
    collator,
    device,
    max_examples: int | None = None,
) -> float:
    model.eval()

    losses = []

    if max_examples is None:
        max_examples = len(dataset)

    with torch.no_grad():
        for index in range(
            min(max_examples, len(dataset))
        ):
            batch = collator([dataset[index]])

            batch = move_batch_to_device(
                batch,
                device,
            )

            outputs = model(
                **batch
            )

            loss = outputs.loss

            if loss is None:
                raise RuntimeError(
                    "Validation returned no loss."
                )

            loss_value = float(
                loss.detach().cpu()
            )

            if not np.isfinite(loss_value):
                raise RuntimeError(
                    f"Non-finite validation loss: "
                    f"{loss_value}"
                )

            losses.append(loss_value)

    if not losses:
        raise RuntimeError(
            "Validation produced no losses."
        )

    return float(np.mean(losses))


def save_metadata(
    path: Path,
    train_count: int,
    val_count: int,
    train_loss: float,
    val_loss: float,
    peak_memory_gb: float,
) -> None:
    metadata = {
        "phase": "9.4E",
        "status": "CONTROLLED_DEV_TRAINING",
        "seed": SEED,
        "base_model": "Qwen2-VL-2B-Instruct",
        "base_snapshot": DEFAULT_QWEN_SNAPSHOT,
        "dataset": "SpaceNet4 RS-VQA development dataset",
        "train_examples": train_count,
        "validation_examples": val_count,
        "epochs": EPOCHS,
        "batch_size": 1,
        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "targets": [
                "q_proj",
                "v_proj",
            ],
        },
        "development_only": True,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "peak_gpu_memory_gb": peak_memory_gb,
        "warning": (
            "This is a development adaptation run. "
            "It is not an independent benchmark."
        ),
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.cuda.reset_peak_memory_stats()

    print("GPU:")
    print(torch.cuda.get_device_name(0))

    print()
    print("Loading datasets...")

    train_dataset = RSVQADataset(
        TRAIN_JSONL
    )

    val_dataset = RSVQADataset(
        VAL_JSONL
    )

    print(
        "Training examples:",
        len(train_dataset),
    )

    print(
        "Validation examples:",
        len(val_dataset),
    )

    print()
    print("Loading Qwen processor...")

    collator = RSVQACollator(
        model_path=DEFAULT_QWEN_SNAPSHOT,
        max_length=MAX_LENGTH,
    )

    print()
    print("Loading Qwen2-VL-2B + LoRA...")

    model = load_model()

    model.print_trainable_parameters()

    validate_trainable_parameters(model)

    device = model_device(model)

    print()
    print("Model device:", device)

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=LEARNING_RATE,
    )

    print()
    print("==================================================")
    print("INITIAL VALIDATION")
    print("==================================================")

    initial_val_loss = evaluate(
        model,
        val_dataset,
        collator,
        device,
        max_examples=len(val_dataset),
    )

    print(
        f"Initial validation loss: "
        f"{initial_val_loss:.6f}"
    )

    print()
    print("==================================================")
    print("TRAINING")
    print("==================================================")

    model.train()

    running_losses = []
    optimizer.zero_grad(
        set_to_none=True
    )

    global_step = 0

    for epoch in range(EPOCHS):
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for index in range(
            len(train_dataset)
        ):
            batch = collator(
                [train_dataset[index]]
            )

            batch = move_batch_to_device(
                batch,
                device,
            )

            outputs = model(
                **batch
            )

            loss = outputs.loss

            if loss is None:
                raise RuntimeError(
                    "Training returned no loss."
                )

            raw_loss = float(
                loss.detach().cpu()
            )

            if not np.isfinite(raw_loss):
                raise RuntimeError(
                    f"Non-finite training loss "
                    f"at example {index}: "
                    f"{raw_loss}"
                )

            running_losses.append(
                raw_loss
            )

            scaled_loss = (
                loss
                / GRADIENT_ACCUMULATION_STEPS
            )

            scaled_loss.backward()

            if (
                (index + 1)
                % GRADIENT_ACCUMULATION_STEPS
                == 0
            ):
                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

            if (
                (index + 1) % 50 == 0
                or index == 0
            ):
                mean_recent = float(
                    np.mean(
                        running_losses[-50:]
                    )
                )

                print(
                    f"  example "
                    f"{index + 1}/"
                    f"{len(train_dataset)} "
                    f"| loss={raw_loss:.4f} "
                    f"| recent50="
                    f"{mean_recent:.4f} "
                    f"| optimizer_step="
                    f"{global_step}"
                )

        # Flush remaining accumulated gradients.
        remaining = (
            len(train_dataset)
            % GRADIENT_ACCUMULATION_STEPS
        )

        if remaining != 0:
            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            global_step += 1

    train_loss = float(
        np.mean(running_losses)
    )

    print()
    print(
        f"Training mean loss: "
        f"{train_loss:.6f}"
    )

    print()
    print("==================================================")
    print("FINAL VALIDATION")
    print("==================================================")

    final_val_loss = evaluate(
        model,
        val_dataset,
        collator,
        device,
        max_examples=len(val_dataset),
    )

    print(
        f"Final validation loss: "
        f"{final_val_loss:.6f}"
    )

    improvement = (
        initial_val_loss
        - final_val_loss
    )

    print(
        f"Validation improvement: "
        f"{improvement:.6f}"
    )

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print()
    print("==================================================")
    print("GPU MEMORY")
    print("==================================================")

    print(
        f"Peak allocated: "
        f"{peak_memory:.3f} GB"
    )

    if peak_memory >= 5.9:
        raise RuntimeError(
            "Peak GPU memory is too close to "
            "the 6 GB hardware limit."
        )

    print("GPU MEMORY SAFETY: PASS")

    print()
    print("==================================================")
    print("SAVING LORA ADAPTER")
    print("==================================================")

    model.save_pretrained(
        OUTPUT_DIR
    )

    collator.processor.save_pretrained(
        OUTPUT_DIR
    )

    metadata_path = (
        OUTPUT_DIR / "training_metadata.json"
    )

    save_metadata(
        metadata_path,
        len(train_dataset),
        len(val_dataset),
        train_loss,
        final_val_loss,
        peak_memory,
    )

    adapter_files = sorted(
        path.name
        for path in OUTPUT_DIR.iterdir()
        if path.is_file()
    )

    print(
        "Saved files:"
    )

    for filename in adapter_files:
        print(
            " ",
            filename,
        )

    print()
    print("==================================================")
    print("PHASE 9.4E CONTROLLED TRAINING COMPLETE")
    print("==================================================")

    print(
        f"Initial validation loss: "
        f"{initial_val_loss:.6f}"
    )

    print(
        f"Final validation loss: "
        f"{final_val_loss:.6f}"
    )

    print(
        f"Training mean loss: "
        f"{train_loss:.6f}"
    )

    print(
        f"Peak GPU memory: "
        f"{peak_memory:.3f} GB"
    )

    print(
        f"Adapter directory: "
        f"{OUTPUT_DIR}"
    )

    del optimizer
    del model
    del collator
    del train_dataset
    del val_dataset

    gc.collect()
    torch.cuda.empty_cache()

    print()
    print("GPU after cleanup:")

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
