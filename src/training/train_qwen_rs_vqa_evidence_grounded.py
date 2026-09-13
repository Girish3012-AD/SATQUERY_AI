from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2VLForConditionalGeneration

from src.training.rs_vqa_evidence_grounded_data import (
    load_train_dataset,
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

DATASET_ROOT = Path(
    "data/remote_sensing/"
    "rs_vqa_evidence_grounded"
)

OUTPUT_DIR = Path(
    "outputs/checkpoints/"
    "qwen2vl_rs_vqa_evidence_grounded_dev"
)

SEED = 20260913

EPOCHS = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
MAX_LENGTH = 512

LOG_EVERY = 25


def seed_everything(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup() -> None:

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def evaluate_loss(
    model,
    dataset,
    collator,
    device,
):
    model.eval()

    total_loss = 0.0
    count = 0

    with torch.no_grad():

        for index in range(len(dataset)):

            example = dataset[index]

            batch = collator(
                [example]
            )

            gpu_batch = {
                key: (
                    value.to(device)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value
                in batch.items()
            }

            outputs = model(
                **gpu_batch
            )

            loss = outputs.loss

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite validation loss "
                    f"at index {index}"
                )

            total_loss += float(
                loss.detach().cpu()
            )

            count += 1

            del outputs
            del loss
            del gpu_batch
            del batch

    model.train()

    return total_loss / count


print()
print("==================================================")
print("[A] INITIALIZE")
print("==================================================")

assert torch.cuda.is_available()

device = torch.device("cuda:0")

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "CUDA:",
    torch.version.cuda
)

seed_everything(SEED)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print()
print("==================================================")
print("[B] LOAD DATASETS")
print("==================================================")

train_dataset = load_train_dataset(
    DATASET_ROOT
)

val_dataset = load_validation_dataset(
    DATASET_ROOT
)

assert len(train_dataset) == 628
assert len(val_dataset) == 156

print(
    "Raw train examples:",
    len(train_dataset)
)

print(
    "Validation examples:",
    len(val_dataset)
)

print(
    "Training configuration:",
    "all 628 examples"
)

print(
    "Validation configuration:",
    "all 156 examples"
)


print()
print("==================================================")
print("[C] CREATE COLLATOR")
print("==================================================")

collator = EvidenceGroundedQwenCollator(
    MODEL_PATH,
    max_length=MAX_LENGTH,
)

print(
    "Collator:",
    type(collator).__name__
)

print(
    "Max length:",
    MAX_LENGTH
)


print()
print("==================================================")
print("[D] LOAD BASE QWEN2-VL")
print("==================================================")

cleanup()

model = (
    Qwen2VLForConditionalGeneration
    .from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
)

model.train()

print(
    "Base model loaded: PASS"
)

print(
    "Model device:",
    next(
        model.parameters()
    ).device
)


print()
print("==================================================")
print("[E] APPLY LORA")
print("==================================================")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "v_proj",
    ],
)

model = get_peft_model(
    model,
    lora_config
)

trainable = 0
total = 0
visual_trainable = 0

for name, parameter in (
    model.named_parameters()
):

    total += parameter.numel()

    if parameter.requires_grad:

        trainable += parameter.numel()

        if (
            "visual" in name
            or "vision" in name
        ):
            visual_trainable += 1

assert trainable == 1_089_536
assert visual_trainable == 0

print(
    "Total parameters:",
    total
)

print(
    "Trainable parameters:",
    trainable
)

print(
    "Trainable percentage:",
    100.0 * trainable / total
)

print(
    "Visual trainable tensors:",
    visual_trainable
)

print(
    "LORA CONFIGURATION: PASS"
)


print()
print("==================================================")
print("[F] INITIAL VALIDATION")
print("==================================================")

cleanup()

torch.cuda.reset_peak_memory_stats()

start = time.perf_counter()

initial_val_loss = evaluate_loss(
    model,
    val_dataset,
    collator,
    device,
)

elapsed = (
    time.perf_counter()
    - start
)

print(
    "Initial validation loss:",
    initial_val_loss
)

print(
    "Validation examples:",
    len(val_dataset)
)

print(
    "Validation time seconds:",
    round(elapsed, 2)
)

assert np.isfinite(
    initial_val_loss
)

print(
    "INITIAL VALIDATION: PASS"
)


print()
print("==================================================")
print("[G] CONTROLLED TRAINING")
print("==================================================")

optimizer = torch.optim.AdamW(
    (
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ),
    lr=LEARNING_RATE,
)

model.train()

global_step = 0
optimizer_steps = 0

running_loss = 0.0
example_count = 0

training_start = time.perf_counter()

for epoch in range(EPOCHS):

    epoch_start = time.perf_counter()

    indices = list(
        range(
            len(train_dataset)
        )
    )

    random.shuffle(indices)

    for local_index, dataset_index in enumerate(
        indices
    ):

        example = train_dataset[
            dataset_index
        ]

        batch = collator(
            [example]
        )

        gpu_batch = {
            key: (
                value.to(device)
                if torch.is_tensor(value)
                else value
            )
            for key, value
            in batch.items()
        }

        outputs = model(
            **gpu_batch
        )

        loss = outputs.loss

        if not torch.isfinite(
            loss
        ).item():

            raise RuntimeError(
                f"Non-finite training loss "
                f"at example "
                f"{dataset_index}"
            )

        scaled_loss = (
            loss
            / GRADIENT_ACCUMULATION_STEPS
        )

        scaled_loss.backward()

        running_loss += float(
            loss.detach().cpu()
        )

        example_count += 1
        global_step += 1

        if (
            global_step
            % GRADIENT_ACCUMULATION_STEPS
            == 0
        ):

            torch.nn.utils.clip_grad_norm_(
                (
                    parameter
                    for parameter
                    in model.parameters()
                    if parameter.requires_grad
                ),
                max_norm=1.0,
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            optimizer_steps += 1

        if (
            global_step == 1
            or global_step % LOG_EVERY == 0
            or global_step == len(train_dataset)
        ):

            mean_loss = (
                running_loss
                / example_count
            )

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"example {global_step}/"
                f"{len(train_dataset)} "
                f"mean_loss={mean_loss:.6f}"
            )

        del outputs
        del loss
        del scaled_loss
        del gpu_batch
        del batch

    if (
        global_step
        % GRADIENT_ACCUMULATION_STEPS
        != 0
    ):

        torch.nn.utils.clip_grad_norm_(
            (
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            max_norm=1.0,
        )

        optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        optimizer_steps += 1

    epoch_time = (
        time.perf_counter()
        - epoch_start
    )

    print(
        "Epoch time seconds:",
        round(epoch_time, 2)
    )


training_time = (
    time.perf_counter()
    - training_start
)

train_mean_loss = (
    running_loss
    / example_count
)

print()
print(
    "Training examples processed:",
    example_count
)

print(
    "Optimizer steps:",
    optimizer_steps
)

print(
    "Mean training loss:",
    train_mean_loss
)

print(
    "Training time seconds:",
    round(training_time, 2)
)

assert example_count == 628
assert optimizer_steps > 0
assert np.isfinite(
    train_mean_loss
)

print(
    "CONTROLLED TRAINING: PASS"
)


print()
print("==================================================")
print("[H] FINAL VALIDATION")
print("==================================================")

cleanup()

torch.cuda.reset_peak_memory_stats()

start = time.perf_counter()

final_val_loss = evaluate_loss(
    model,
    val_dataset,
    collator,
    device,
)

elapsed = (
    time.perf_counter()
    - start
)

print(
    "Final validation loss:",
    final_val_loss
)

print(
    "Validation examples:",
    len(val_dataset)
)

print(
    "Validation time seconds:",
    round(elapsed, 2)
)

assert np.isfinite(
    final_val_loss
)

improvement = (
    initial_val_loss
    - final_val_loss
)

if initial_val_loss != 0:

    improvement_percent = (
        100.0
        * improvement
        / abs(initial_val_loss)
    )

else:
    improvement_percent = 0.0

print(
    "Validation loss improvement:",
    improvement
)

print(
    "Validation loss improvement percent:",
    improvement_percent
)

print(
    "FINAL VALIDATION: PASS"
)


print()
print("==================================================")
print("[I] SAVE ADAPTER")
print("==================================================")

model.save_pretrained(
    OUTPUT_DIR
)

collator.processor.save_pretrained(
    OUTPUT_DIR
)

metadata = {
    "phase": "9.4H.8B.6",
    "status": "CONTROLLED_DEV_TRAINING",
    "seed": SEED,
    "base_model": "Qwen2-VL-2B-Instruct",
    "base_snapshot": MODEL_PATH,
    "dataset": (
        "SpaceNet4 evidence-grounded "
        "RS-VQA development dataset"
    ),
    "train_examples": len(
        train_dataset
    ),
    "validation_examples": len(
        val_dataset
    ),
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
    "trainable_parameters": trainable,
    "trainable_percentage": (
        100.0 * trainable / total
    ),
    "initial_validation_loss": (
        initial_val_loss
    ),
    "training_mean_loss": (
        train_mean_loss
    ),
    "final_validation_loss": (
        final_val_loss
    ),
    "validation_loss_improvement": (
        improvement
    ),
    "validation_loss_improvement_percent": (
        improvement_percent
    ),
    "development_only": True,
    "visual_encoder_trainable": False,
    "warning": (
        "This adapter is a development "
        "evidence-grounded adaptation. "
        "Validation loss improvement does "
        "not establish visual reasoning or "
        "production readiness. Dedicated "
        "grounding and perturbation audits "
        "are required before integration."
    ),
}

metadata_path = (
    OUTPUT_DIR / "training_metadata.json"
)

metadata_path.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8",
)

print(
    "Adapter directory:",
    OUTPUT_DIR.resolve()
)

print(
    "Metadata:",
    metadata_path.resolve()
)

assert (
    OUTPUT_DIR / "adapter_model.safetensors"
).exists()

assert metadata_path.exists()

print(
    "ADAPTER SAVE: PASS"
)


print()
print("==================================================")
print("[J] GPU MEMORY")
print("==================================================")

peak = (
    torch.cuda.max_memory_allocated()
    / (1024 ** 3)
)

allocated = (
    torch.cuda.memory_allocated()
    / (1024 ** 3)
)

reserved = (
    torch.cuda.memory_reserved()
    / (1024 ** 3)
)

print(
    "Peak allocated GB:",
    round(peak, 4)
)

print(
    "Allocated GB:",
    round(allocated, 4)
)

print(
    "Reserved GB:",
    round(reserved, 4)
)

assert peak < 5.8

print(
    "GPU MEMORY SAFETY: PASS"
)


print()
print("==================================================")
print("[K] CLEANUP")
print("==================================================")

del optimizer
del model
del collator
del train_dataset
del val_dataset

cleanup()

used_mib = (
    torch.cuda.memory_allocated()
    / (1024 ** 2)
)

print(
    "GPU allocated after cleanup:",
    round(used_mib, 2),
    "MiB"
)

assert used_mib < 256

print(
    "GPU CLEANUP: PASS"
)


print()
print("==================================================")
print("PHASE 9.4H.8B.6 CONTROLLED TRAINING: PASS")
print("==================================================")
