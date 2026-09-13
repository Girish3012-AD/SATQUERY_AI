from __future__ import annotations

import gc
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2VLForConditionalGeneration

from src.training.rs_vqa_evidence_grounded_data import (
    load_train_dataset,
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


def cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


print()
print("==================================================")
print("[A] ENVIRONMENT")
print("==================================================")

assert torch.cuda.is_available()

device = torch.device("cuda:0")

print(
    "CUDA available:",
    torch.cuda.is_available()
)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "CUDA version:",
    torch.version.cuda
)


print()
print("==================================================")
print("[B] LOAD DATASET")
print("==================================================")

dataset = load_train_dataset(
    DATASET_ROOT
)

assert len(dataset) == 628

example = dataset[0]

print(
    "Example:",
    example.example_id
)

print(
    "Task:",
    example.task
)

print(
    "Image:",
    example.image
)

print(
    "Source patch:",
    example.source_raster_patch
)

print(
    "Evidence:",
    example.evidence
)

print(
    "Question:",
    example.question
)

print(
    "Target:",
    example.answer
)


print()
print("==================================================")
print("[C] BUILD EVIDENCE-AWARE BATCH")
print("==================================================")

collator = (
    EvidenceGroundedQwenCollator(
        MODEL_PATH,
        max_length=512,
    )
)

batch = collator(
    [example]
)

print(
    "Batch keys:",
    sorted(batch.keys())
)

assert "input_ids" in batch
assert "attention_mask" in batch
assert "pixel_values" in batch
assert "image_grid_thw" in batch
assert "labels" in batch

print(
    "input_ids:",
    tuple(
        batch["input_ids"].shape
    )
)

print(
    "pixel_values:",
    tuple(
        batch["pixel_values"].shape
    )
)

print(
    "labels:",
    tuple(
        batch["labels"].shape
    )
)


print()
print("==================================================")
print("[D] LOAD QWEN2-VL-2B")
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
    "Model loaded:",
    True
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

model.print_trainable_parameters()


trainable_params = 0
total_params = 0
trainable_tensors = 0
visual_trainable = 0
unexpected_trainable = []

for name, parameter in model.named_parameters():

    total_params += parameter.numel()

    if parameter.requires_grad:

        trainable_params += (
            parameter.numel()
        )

        trainable_tensors += 1

        if (
            "visual" in name
            or "vision" in name
        ):
            visual_trainable += 1

        if (
            "lora_A" not in name
            and "lora_B" not in name
        ):
            unexpected_trainable.append(
                name
            )


trainable_percent = (
    100.0
    * trainable_params
    / total_params
)

print(
    "Total parameters:",
    total_params
)

print(
    "Trainable parameters:",
    trainable_params
)

print(
    "Trainable percentage:",
    trainable_percent
)

print(
    "Trainable tensors:",
    trainable_tensors
)

print(
    "Visual trainable tensors:",
    visual_trainable
)

print(
    "Unexpected trainable tensors:",
    len(
        unexpected_trainable
    )
)

assert trainable_params > 0
assert trainable_tensors == 112
assert visual_trainable == 0
assert not unexpected_trainable

print(
    "LORA TARGET VERIFICATION: PASS"
)


print()
print("==================================================")
print("[F] MOVE BATCH TO GPU")
print("==================================================")

gpu_batch = {}

for key, value in batch.items():

    if torch.is_tensor(value):

        gpu_batch[key] = value.to(
            device
        )

        print(
            key,
            "->",
            gpu_batch[key].device
        )

    else:
        gpu_batch[key] = value


print()
print("==================================================")
print("[G] FORWARD PASS")
print("==================================================")

cleanup()

torch.cuda.reset_peak_memory_stats()

outputs = model(
    **gpu_batch
)

loss = outputs.loss

print(
    "Loss:",
    float(loss.detach().cpu())
)

assert loss is not None
assert torch.isfinite(
    loss
).item()

print(
    "FINITE LOSS: PASS"
)


print()
print("==================================================")
print("[H] BACKWARD PASS")
print("==================================================")

model.zero_grad(
    set_to_none=True
)

loss.backward()

gradient_tensors = 0
nonzero_gradient_tensors = 0

for name, parameter in (
    model.named_parameters()
):

    if (
        parameter.requires_grad
        and parameter.grad is not None
    ):

        gradient_tensors += 1

        if (
            parameter.grad
            .detach()
            .abs()
            .sum()
            .item()
            > 0
        ):

            nonzero_gradient_tensors += 1


print(
    "LoRA tensors with gradients:",
    gradient_tensors
)

print(
    "LoRA tensors with nonzero gradients:",
    nonzero_gradient_tensors
)

assert gradient_tensors == 112
assert nonzero_gradient_tensors > 0

print(
    "BACKWARD GRADIENTS: PASS"
)


print()
print("==================================================")
print("[I] OPTIMIZER STEP")
print("==================================================")

optimizer = torch.optim.AdamW(
    (
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ),
    lr=1e-4,
)

optimizer.step()

print(
    "Optimizer step:",
    "completed"
)

print(
    "OPTIMIZER STEP: PASS"
)


print()
print("==================================================")
print("[J] GPU MEMORY")
print("==================================================")

allocated = (
    torch.cuda.memory_allocated()
    / (1024 ** 3)
)

reserved = (
    torch.cuda.memory_reserved()
    / (1024 ** 3)
)

peak = (
    torch.cuda.max_memory_allocated()
    / (1024 ** 3)
)

print(
    "Allocated GB:",
    round(allocated, 4)
)

print(
    "Reserved GB:",
    round(reserved, 4)
)

print(
    "Peak allocated GB:",
    round(peak, 4)
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
del outputs
del loss
del model
del collator
del batch
del gpu_batch

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
print("PHASE 9.4H.8B.5 CUDA LORA SMOKE: PASS")
print("==================================================")
