from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


BASE_MODEL = (
    "/home/lenovo/.cache/huggingface/hub/"
    "models--Qwen--Qwen2-VL-2B-Instruct/"
    "snapshots/895c3a49bc3fa70a340399125c650a463535e71c"
)

DATA_ROOT = Path(
    "data/remote_sensing/rs_vqa_visual_binary_balanced"
)

TRAIN_JSONL = DATA_ROOT / "train.jsonl"
VAL_JSONL = DATA_ROOT / "val.jsonl"

OUTPUT_DIR = Path(
    "outputs/checkpoints/qwen2vl_rs_vqa_visual_binary_dev"
)

SEED = 20260913
EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
LEARNING_RATE = 1e-4
MAX_LENGTH = 256
MAX_NEW_TOKENS = 8


def seed_everything():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_user_messages(question: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]


def build_training_messages(question: str, answer: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer},
            ],
        },
    ]


def make_example(processor, record):
    image = Image.open(record["image"]).convert("RGB")

    messages = build_training_messages(
        record["question"],
        record["answer"],
    )

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    labels = inputs["input_ids"].clone()

    # Find the prompt-only sequence.
    prompt_only = processor.apply_chat_template(
        build_user_messages(record["question"]),
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_inputs = processor(
        text=[prompt_only],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    prompt_length = prompt_inputs["input_ids"].shape[1]

    labels[:, :prompt_length] = -100

    if "attention_mask" in inputs:
        labels[
            inputs["attention_mask"] == 0
        ] = -100

    pad_token_id = processor.tokenizer.pad_token_id

    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100

    inputs["labels"] = labels

    return inputs


def move_to_device(batch, device):
    result = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value

    return result


def evaluate_loss(model, processor, records, limit=None):
    model.eval()

    selected = records if limit is None else records[:limit]

    losses = []

    with torch.no_grad():
        for record in selected:
            batch = make_example(processor, record)
            batch = move_to_device(
                batch,
                next(model.parameters()).device,
            )

            output = model(**batch)

            loss = float(output.loss.detach().cpu())

            if not torch.isfinite(output.loss):
                raise RuntimeError(
                    f"Non-finite validation loss for "
                    f"{record['id']}"
                )

            losses.append(loss)

    model.train()

    return sum(losses) / len(losses)


def main():
    seed_everything()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print("=" * 60)
    print("PHASE 9.4H.9C.3B - IMAGE-ONLY RS-VQA LoRA TRAINING")
    print("=" * 60)

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

    train_records = load_jsonl(TRAIN_JSONL)
    val_records = load_jsonl(VAL_JSONL)

    print("Training examples:", len(train_records))
    print("Validation examples:", len(val_records))

    yes = sum(r["answer"] == "YES" for r in train_records)
    no = sum(r["answer"] == "NO" for r in train_records)

    print("Training YES:", yes)
    print("Training NO :", no)

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    )

    print("Loading Qwen2-VL-2B...")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    model.eval()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora_config)

    model.train()

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    visual_trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and any(
            token in name.lower()
            for token in [
                "visual",
                "vision",
                "visual.merger",
            ]
        )
    ]

    unexpected_trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and "lora_" not in name
    ]

    print()
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    print(
        "Trainable percentage:",
        100.0 * trainable_params / total_params,
    )
    print("Visual trainable tensors:", len(visual_trainable))
    print("Unexpected trainable tensors:", len(unexpected_trainable))

    assert len(visual_trainable) == 0
    assert len(unexpected_trainable) == 0

    print()
    print("LoRA parameter audit: PASS")
    print("Visual encoder frozen: PASS")

    initial_val_loss = evaluate_loss(
        model,
        processor,
        val_records,
        limit=None,
    )

    print()
    print("Initial validation loss:", initial_val_loss)

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    optimizer.zero_grad(set_to_none=True)

    start_time = time.perf_counter()

    losses = []
    optimizer_steps = 0

    for epoch in range(EPOCHS):
        random.shuffle(train_records)

        for index, record in enumerate(train_records):
            batch = make_example(processor, record)

            batch = move_to_device(
                batch,
                next(model.parameters()).device,
            )

            output = model(**batch)

            loss = output.loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at "
                    f"example {index}"
                )

            scaled_loss = loss / GRAD_ACCUMULATION

            scaled_loss.backward()

            losses.append(float(loss.detach().cpu()))

            if (
                (index + 1) % GRAD_ACCUMULATION == 0
                or index + 1 == len(train_records)
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                optimizer_steps += 1

            if (index + 1) % 50 == 0:
                mean_recent = sum(losses[-50:]) / min(
                    50,
                    len(losses),
                )

                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"example {index + 1}/{len(train_records)} "
                    f"recent_loss={mean_recent:.6f}"
                )

    training_time = time.perf_counter() - start_time

    final_val_loss = evaluate_loss(
        model,
        processor,
        val_records,
        limit=None,
    )

    improvement = initial_val_loss - final_val_loss

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
    )

    processor.save_pretrained(OUTPUT_DIR)

    metadata = {
        "phase": "9.4H.9C.3B",
        "status": "CONTROLLED_DEV_TRAINING",
        "base_model": "Qwen2-VL-2B-Instruct",
        "base_snapshot": BASE_MODEL,
        "dataset": "rs_vqa_visual_binary_balanced",
        "train_examples": len(train_records),
        "validation_examples": len(val_records),
        "train_yes": yes,
        "train_no": no,
        "validation_yes": sum(
            r["answer"] == "YES"
            for r in val_records
        ),
        "validation_no": sum(
            r["answer"] == "NO"
            for r in val_records
        ),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "targets": ["q_proj", "v_proj"],
        },
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_percentage": (
            100.0 * trainable_params / total_params
        ),
        "visual_encoder_frozen": True,
        "evidence_supplied": False,
        "ground_truth_supplied": False,
        "image_only": True,
        "initial_validation_loss": initial_val_loss,
        "final_validation_loss": final_val_loss,
        "validation_loss_improvement": improvement,
        "mean_training_loss": (
            sum(losses) / len(losses)
        ),
        "optimizer_steps": optimizer_steps,
        "training_seconds": training_time,
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        ),
        "development_only": True,
        "warning": (
            "Training loss reduction does not establish "
            "visual RS reasoning. The adapter must pass "
            "image discrimination, counterfactual and "
            "held-out evaluation before production use."
        ),
    }

    metadata_path = OUTPUT_DIR / "training_metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    print("Initial validation loss:", initial_val_loss)
    print("Final validation loss:", final_val_loss)
    print("Improvement:", improvement)
    print(
        "Mean training loss:",
        metadata["mean_training_loss"],
    )
    print("Optimizer steps:", optimizer_steps)
    print(
        "Peak GPU memory:",
        metadata["peak_gpu_memory_gb"],
        "GB",
    )

    print()
    print("Adapter saved:", OUTPUT_DIR)
    print("Metadata saved:", metadata_path)

    del model
    del processor
    del optimizer

    gc.collect()
    torch.cuda.empty_cache()

    print()
    print(
        "GPU after cleanup:",
        torch.cuda.memory_allocated()
        / (1024 ** 3),
        "GB allocated",
    )

    print()
    print("PHASE 9.4H.9C.3B TRAINING COMPLETE")


if __name__ == "__main__":
    main()
