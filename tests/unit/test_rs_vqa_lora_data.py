from pathlib import Path

from src.training.rs_vqa_lora_data import (
    DEFAULT_DATASET_ROOT,
    RSVQADataset,
    build_qwen_messages,
    build_qwen_prompt_messages,
    load_image,
)


def test_train_dataset_loads():
    dataset = RSVQADataset(
        DEFAULT_DATASET_ROOT / "train.jsonl"
    )

    assert len(dataset) == 628

    example = dataset[0]

    assert example.example_id
    assert example.image.exists()
    assert example.question
    assert example.answer
    assert example.task
    assert example.source == "SpaceNet4"
    assert example.split == "train"


def test_validation_dataset_loads():
    dataset = RSVQADataset(
        DEFAULT_DATASET_ROOT / "val.jsonl"
    )

    assert len(dataset) == 156

    example = dataset[0]

    assert example.image.exists()
    assert example.split == "val"


def test_image_loading():
    dataset = RSVQADataset(
        DEFAULT_DATASET_ROOT / "train.jsonl"
    )

    image = load_image(dataset[0])

    assert image.mode == "RGB"
    assert image.width > 0
    assert image.height > 0


def test_qwen_multimodal_messages():
    dataset = RSVQADataset(
        DEFAULT_DATASET_ROOT / "train.jsonl"
    )

    example = dataset[0]

    messages = build_qwen_messages(example)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"

    user_content = messages[0]["content"]

    assert user_content[0]["type"] == "image"
    assert user_content[1]["type"] == "text"
    assert user_content[1]["text"] == example.question

    assert messages[1]["content"][0]["text"] == example.answer


def test_qwen_prompt_contains_only_user_turn():
    dataset = RSVQADataset(
        DEFAULT_DATASET_ROOT / "train.jsonl"
    )

    example = dataset[0]

    messages = build_qwen_prompt_messages(example)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "image"
    assert messages[0]["content"][1]["text"] == example.question
