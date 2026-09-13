from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "data" / "remote_sensing" / "rs_vqa_balanced"
)


@dataclass(frozen=True)
class RSVQAExample:
    """
    One grounded remote-sensing VQA training example.

    The image and answer originate from the project-generated
    SpaceNet RS-VQA dataset.
    """

    example_id: str
    image: Path
    question: str
    answer: str
    task: str
    source: str
    split: str
    ground_truth: dict[str, Any]


class RSVQADataset(Dataset[RSVQAExample]):
    """
    Dataset reader for the generated RS-VQA JSONL files.

    This class intentionally performs no model-specific processing.
    It only validates and exposes grounded examples.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        dataset_root: str | Path | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)

        if dataset_root is None:
            dataset_root = DEFAULT_DATASET_ROOT

        self.dataset_root = Path(dataset_root)

        if not self.jsonl_path.exists():
            raise FileNotFoundError(
                f"RS-VQA JSONL file not found: {self.jsonl_path}"
            )

        self.examples = self._load()

        if not self.examples:
            raise ValueError(
                f"RS-VQA dataset is empty: {self.jsonl_path}"
            )

    def _resolve_image(self, image_value: str) -> Path:
        image_path = Path(image_value)

        if image_path.is_absolute():
            resolved = image_path
        else:
            resolved = PROJECT_ROOT / image_path

            if not resolved.exists():
                resolved = self.dataset_root / image_path

        return resolved

    def _load(self) -> list[RSVQAExample]:
        examples: list[RSVQAExample] = []

        with self.jsonl_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {line_number}: "
                        f"{self.jsonl_path}"
                    ) from exc

                required = [
                    "id",
                    "image",
                    "question",
                    "answer",
                    "task",
                    "source",
                    "split",
                    "ground_truth",
                ]

                missing = [
                    key
                    for key in required
                    if key not in record
                ]

                if missing:
                    raise ValueError(
                        f"Missing fields at line {line_number}: "
                        f"{missing}"
                    )

                image = self._resolve_image(
                    str(record["image"])
                )

                if not image.exists():
                    raise FileNotFoundError(
                        f"Image referenced by line {line_number} "
                        f"does not exist: {image}"
                    )

                examples.append(
                    RSVQAExample(
                        example_id=str(record["id"]),
                        image=image,
                        question=str(record["question"]),
                        answer=str(record["answer"]),
                        task=str(record["task"]),
                        source=str(record["source"]),
                        split=str(record["split"]),
                        ground_truth=dict(
                            record["ground_truth"]
                        ),
                    )
                )

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self,
        index: int,
    ) -> RSVQAExample:
        return self.examples[index]


def load_image(example: RSVQAExample) -> Image.Image:
    """
    Load one RS-VQA image as RGB.

    The source generator intentionally created RGB visualization
    images for Qwen input while retaining the original source
    remote-sensing arrays separately.
    """

    with Image.open(example.image) as image:
        return image.convert("RGB").copy()


def build_qwen_messages(
    example: RSVQAExample,
) -> list[dict[str, Any]]:
    """
    Construct the Qwen2-VL multimodal conversation.

    The answer is kept separate because the training collator
    will construct prompt/answer labels explicitly.
    """

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(example.image),
                },
                {
                    "type": "text",
                    "text": example.question,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": example.answer,
                }
            ],
        },
    ]


def build_qwen_prompt_messages(
    example: RSVQAExample,
) -> list[dict[str, Any]]:
    """
    Construct only the user-side conversation.

    Used to determine which tokens belong to the prompt and
    should therefore be masked from the language-model loss.
    """

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(example.image),
                },
                {
                    "type": "text",
                    "text": example.question,
                },
            ],
        }
    ]
