from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

from src.training.rs_vqa_lora_data import (
    RSVQAExample,
    build_qwen_messages,
    build_qwen_prompt_messages,
)


DEFAULT_QWEN_SNAPSHOT = str(
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct"
    / "snapshots"
    / "895c3a49bc3fa70a340399125c650a463535e71c"
)


@dataclass
class RSVQABatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    labels: torch.Tensor


class RSVQACollator:
    """
    Qwen2-VL processor wrapper for supervised RS-VQA.

    The collator builds the complete conversation for the model,
    while masking prompt tokens from the language-model loss.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_QWEN_SNAPSHOT,
        max_length: int = 512,
    ) -> None:
        self.model_path = model_path
        self.max_length = max_length

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Qwen processor path does not exist: {model_path}"
            )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
        )

    def _render(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _render_prompt(
        self,
        example: RSVQAExample,
    ) -> str:
        messages = build_qwen_prompt_messages(example)

        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _tokenize(
        self,
        example: RSVQAExample,
    ) -> dict[str, torch.Tensor]:
        messages = build_qwen_messages(example)

        text = self._render(messages)

        image = Image.open(example.image).convert("RGB")

        processed = self.processor(
            text=[text],
            images=[image],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            key: value
            for key, value in processed.items()
            if isinstance(value, torch.Tensor)
        }

    def _build_labels(
        self,
        example: RSVQAExample,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mask the user/prompt portion and retain only answer tokens.

        We determine the prompt length using the same tokenizer and
        chat template used to construct the actual training example.
        """

        prompt_text = self._render_prompt(example)

        prompt_tokens = self.processor.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]

        prompt_length = prompt_tokens.shape[1]

        labels = input_ids.clone()

        if prompt_length >= labels.shape[1]:
            raise ValueError(
                "Prompt consumes the entire sequence. "
                "Increase max_length."
            )

        labels[:, :prompt_length] = -100

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return labels

    def __call__(
        self,
        examples: list[RSVQAExample],
    ) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError(
                "RS-VQA collator received an empty batch."
            )

        if len(examples) != 1:
            raise NotImplementedError(
                "Phase 9.4D.2 supports batch_size=1 only. "
                "Dynamic multimodal padding will be added after "
                "the single-example smoke test passes."
            )

        example = examples[0]

        tensors = self._tokenize(example)

        if "input_ids" not in tensors:
            raise RuntimeError(
                "Qwen processor did not return input_ids."
            )

        labels = self._build_labels(
            example,
            tensors["input_ids"],
        )

        tensors["labels"] = labels

        return tensors
