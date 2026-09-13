from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from src.training.rs_vqa_evidence_grounded_data import (
    EvidenceGroundedRSVQAExample,
)


DEFAULT_QWEN_MODEL = (
    "/home/lenovo/.cache/huggingface/hub/"
    "models--Qwen--Qwen2-VL-2B-Instruct/"
    "snapshots/"
    "895c3a49bc3fa70a340399125c650a463535e71c"
)


class EvidenceGroundedQwenCollator:
    """
    Qwen2-VL supervised collator for evidence-grounded
    remote-sensing VQA.

    Input:
        RGB image
        structured remote-sensing evidence
        natural-language question
        supervised answer

    Output:
        multimodal Qwen tensors with user/prompt tokens
        masked from the loss and assistant answer tokens
        retained as training targets.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_QWEN_MODEL,
        max_length: int = 512,
    ) -> None:

        self.model_name_or_path = (
            model_name_or_path
        )

        self.max_length = max_length

        self.processor = (
            AutoProcessor.from_pretrained(
                self.model_name_or_path,
                local_files_only=True,
            )
        )

    @staticmethod
    def _build_messages(
        example: EvidenceGroundedRSVQAExample,
    ) -> list[dict[str, Any]]:

        return [
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

    def _build_prompt_messages(
        self,
        example: EvidenceGroundedRSVQAExample,
    ) -> list[dict[str, Any]]:

        return [
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

    def _tokenize_conversation(
        self,
        example: EvidenceGroundedRSVQAExample,
    ):
        messages = self._build_messages(
            example
        )

        image = (
            __import__("PIL")
            .Image
            .open(example.image)
            .convert("RGB")
        )

        prompt_text = (
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

        prompt_only_text = (
            self.processor.apply_chat_template(
                self._build_prompt_messages(
                    example
                ),
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        full_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        prompt_inputs = self.processor(
            text=[prompt_only_text],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        return (
            full_inputs,
            prompt_inputs,
        )

    def __call__(
        self,
        examples: list[
            EvidenceGroundedRSVQAExample
        ],
    ) -> dict[str, torch.Tensor]:

        if not examples:
            raise ValueError(
                "Cannot collate an empty batch."
            )

        if len(examples) != 1:
            raise ValueError(
                "EvidenceGroundedQwenCollator "
                "currently supports batch size 1."
            )

        example = examples[0]

        full_inputs, prompt_inputs = (
            self._tokenize_conversation(
                example
            )
        )

        input_ids = full_inputs[
            "input_ids"
        ]

        attention_mask = full_inputs[
            "attention_mask"
        ]

        labels = input_ids.clone()

        prompt_length = (
            prompt_inputs[
                "input_ids"
            ].shape[1]
        )

        prompt_length = min(
            prompt_length,
            labels.shape[1],
        )

        labels[
            :,
            :prompt_length,
        ] = -100

        pad_token_id = (
            self.processor.tokenizer
            .pad_token_id
        )

        if pad_token_id is not None:
            labels[
                input_ids
                == pad_token_id
            ] = -100

        batch = {
            key: value
            for key, value
            in full_inputs.items()
            if torch.is_tensor(value)
        }

        batch["labels"] = labels

        return batch
