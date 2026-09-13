import gc
import hashlib
import os
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

from src.schemas import Evidence

from .specialist import Specialist


DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/"
    "snapshots/895c3a49bc3fa70a340399125c650a463535e71c"
)


class VqaSpecialist(Specialist):
    MODEL_NAME = "Qwen2-VL-2B-Instruct"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        unload_after_inference: bool = True,
    ) -> None:
        self.model_path = model_path
        self.unload_after_inference = unload_after_inference
        self._model: Qwen2VLForConditionalGeneration | None = None
        self._processor: Any | None = None

    @property
    def capability(self) -> str:
        return "vqa"

    def _load_model(self) -> None:
        if (
            self._model is not None
            and self._processor is not None
        ):
            return

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"VQA model path does not exist: {self.model_path}"
            )

        self._processor = AutoProcessor.from_pretrained(
            self.model_path
        )

        self._model = (
            Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        )

    def unload(self) -> None:
        self._model = None
        self._processor = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _resolve_query(
        parameters: dict[str, Any],
    ) -> str:
        query = parameters.get("query")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "VQA requires a non-empty 'query' parameter."
            )

        return query.strip()

    @staticmethod
    def _resolve_image(inputs: list[str]) -> str:
        if not inputs:
            raise ValueError(
                "VQA requires at least one image input."
            )

        image_path = inputs[0]

        if (
            not isinstance(image_path, str)
            or not image_path.strip()
        ):
            raise ValueError(
                "VQA image input must be a non-empty path."
            )

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"VQA image does not exist: {image_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"VQA image path is not a file: {image_path}"
            )

        return str(path)

    @staticmethod
    def _infer_sensor(
        parameters: dict[str, Any],
    ) -> str | None:
        sensor = parameters.get("sensor")

        if isinstance(sensor, str) and sensor.strip():
            return sensor.strip()

        return None

    @staticmethod
    def _infer_modality(
        parameters: dict[str, Any],
    ) -> str:
        modality = parameters.get("modality")

        if isinstance(modality, str) and modality.strip():
            return modality.strip()

        return "optical"

    @staticmethod
    def _resolve_max_new_tokens(
        parameters: dict[str, Any],
    ) -> int:
        value = parameters.get(
            "max_new_tokens",
            128,
        )

        try:
            max_new_tokens = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "VQA 'max_new_tokens' must be an integer."
            ) from exc

        if not 1 <= max_new_tokens <= 1024:
            raise ValueError(
                "VQA 'max_new_tokens' must be between 1 and 1024."
            )

        return max_new_tokens

    @staticmethod
    def _make_evidence_id(
        image_path: str,
        query: str,
    ) -> str:
        image_stem = Path(image_path).stem

        query_hash = hashlib.sha256(
            query.encode("utf-8")
        ).hexdigest()[:12]

        return f"VQA_{image_stem}_{query_hash}"

    def infer(
        self,
        inputs: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> Evidence:
        """
        Perform genuine image-conditioned VQA
        and return normalized Evidence.
        """

        parameters = parameters or {}

        image_path = self._resolve_image(inputs)
        query = self._resolve_query(parameters)
        max_new_tokens = self._resolve_max_new_tokens(
            parameters
        )

        unload = parameters.get(
            "unload_after_inference",
            self.unload_after_inference,
        )

        if not isinstance(unload, bool):
            raise ValueError(
                "VQA 'unload_after_inference' must be boolean."
            )

        self._load_model()

        try:
            assert self._model is not None
            assert self._processor is not None

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": os.path.abspath(
                                image_path
                            ),
                        },
                        {
                            "type": "text",
                            "text": query,
                        },
                    ],
                }
            ]

            text = (
                self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            image_inputs, video_inputs = (
                process_vision_info(messages)
            )

            model_inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            model_device = self._model.device

            model_inputs = {
                key: value.to(model_device)
                if hasattr(value, "to")
                else value
                for key, value in model_inputs.items()
            }

            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                )

            generated_ids_trimmed = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(
                    model_inputs["input_ids"],
                    generated_ids,
                )
            ]

            answers = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            answer = (
                answers[0].strip()
                if answers
                else ""
            )

            if not answer:
                raise RuntimeError(
                    "VQA model produced an empty answer."
                )

            evidence = Evidence(
                evidence_id=self._make_evidence_id(
                    image_path,
                    query,
                ),
                source=image_path,
                task="vqa",
                model=self.MODEL_NAME,
                sensor=self._infer_sensor(parameters),
                modality=self._infer_modality(parameters),
                result={
                    "question": query,
                    "answer": answer,
                },
                confidence=0.5,
                provenance={
                    "inference_type": (
                        "image_conditioned_vqa"
                    ),
                    "model_path": self.model_path,
                    "device": str(
                        self._model.device
                    ),
                    "confidence_method": (
                        "uncalibrated_default; "
                        "not a model probability"
                    ),
                },
                metadata={
                    "max_new_tokens": max_new_tokens,
                    "image_path": os.path.abspath(
                        image_path
                    ),
                },
            )

            return evidence

        finally:
            if unload:
                self.unload()
