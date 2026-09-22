import gc
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from src.schemas import Evidence

from .specialist import Specialist


DEFAULT_MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

DEFAULT_ADAPTER_PATH = os.path.abspath(
    "outputs/checkpoints/"
    "qwen2vl_rs_vqa_evidence_grounded_dev"
)


class VqaSpecialist(Specialist):
    CAPABILITY = "vqa"
    MODEL_NAME = "Qwen2-VL-2B-Instruct"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        adapter_path: str | None = None,
        unload_after_inference: bool = True,
    ) -> None:
        self.model_path = model_path
        self.adapter_path = (
            os.path.abspath(adapter_path)
            if adapter_path is not None
            else None
        )
        self.unload_after_inference = unload_after_inference
        self._model: Qwen2VLForConditionalGeneration | PeftModel | None = None
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

        self._processor = AutoProcessor.from_pretrained(
            self.model_path
        )

        base_model = (
            Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        )

        if self.adapter_path is not None:
            adapter = Path(self.adapter_path)

            if not adapter.exists():
                raise FileNotFoundError(
                    f"VQA adapter path does not exist: "
                    f"{self.adapter_path}"
                )

            if not (adapter / "adapter_config.json").exists():
                raise FileNotFoundError(
                    f"VQA adapter configuration does not exist: "
                    f"{adapter / 'adapter_config.json'}"
                )

            if not (
                adapter / "adapter_model.safetensors"
            ).exists():
                raise FileNotFoundError(
                    f"VQA adapter weights do not exist: "
                    f"{adapter / 'adapter_model.safetensors'}"
                )

            self._model = PeftModel.from_pretrained(
                base_model,
                self.adapter_path,
            )
        else:
            self._model = base_model

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
    def _prepare_model_image(
        image_path: str,
    ) -> tuple[str, str | None]:
        """
        Prepare an image path for Qwen2-VL.

        Standard raster images are passed through unchanged.

        Multi-band GeoTIFF inputs are read with Rasterio and converted
        to a temporary uint8 RGB PNG for the VLM. The original source
        path is preserved separately for Evidence provenance.

        Returns:
            (model_image_path, temporary_path)
        """
        path = Path(image_path)

        suffix = path.suffix.lower()

        if suffix == ".npy":
            array = np.load(path)

            if array.ndim != 3:
                raise ValueError(
                    "VQA NumPy image must have shape "
                    "(channels, height, width)."
                )

            channels, height, width = array.shape

            if channels < 3:
                raise ValueError(
                    "VQA NumPy image requires at least 3 channels "
                    "for RGB conversion."
                )

            if height <= 0 or width <= 0:
                raise ValueError(
                    "VQA NumPy image dimensions must be positive."
                )

            array = array[:3].astype(np.float32)

            valid = np.isfinite(array)

            if not valid.any():
                raise ValueError(
                    "VQA NumPy image contains no finite pixels."
                )

            low = float(np.percentile(array[valid], 1))
            high = float(np.percentile(array[valid], 99))

            if not np.isfinite(low) or not np.isfinite(high):
                raise ValueError(
                    "VQA NumPy RGB normalization produced "
                    "non-finite percentiles."
                )

            if high <= low:
                raise ValueError(
                    "VQA NumPy RGB normalization range is invalid."
                )

            rgb = np.clip(
                (array - low) / (high - low) * 255.0,
                0,
                255,
            )

            rgb = np.transpose(
                rgb.astype(np.uint8),
                (1, 2, 0),
            )

            from PIL import Image

            image = Image.fromarray(rgb, mode="RGB")

            temporary = tempfile.NamedTemporaryFile(
                suffix=".png",
                prefix="satquery_vqa_",
                delete=False,
            )

            temporary_path = temporary.name
            temporary.close()

            image.save(temporary_path, format="PNG")

            return temporary_path, temporary_path

        if suffix not in {".tif", ".tiff"}:
            return str(path), None

        with rasterio.open(path) as src:
            if src.count < 3:
                raise ValueError(
                    "VQA GeoTIFF requires at least 3 bands "
                    "for RGB conversion."
                )

            rgb = src.read([3, 2, 1]).astype(np.float32)

        valid = np.isfinite(rgb)

        if not valid.any():
            raise ValueError(
                "VQA GeoTIFF contains no finite RGB pixels."
            )

        low = float(np.percentile(rgb[valid], 1))
        high = float(np.percentile(rgb[valid], 99))

        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError(
                "VQA GeoTIFF RGB normalization produced "
                "non-finite percentiles."
            )

        if high <= low:
            raise ValueError(
                "VQA GeoTIFF RGB normalization range is invalid."
            )

        rgb = np.clip(
            (rgb - low) / (high - low) * 255.0,
            0,
            255,
        )

        rgb = np.transpose(
            rgb.astype(np.uint8),
            (1, 2, 0),
        )

        from PIL import Image

        image = Image.fromarray(rgb, mode="RGB")

        fd, temporary_path = tempfile.mkstemp(
            prefix="satquery_vqa_",
            suffix=".png",
        )

        os.close(fd)

        image.save(
            temporary_path,
            format="PNG",
        )

        return temporary_path, temporary_path

    @staticmethod
    def _resolve_evidence(
        parameters: dict[str, Any],
    ) -> str | None:
        """Resolve optional evidence supplied to the VQA specialist."""
        evidence = parameters.get("evidence")

        if evidence is None:
            return None

        if not isinstance(evidence, str):
            raise ValueError(
                "VQA 'evidence' must be a string when provided."
            )

        evidence = evidence.strip()

        if not evidence:
            return None

        return evidence

    @staticmethod
    def _build_user_prompt(
        query: str,
        evidence: str | None = None,
    ) -> str:
        """Build the VQA prompt with optional structured evidence."""
        if not isinstance(query, str):
            raise ValueError(
                "VQA query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "VQA query must be a non-empty string."
            )

        if evidence is None:
            return query

        return (
            "REMOTE-SENSING EVIDENCE\n"
            "-----------------------\n"
            f"{evidence}\n"
            "END REMOTE-SENSING EVIDENCE\n\n"
            "USER QUESTION\n"
            "-------------\n"
            f"{query}\n\n"
            "Use the supplied remote-sensing evidence for "
            "measured or derived facts. Use the image for "
            "visual context. Do not invent measurements that "
            "are not supported by the supplied evidence."
        )

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

        evidence_text = self._resolve_evidence(parameters)

        structured_evidence = parameters.get(
            "vqa_structured_evidence"
        )

        if structured_evidence is not None:
            if not isinstance(structured_evidence, dict):
                raise TypeError(
                    "vqa_structured_evidence must be a dictionary."
                )

            # Copy so the Evidence record owns an immutable snapshot
            # of the caller-provided structured evidence.
            structured_evidence = dict(structured_evidence)

        user_prompt = self._build_user_prompt(
            query=query,
            evidence=evidence_text,
        )
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

            model_image_path, temporary_image_path = (
                self._prepare_model_image(image_path)
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": os.path.abspath(
                                model_image_path
                            ),
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
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
                    "vqa_structured_evidence": structured_evidence,
                },
                confidence=0.85,
                provenance={
                    "inference_type": (
                        "image_conditioned_vqa"
                    ),
                    "evidence_conditioned": (
                        evidence_text is not None
                    ),
                    "evidence_provided": (
                        evidence_text is not None
                    ),
                    "model_path": self.model_path,
                    "original_image_path": os.path.abspath(
                        image_path
                    ),
                    "model_input_path": os.path.abspath(
                        model_image_path
                    ),
                    "image_input_conversion": (
                        "npy_to_rgb_png"
                        if Path(image_path).suffix.lower() == ".npy"
                        and temporary_image_path is not None
                        else (
                            "geotiff_to_rgb_png"
                            if Path(image_path).suffix.lower()
                            in {".tif", ".tiff"}
                            and temporary_image_path is not None
                            else "none"
                        )
                    ),
                    "adapter_path": self.adapter_path,
                    "adapter_loaded": (
                        self.adapter_path is not None
                    ),
                    "adapter_type": (
                        "PEFT_LORA"
                        if self.adapter_path is not None
                        else None
                    ),
                    "remote_sensing_adapted": (
                        self.adapter_path is not None
                    ),
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
            if "temporary_image_path" in locals():
                if temporary_image_path is not None:
                    try:
                        Path(temporary_image_path).unlink(
                            missing_ok=True
                        )
                    except OSError:
                        pass

            if unload:
                self.unload()
