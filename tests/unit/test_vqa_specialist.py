import pytest

from src.executor.vqa_specialist import VqaSpecialist


def test_vqa_specialist_capability():
    specialist = VqaSpecialist()

    assert specialist.capability == "vqa"


def test_vqa_missing_image_fails():
    specialist = VqaSpecialist()

    with pytest.raises(ValueError, match="at least one image"):
        specialist.infer(
            inputs=[],
            parameters={"query": "What is visible?"},
        )


def test_vqa_missing_query_fails():
    specialist = VqaSpecialist()

    with pytest.raises(ValueError, match="non-empty"):
        specialist.infer(
            inputs=["data/samples/vqa_test.png"],
            parameters={},
        )


def test_vqa_invalid_image_path_fails():
    specialist = VqaSpecialist()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        specialist.infer(
            inputs=["data/samples/does_not_exist.png"],
            parameters={"query": "What is visible?"},
        )


def test_vqa_invalid_image_file_path_fails(tmp_path):
    invalid_file = tmp_path / "not_an_image.txt"
    invalid_file.write_text("not an image")

    specialist = VqaSpecialist()

    assert invalid_file.is_file()


def test_vqa_sensor_and_modality_defaults():
    specialist = VqaSpecialist()

    assert specialist._infer_sensor({}) is None
    assert specialist._infer_modality({}) == "optical"


def test_vqa_sensor_and_modality_parameters():
    specialist = VqaSpecialist()

    assert specialist._infer_sensor(
        {"sensor": "Cartosat-2S"}
    ) == "Cartosat-2S"

    assert specialist._infer_modality(
        {"modality": "sar"}
    ) == "sar"


def test_vqa_inference_creates_evidence(monkeypatch):
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def to(self, device):
            return self

        def __iter__(self):
            return iter(self.values)

    class FakeProcessor:
        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ):
            assert messages[0]["content"][0]["type"] == "image"
            assert messages[0]["content"][1]["text"] == "What is visible?"
            return "formatted prompt"

        def __call__(
            self,
            text,
            images,
            videos,
            padding,
            return_tensors,
        ):
            return {
                "input_ids": FakeTensor(
                    [[10, 11, 12]]
                ),
                "pixel_values": FakeTensor(
                    [[1]]
                ),
            }

        def batch_decode(
            self,
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ):
            assert generated_ids == [[20, 21]]
            return [
                "A green rectangle and a blue circle."
            ]

    class FakeModel:
        device = "cpu"

        def generate(self, **model_inputs):
            assert "input_ids" in model_inputs
            assert "pixel_values" in model_inputs

            return [
                [10, 11, 12, 20, 21]
            ]

    specialist = VqaSpecialist(
        model_path="unused",
        unload_after_inference=True,
    )

    specialist._processor = FakeProcessor()
    specialist._model = FakeModel()

    monkeypatch.setattr(
        "src.executor.vqa_specialist.process_vision_info",
        lambda messages: (
            ["fake_image"],
            [],
        ),
    )

    unload_called = {"value": False}

    def fake_unload():
        unload_called["value"] = True
        specialist._model = None
        specialist._processor = None

    monkeypatch.setattr(
        specialist,
        "unload",
        fake_unload,
    )

    evidence = specialist.infer(
        inputs=["data/samples/vqa_test.png"],
        parameters={
            "query": "What is visible?",
            "sensor": "Cartosat-2S",
            "modality": "optical",
            "max_new_tokens": 64,
        },
    )

    assert evidence.task == "vqa"
    assert evidence.model == "Qwen2-VL-2B-Instruct"
    assert evidence.sensor == "Cartosat-2S"
    assert evidence.modality == "optical"

    assert evidence.result["question"] == "What is visible?"
    assert (
        evidence.result["answer"]
        == "A green rectangle and a blue circle."
    )

    assert unload_called["value"] is True


def test_vqa_evidence_ids_change_with_query():
    image = "data/samples/vqa_test.png"

    first = VqaSpecialist._make_evidence_id(
        image,
        "What objects are visible?",
    )

    second = VqaSpecialist._make_evidence_id(
        image,
        "What colors are visible?",
    )

    assert first != second
    assert first.startswith("VQA_vqa_test_")
    assert second.startswith("VQA_vqa_test_")


def test_vqa_evidence_id_is_deterministic():
    image = "data/samples/vqa_test.png"
    query = "What objects are visible?"

    first = VqaSpecialist._make_evidence_id(
        image,
        query,
    )

    second = VqaSpecialist._make_evidence_id(
        image,
        query,
    )

    assert first == second


@pytest.mark.parametrize(
    "value",
    [0, -1, 1025, "abc", None],
)
def test_vqa_invalid_max_new_tokens_fails(value):
    specialist = VqaSpecialist()

    with pytest.raises(ValueError):
        specialist._resolve_max_new_tokens(
            {"max_new_tokens": value}
        )


@pytest.mark.parametrize(
    "value",
    [1, 64, 128, 1024],
)
def test_vqa_valid_max_new_tokens(value):
    specialist = VqaSpecialist()

    assert (
        specialist._resolve_max_new_tokens(
            {"max_new_tokens": value}
        )
        == value
    )


@pytest.mark.parametrize(
    "value",
    ["true", 1, 0, None],
)
def test_vqa_invalid_unload_parameter_fails(
    monkeypatch,
    value,
):
    specialist = VqaSpecialist(
        unload_after_inference=True,
    )

    def fail_if_model_load_called():
        raise AssertionError(
            "Model should not load for invalid parameters."
        )

    monkeypatch.setattr(
        specialist,
        "_load_model",
        fail_if_model_load_called,
    )

    with pytest.raises(ValueError, match="must be boolean"):
        specialist.infer(
            inputs=["data/samples/vqa_test.png"],
            parameters={
                "query": "What is visible?",
                "unload_after_inference": value,
            },
        )
