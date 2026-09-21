import json
import re
from pathlib import Path
import pytest


RESULTS = Path(
    "outputs/vqa_evidence_conditioned_acceptance_2AY-Z-K.json"
)

pytestmark = pytest.mark.skipif(
    not RESULTS.exists(),
    reason=f"DATASET_REQUIRED: VQA benchmark outputs not found at {RESULTS}",
)


def extract_percentages(text):
    return [
        float(value)
        for value in re.findall(
            r"(?<![A-Za-z])([0-9]+(?:\.[0-9]+)?)\s*%",
            text or "",
        )
    ]


def evaluate_record(record):
    ground_truth = record["ground_truth"]
    answer = record["answer"]
    task = record.get("task", "")

    answer_lower = answer.lower()

    expected_percentage = ground_truth.get(
        "building_percentage"
    )

    percentages = extract_percentages(answer)

    if expected_percentage is not None:
        expected_percentage = float(expected_percentage)

        numeric_match = any(
            abs(value - expected_percentage) <= 0.02
            for value in percentages
        )
    else:
        numeric_match = None

    expected_density = ground_truth.get("density")
    expected_coverage = ground_truth.get("coverage")
    expected_scene = ground_truth.get("scene")

    if expected_density is not None:
        density_match = (
            expected_density.lower() in answer_lower
        )
    else:
        density_match = None

    if expected_scene is not None:
        scene_match = (
            expected_scene.lower() in answer_lower
        )
    else:
        scene_match = None

    coverage_categories = sorted(
        [
            "very high building coverage",
            "high building coverage",
            "medium building coverage",
            "moderate building coverage",
            "very low building coverage",
            "low building coverage",
        ],
        key=len,
        reverse=True,
    )

    mentioned_coverage_categories = [
        category
        for category in coverage_categories
        if category in answer_lower
    ]

    if expected_coverage is not None:
        if mentioned_coverage_categories:
            actual_category = (
                mentioned_coverage_categories[0]
            )

            coverage_match = (
                actual_category
                == expected_coverage.lower()
            )
        elif numeric_match is True:
            coverage_match = True
        else:
            coverage_match = False
    else:
        coverage_match = None

    return {
        "id": record["id"],
        "task": task,
        "numeric_match": numeric_match,
        "density_match": density_match,
        "coverage_match": coverage_match,
        "scene_match": scene_match,
        "evidence_conditioned": (
            record.get("evidence_conditioned") is True
        ),
        "adapter_loaded": (
            record.get("adapter_loaded") is True
        ),
        "expected_percentage": expected_percentage,
        "answer_percentages": percentages,
        "expected_density": expected_density,
        "expected_coverage": expected_coverage,
        "expected_scene": expected_scene,
        "answer": answer,
    }


def test_evaluator_detects_known_coverage_mismatch():
    record = {
        "id": "test",
        "task": "grounded_building_coverage",
        "ground_truth": {
            "building_percentage": 0.29296875,
            "coverage": "low building coverage",
            "density": "very low building density",
            "scene": "mostly non-built-up",
        },
        "answer": (
            "The building coverage is very low building coverage, "
            "with 0.29% of the patch classified as building pixels."
        ),
        "evidence_conditioned": True,
        "adapter_loaded": True,
    }

    result = evaluate_record(record)

    assert result["numeric_match"] is True
    assert result["coverage_match"] is False


def test_evaluator_accepts_exact_grounded_answer():
    record = {
        "id": "test",
        "task": "grounded_building_summary",
        "ground_truth": {
            "building_percentage": 22.65625,
            "coverage": "high building coverage",
            "density": "high building density",
            "scene": "mostly built-up",
        },
        "answer": (
            "The patch has 22.66% building coverage, corresponding "
            "to high building density and a mostly built-up scene."
        ),
        "evidence_conditioned": True,
        "adapter_loaded": True,
    }

    result = evaluate_record(record)

    assert result["numeric_match"] is True
    assert result["density_match"] is True
    assert result["coverage_match"] is True
    assert result["scene_match"] is True


def test_all_saved_results_have_required_grounding_flags():
    results = json.loads(RESULTS.read_text())

    assert len(results) == 8

    for record in results:
        assert record["evidence_conditioned"] is True
        assert record["evidence_provided"] is True
        assert record["adapter_loaded"] is True
        assert record["adapter_type"] == "PEFT_LORA"


def test_saved_results_contain_answers():
    results = json.loads(RESULTS.read_text())

    assert len(results) == 8

    for record in results:
        assert isinstance(record["answer"], str)
        assert record["answer"].strip()


def test_saved_results_cover_expected_tasks():
    results = json.loads(RESULTS.read_text())

    tasks = [record["task"] for record in results]

    assert tasks.count("grounded_building_summary") == 2
    assert tasks.count("grounded_building_density") == 2
    assert tasks.count("grounded_building_coverage") == 2
    assert tasks.count("grounded_scene_description") == 2
