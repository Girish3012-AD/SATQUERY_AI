from __future__ import annotations

import re
from typing import Any


_PERCENTAGE_RE = re.compile(
    r"(?<![A-Za-z])([0-9]+(?:\.[0-9]+)?)\s*%"
)

_COVERAGE_CATEGORIES = sorted(
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


def extract_percentages(text: str | None) -> list[float]:
    """Extract explicit percentage values from a VQA answer."""
    return [
        float(value)
        for value in _PERCENTAGE_RE.findall(text or "")
    ]


def evaluate_vqa_consistency(
    *,
    answer: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluate an evidence-conditioned VQA answer against structured
    remote-sensing evidence.

    The evaluator is deterministic and does not generate observations.
    It only checks whether claims explicitly present in the answer
    agree with supplied structured evidence.

    Expected evidence fields may include:
        building_percentage
        density
        coverage
        scene

    Missing fields produce None rather than inventing a result.
    """

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")

    if evidence is None:
        return {
            "numeric_match": None,
            "density_match": None,
            "coverage_match": None,
            "scene_match": None,
            "evaluated": False,
            "reason": "No structured remote-sensing evidence was supplied.",
        }

    if not isinstance(evidence, dict):
        raise TypeError("evidence must be a dictionary or None")

    answer_lower = answer.lower()

    expected_percentage = evidence.get("building_percentage")
    percentages = extract_percentages(answer)

    if expected_percentage is not None:
        expected_percentage = float(expected_percentage)

        numeric_match = any(
            abs(value - expected_percentage) <= 0.02
            for value in percentages
        )
    else:
        numeric_match = None

    expected_density = evidence.get("density")

    if expected_density is not None:
        if not isinstance(expected_density, str):
            raise TypeError("evidence density must be a string or None")

        density_match = (
            expected_density.lower() in answer_lower
        )
    else:
        density_match = None

    expected_scene = evidence.get("scene")

    if expected_scene is not None:
        if not isinstance(expected_scene, str):
            raise TypeError("evidence scene must be a string or None")

        scene_match = (
            expected_scene.lower() in answer_lower
        )
    else:
        scene_match = None

    expected_coverage = evidence.get("coverage")

    if expected_coverage is not None:
        if not isinstance(expected_coverage, str):
            raise TypeError("evidence coverage must be a string or None")

        mentioned_categories = [
            category
            for category in _COVERAGE_CATEGORIES
            if category in answer_lower
        ]

        if mentioned_categories:
            actual_category = mentioned_categories[0]
            coverage_match = (
                actual_category == expected_coverage.lower()
            )
        elif numeric_match is True:
            coverage_match = True
        else:
            coverage_match = False
    else:
        coverage_match = None

    evaluated_fields = {
        "numeric_match": numeric_match,
        "density_match": density_match,
        "coverage_match": coverage_match,
        "scene_match": scene_match,
    }

    evaluated_count = sum(
        value is not None
        for value in evaluated_fields.values()
    )

    return {
        **evaluated_fields,
        "evaluated": evaluated_count > 0,
        "evaluated_count": evaluated_count,
        "answer_percentages": percentages,
    }
