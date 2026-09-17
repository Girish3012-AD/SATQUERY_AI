import pytest

from src.planner import EvidencePlanner
from src.schemas import TaskSpec


def test_simple_vqa_plan():
    task = TaskSpec(
        task_id="TASK-001",
        query="What is shown in this image?",
        task_type="vqa",
        required_capabilities=["vqa"],
        required_modalities=[],
        input_count=1,
    )

    plan = EvidencePlanner().create_plan(task)

    assert plan.plan_id == "PLAN-TASK-001"
    assert plan.step_ids() == ["T1", "T2"]
    assert plan.get_step("T1").operation == "visual_question_answering"
    assert plan.get_step("T2").operation == "verification"
    assert plan.get_step("T2").depends_on == ["T1"]


def test_complex_flood_building_query():
    task = TaskSpec(
        task_id="TASK-001",
        query="Find newly constructed buildings within 500 meters of flooded areas.",
        task_type="temporal_analysis",
        required_capabilities=[
            "temporal_analysis",
            "flood_detection",
            "building_detection",
        ],
        required_modalities=[],
        input_count=2,
        requires_temporal_pair=True,
        spatial_operations=["buffer", "intersection"],
        parameters={"distance_m": 500},
    )

    plan = EvidencePlanner().create_plan(task)

    assert plan.step_ids() == [
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
    ]

    assert plan.get_step("T1").task == "temporal_analysis"
    assert plan.get_step("T2").task == "flood_detection"
    assert plan.get_step("T3").task == "building_detection"

    assert plan.get_step("T4").operation == "buffer"
    assert plan.get_step("T4").parameters["distance_m"] == 500

    assert plan.get_step("T5").operation == "intersection"

    assert plan.get_step("T6").operation == "verification"
    assert plan.get_step("T6").depends_on == ["T5"]


def test_sar_plan():
    task = TaskSpec(
        task_id="TASK-002",
        query="Analyze the SAR image.",
        task_type="specialized_analysis",
        required_capabilities=["sar_analysis"],
        required_modalities=["sar"],
        input_count=1,
    )

    plan = EvidencePlanner().create_plan(task)

    assert plan.step_ids() == ["T1", "T2"]
    assert plan.get_step("T1").task == "sar_analysis"
    assert plan.get_step("T2").operation == "verification"


def test_plan_step_lookup_failure():
    task = TaskSpec(
        task_id="TASK-003",
        query="What is shown?",
        task_type="vqa",
        required_capabilities=["vqa"],
        input_count=1,
    )

    plan = EvidencePlanner().create_plan(task)

    with pytest.raises(KeyError):
        plan.get_step("DOES_NOT_EXIST")


def test_optical_sar_multimodal_plan():
    task = TaskSpec(
        task_id="TASK-MM-001",
        query="Analyze the scene using optical and SAR evidence.",
        task_type="specialized_analysis",
        required_capabilities=[
            "building_detection",
            "sar_analysis",
        ],
        required_modalities=[
            "optical",
            "sar",
        ],
        input_count=2,
    )

    plan = EvidencePlanner().create_plan(task)

    assert plan.step_ids() == [
        "T1",
        "T2",
        "T3",
        "T4",
    ]

    assert plan.get_step("T1").task == "building_detection"
    assert plan.get_step("T1").operation == "specialist_inference"

    assert plan.get_step("T2").task == "sar_analysis"
    assert plan.get_step("T2").operation == "sar_analysis"

    assert plan.get_step("T3").task == "multimodal_alignment"
    assert plan.get_step("T3").operation == "multimodal_alignment"
    assert plan.get_step("T3").depends_on == [
        "T1",
        "T2",
    ]

    assert plan.get_step("T4").operation == "verification"
    assert plan.get_step("T4").depends_on == ["T3"]
