import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.evidence import EvidenceRegistry
from src.executor.executor import ExecutionEngine
from src.executor.sar_specialist import SARSpecialist
from src.planner import EvidencePlanner
from src.schemas import TaskSpec
from src.schemas.evidence import Evidence


class FakeOpticalSpecialist:
    CAPABILITY = "building_detection"
    MODEL_NAME = "Fake_Optical_Model"

    @property
    def capability(self):
        return self.CAPABILITY

    def infer(self, inputs, parameters=None):
        image_path = inputs[0]

        return Evidence(
            evidence_id="OPT-AUTO-001",
            source="FakeOpticalSpecialist",
            task="building_detection",
            model=self.MODEL_NAME,
            sensor="SpaceNet-4",
            modality="optical",
            timestamp="2026-01-20T00:00:00+00:00",
            geometry=None,
            measurement={
                "crs": "EPSG:32643",
                "resolution": 10.0,
            },
            result={
                "image_path": image_path,
                "crs": "EPSG:32643",
                "resolution": [10.0, 10.0],
            },
            confidence=0.80,
            provenance={
                "image_path": image_path,
                "data_status": "real_validation",
            },
        )


def _create_raster(path):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="uint16",
        crs="EPSG:32643",
        transform=from_origin(
            362000,
            2065000,
            10,
            10,
        ),
    ) as dataset:
        dataset.write(
            np.ones(
                (10, 10),
                dtype=np.uint16,
            ),
            1,
        )


def test_execution_engine_automatically_resolves_multimodal_dependencies(
    tmp_path,
):
    optical = tmp_path / "optical.tif"
    sar = tmp_path / "sar.tif"

    _create_raster(optical)
    _create_raster(sar)

    registry = EvidenceRegistry()

    optical_specialist = FakeOpticalSpecialist()
    sar_specialist = SARSpecialist()

    task = TaskSpec(
        task_id="TASK-MM-AUTO",
        query="Analyze this scene using optical and SAR evidence.",
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

    assert [step.step_id for step in plan.steps] == [
        "T1",
        "T2",
        "T3",
        "T4",
    ]

    assert plan.get_step("T3").depends_on == [
        "T1",
        "T2",
    ]

    engine = ExecutionEngine(
        evidence_registry=registry,
        specialists={
            "building_detection": optical_specialist,
            "sar_analysis": sar_specialist,
        },
    )

    results = engine.execute(
        plan,
        inputs=[
            str(optical),
            str(sar),
        ],
    )

    assert len(results) == 4

    t1 = results[0]
    t2 = results[1]
    t3 = results[2]
    t4 = results[3]

    assert t1.success is True
    assert t1.step_id == "T1"
    assert len(t1.evidence_ids) == 1

    assert t2.success is True
    assert t2.step_id == "T2"
    assert len(t2.evidence_ids) == 1

    assert t3.success is True
    assert t3.step_id == "T3"
    assert len(t3.evidence_ids) == 1

    multimodal_evidence = registry.get(
        t3.evidence_ids[0]
    )

    assert multimodal_evidence.task == "multimodal_alignment"
    assert multimodal_evidence.modality == "multimodal"

    assert multimodal_evidence.result[
        "source_evidence_ids"
    ] == [
        t1.evidence_ids[0],
        t2.evidence_ids[0],
    ]

    assert t4.step_id == "T4"
    assert t4.success is True


def test_execution_engine_stops_when_multimodal_dependency_fails(
    tmp_path,
):
    optical = tmp_path / "optical.tif"
    _create_raster(optical)

    registry = EvidenceRegistry()

    optical_specialist = FakeOpticalSpecialist()

    task = TaskSpec(
        task_id="TASK-MM-FAIL",
        query="Analyze this scene using optical and SAR evidence.",
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

    engine = ExecutionEngine(
        evidence_registry=registry,
        specialists={
            "building_detection": optical_specialist,
        },
    )

    results = engine.execute(
        plan,
        inputs=[
            str(optical),
        ],
    )

    assert len(results) >= 1

    assert results[0].success is True

    if len(results) > 1:
        assert results[1].success is False

    assert not any(
        result.step_id == "T3" and result.success
        for result in results
    )
