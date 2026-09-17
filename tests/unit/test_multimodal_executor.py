import rasterio
from rasterio.transform import from_origin

from src.evidence import EvidenceRegistry
from src.executor.executor import ExecutionEngine
from src.planner.evidence_planner import PlanStep
from src.schemas.evidence import Evidence


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
        import numpy as np

        dataset.write(
            np.ones(
                (10, 10),
                dtype=np.uint16,
            ),
            1,
        )


def _make_evidence(
    evidence_id,
    modality,
    sensor,
    path,
    timestamp,
):
    return Evidence(
        evidence_id=evidence_id,
        source="test",
        task=(
            "building_detection"
            if modality == "optical"
            else "sar_analysis"
        ),
        model="test-model",
        sensor=sensor,
        modality=modality,
        timestamp=timestamp,
        geometry=None,
        measurement={
            "crs": "EPSG:32643",
            "resolution": 10.0,
        },
        result={
            "image_path": str(path),
            "crs": "EPSG:32643",
            "resolution": [10.0, 10.0],
        },
        confidence=0.80,
        provenance={
            "image_path": str(path),
            "data_status": "real_validation",
        },
    )


def test_execution_engine_runs_multimodal_alignment(tmp_path):
    optical = tmp_path / "optical.tif"
    sar = tmp_path / "sar.tif"

    _create_raster(optical)
    _create_raster(sar)

    registry = EvidenceRegistry()

    registry.add(
        _make_evidence(
            "OPT-EXEC-001",
            "optical",
            "SpaceNet-4",
            optical,
            "2026-01-20T00:00:00+00:00",
        )
    )

    registry.add(
        _make_evidence(
            "SAR-EXEC-001",
            "sar",
            "Sentinel-1",
            sar,
            "2026-01-20T01:00:00+00:00",
        )
    )

    engine = ExecutionEngine(registry)

    step = PlanStep(
        step_id="T3",
        task="multimodal_alignment",
        operation="multimodal_alignment",
        depends_on=["T1", "T2"],
    )

    result = engine.execute_step(
        step,
        dependency_evidence_ids=[
            "OPT-EXEC-001",
            "SAR-EXEC-001",
        ],
    )

    assert result.success is True
    assert result.task == "multimodal_alignment"
    assert len(result.evidence_ids) == 1

    evidence = registry.get(result.evidence_ids[0])

    assert evidence.task == "multimodal_alignment"
    assert evidence.modality == "multimodal"
    assert evidence.result["status"] in {
        "compatible",
        "uncertain",
    }
    assert evidence.result["source_evidence_ids"] == [
        "OPT-EXEC-001",
        "SAR-EXEC-001",
    ]
    assert evidence.provenance["source_evidence_ids"] == [
        "OPT-EXEC-001",
        "SAR-EXEC-001",
    ]
    assert evidence.metadata["deterministic"] is True


def test_multimodal_execution_rejects_two_optical_sources(
    tmp_path,
):
    optical_a = tmp_path / "optical_a.tif"
    optical_b = tmp_path / "optical_b.tif"

    _create_raster(optical_a)
    _create_raster(optical_b)

    registry = EvidenceRegistry()

    registry.add(
        _make_evidence(
            "OPT-A",
            "optical",
            "SpaceNet-4",
            optical_a,
            "2026-01-20T00:00:00+00:00",
        )
    )

    registry.add(
        _make_evidence(
            "OPT-B",
            "optical",
            "SpaceNet-4",
            optical_b,
            "2026-01-20T01:00:00+00:00",
        )
    )

    engine = ExecutionEngine(registry)

    step = PlanStep(
        step_id="T3",
        task="multimodal_alignment",
        operation="multimodal_alignment",
        depends_on=["T1", "T2"],
    )

    result = engine.execute_step(
        step,
        dependency_evidence_ids=[
            "OPT-A",
            "OPT-B",
        ],
    )

    assert result.success is False
    assert result.evidence_ids == []
    assert "exactly one optical" in result.message
