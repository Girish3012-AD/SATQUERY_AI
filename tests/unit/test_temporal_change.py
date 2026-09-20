import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.executor.change_specialist import ChangeSpecialist
from src.schemas.evidence import Evidence
from src.verifier.georeason_verifier import GeoReasonVerifier

def test_change_specialist_input_validation():
    spec = ChangeSpecialist()
    with pytest.raises(ValueError, match="exactly two raster inputs"):
        spec.infer(["one_input"])
    
    with pytest.raises(FileNotFoundError, match="Before raster does not exist"):
        spec.infer(["nonexistent_t1.tif", "nonexistent_t2.tif"])

def test_change_specialist_metadata_injection(tmp_path):
    # create dummy npy files
    import numpy as np
    t1_path = tmp_path / "t1.npy"
    t2_path = tmp_path / "t2.npy"
    t1_data = np.zeros((1, 10, 10), dtype=np.float32)
    t1_data[0, :5, :] = 1.0
    t2_data = np.zeros((1, 10, 10), dtype=np.float32)
    t2_data[0, 5:, :] = 1.0
    np.save(t1_path, t1_data)
    np.save(t2_path, t2_data)

    spec = ChangeSpecialist(default_threshold=0.5)
    ev = spec.infer(
        [str(t1_path), str(t2_path)],
        parameters={"t1_timestamp": "2023-01-01", "t2_timestamp": "2024-01-01"}
    )
    
    assert ev.t1_timestamp == "2023-01-01"
    assert ev.t2_timestamp == "2024-01-01"
    assert ev.result["changed"] is True
    assert ev.result["change_percentage"] == 100.0

def test_verifier_rejects_missing_temporal_metadata():
    ev = Evidence(
        evidence_id="change_1", source="ChangeSpecialist", task="temporal_analysis",
        model="deterministic", sensor="optical", modality="optical", timestamp="2024-01-01",
        geometry={"type": "Polygon", "coordinates": []}, measurement={}, result={}, confidence=0.8,
        provenance={}, metadata={}
    )
    # Missing t1_timestamp and t2_timestamp
    verifier = GeoReasonVerifier(minimum_confidence=0.6)
    res = verifier.verify([ev], expected_task="temporal_analysis")
    assert res.status == "abstain"
    assert any("missing T1 or T2" in r for r in res.reasons)

def test_verifier_rejects_missing_geometry():
    ev = Evidence(
        evidence_id="change_2", source="ChangeSpecialist", task="temporal_analysis",
        model="deterministic", sensor="optical", modality="optical", timestamp="2024-01-01",
        geometry=None, # Missing geometry
        t1_timestamp="2023-01-01", t2_timestamp="2024-01-01",
        measurement={}, result={}, confidence=0.8,
        provenance={}, metadata={}
    )
    verifier = GeoReasonVerifier(minimum_confidence=0.6)
    res = verifier.verify([ev], expected_task="temporal_analysis")
    assert res.status == "abstain"
    assert any("lacks valid geometry" in r for r in res.reasons)
