import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.executor.multimodal_flood_specialist import MultimodalFloodSpecialist
from src.evidence.registry import EvidenceRegistry
from src.schemas.evidence import Evidence
from src.verifier.georeason_verifier import GeoReasonVerifier

@pytest.fixture
def evidence_registry():
    return EvidenceRegistry()

@pytest.fixture
def mock_evidence():
    opt_ev = Evidence(
        evidence_id="opt_1", source="FloodSpecialist", task="flood_analysis",
        model="opt_model", sensor="Sentinel-2", modality="optical", timestamp="2024-10-15T00:00:00Z",
        geometry=None, measurement={}, result={}, confidence=0.8,
        provenance={"source_path": "dummy_opt.tif"}, metadata={}
    )
    sar_ev = Evidence(
        evidence_id="sar_1", source="SARSpecialist", task="sar_analysis",
        model="sar_model", sensor="Sentinel-1", modality="sar", timestamp="2024-10-15T00:00:00Z",
        geometry=None, measurement={}, result={}, confidence=0.8,
        provenance={"image_path": "dummy_sar.tif"}, metadata={}
    )
    return opt_ev, sar_ev

def test_both_modalities_required(evidence_registry, mock_evidence):
    opt_ev, _ = mock_evidence
    evidence_registry.add(opt_ev)
    spec = MultimodalFloodSpecialist(evidence_registry)
    with pytest.raises(ValueError, match="exactly two evidence IDs"):
        spec.infer([opt_ev.evidence_id])

def test_spatial_compatibility_and_crs_mismatch_handling(evidence_registry, mock_evidence):
    opt_ev, sar_ev = mock_evidence
    evidence_registry.add(opt_ev)
    evidence_registry.add(sar_ev)
    spec = MultimodalFloodSpecialist(evidence_registry)
    
    with patch("src.executor.multimodal_flood_specialist.validate_optical_sar_evidence_compatibility") as mock_compat:
        mock_compat.return_value = {"status": "incompatible", "spatial": {"crs_compatible": False}}
        ev = spec.infer([opt_ev.evidence_id, sar_ev.evidence_id])
        assert ev.result["status"] == "abstained"
        assert "compatible" in ev.result["reason"]

def test_verification_rejection_when_modality_missing():
    verifier = GeoReasonVerifier(minimum_confidence=0.60)
    opt_ev = Evidence(
        evidence_id="opt_1", source="FloodSpecialist", task="multimodal_flood_analysis",
        model="opt_model", sensor="Sentinel-2", modality="optical", timestamp="2024-10-15T00:00:00Z",
        geometry=None, measurement={}, result={}, confidence=0.8, provenance={}, metadata={}
    )
    res = verifier.verify([opt_ev], expected_task="multimodal_flood_analysis", required_modalities=["optical", "sar", "optical_sar"])
    assert res.status == "abstain"
    assert any("Missing required modalities" in r for r in res.reasons)

def test_verification_rejection_multimodal_provenance():
    verifier = GeoReasonVerifier(minimum_confidence=0.60)
    opt_sar_ev = Evidence(
        evidence_id="fuse_1", source="MultimodalFloodSpecialist", task="multimodal_flood_analysis",
        model="fuse_model", sensor="optical_sar", modality="optical_sar", timestamp="2024-10-15T00:00:00Z",
        geometry=None, measurement={}, result={}, confidence=0.8, 
        provenance={"optical_evidence_id": "opt_1"}, metadata={} # Missing sar_evidence_id
    )
    res = verifier.verify([opt_sar_ev], expected_task="multimodal_flood_analysis", required_modalities=["optical_sar"])
    assert res.status == "abstain"
    assert any("lacks provenance from both modalities" in r for r in res.reasons)
