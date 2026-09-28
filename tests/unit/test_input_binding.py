"""
Tests for Input Binding and Asset Router capabilities.
Verifies heterogeneous workflows properly map typed assets to plan steps.
"""
from __future__ import annotations

import pytest
from src.data.input_metadata import InputMetadataResolver
from src.orchestration.orchestrator import SATQueryOrchestrator
from src.planner.evidence_planner import PlanStep, EvidencePlan
from src.executor.specialist import Specialist
from src.schemas import Evidence

class DummyOpticalSpecialist(Specialist):
    REQUIRED_INPUT_PROFILE = {"modality": "optical", "count": 1}
    @property
    def capability(self): return "dummy_optical"
    def infer(self, inputs, parameters=None):
        return None

class DummySARSpecialist(Specialist):
    REQUIRED_INPUT_PROFILE = {"modality": "sar", "count": 1}
    @property
    def capability(self): return "dummy_sar"
    def infer(self, inputs, parameters=None):
        return None

class DummyBiTemporalSpecialist(Specialist):
    REQUIRED_INPUT_PROFILE = {"modality": "optical", "temporal": "bi-temporal", "count": 2}
    @property
    def capability(self): return "dummy_bitemporal"
    def infer(self, inputs, parameters=None):
        return None

class DummyLegacySpecialist(Specialist):
    @property
    def capability(self): return "dummy_legacy"
    def infer(self, inputs, parameters=None):
        return None

def test_asset_classifier_optical():
    resolver = InputMetadataResolver()
    assets = resolver.classify_assets(["some_S2_image.tif", "random_B03_file.tif"])
    assert assets[0].modality == "optical"
    assert assets[0].sensor == "Sentinel-2"
    assert assets[1].modality == "optical"
    assert "B03" in assets[1].bands

def test_asset_classifier_sar():
    resolver = InputMetadataResolver()
    assets = resolver.classify_assets(["S1_GRD_image.tif", "unknown_SAR_file.tif"])
    assert assets[0].modality == "sar"
    assert assets[0].sensor == "Sentinel-1"
    assert assets[1].modality == "sar"

def test_temporal_tagging():
    resolver = InputMetadataResolver()
    assets = resolver.classify_assets(["pre_T1_image.tif", "post_T2_image.tif"])
    assert assets[0].temporal_tag == "T1"
    assert assets[1].temporal_tag == "T2"

def test_input_binding_map_generation():
    orchestrator = SATQueryOrchestrator()
    orchestrator._specialists = {
        "dummy_optical": DummyOpticalSpecialist(),
        "dummy_sar": DummySARSpecialist(),
    }
    plan = EvidencePlan(
        plan_id="plan1",
        task_id="task1",
        query="test",
        steps=[
            PlanStep(step_id="s1", task="dummy_optical", operation="specialist_inference"),
            PlanStep(step_id="s2", task="dummy_sar", operation="specialist_inference")
        ]
    )
    classified = orchestrator.input_metadata_resolver.classify_assets([
        "S1_image.tif",
        "S2_image.tif"
    ])
    bindings = orchestrator._bind_inputs_to_plan(plan, classified, {})
    
    assert bindings["s1"] == ["S2_image.tif"]
    assert bindings["s2"] == ["S1_image.tif"]

def test_temporal_ordering_is_preserved():
    orchestrator = SATQueryOrchestrator()
    orchestrator._specialists = {
        "dummy_bitemporal": DummyBiTemporalSpecialist(),
    }
    plan = EvidencePlan(
        plan_id="plan1",
        task_id="task1",
        query="test",
        steps=[
            PlanStep(step_id="s1", task="dummy_bitemporal", operation="specialist_inference"),
        ]
    )
    # Give it out of order
    classified = orchestrator.input_metadata_resolver.classify_assets([
        "S2_T2.tif",
        "S2_T1.tif"
    ])
    bindings = orchestrator._bind_inputs_to_plan(plan, classified, {})
    
    # Must be T1 then T2
    assert bindings["s1"] == ["S2_T1.tif", "S2_T2.tif"]

def test_missing_required_asset_produces_error():
    orchestrator = SATQueryOrchestrator()
    orchestrator._specialists = {
        "dummy_sar": DummySARSpecialist(),
    }
    plan = EvidencePlan(
        plan_id="plan1",
        task_id="task1",
        query="test",
        steps=[
            PlanStep(step_id="s1", task="dummy_sar", operation="specialist_inference"),
        ]
    )
    classified = orchestrator.input_metadata_resolver.classify_assets(["S2_image.tif"])
    bindings = orchestrator._bind_inputs_to_plan(plan, classified, {})
    
    # Must produce a dict with "error" instead of a list
    assert isinstance(bindings["s1"], dict)
    assert "error" in bindings["s1"]
    assert "Missing required assets" in bindings["s1"]["error"]

def test_legacy_specialist_receives_all_inputs():
    orchestrator = SATQueryOrchestrator()
    orchestrator._specialists = {
        "dummy_legacy": DummyLegacySpecialist(),
    }
    plan = EvidencePlan(
        plan_id="plan1",
        task_id="task1",
        query="test",
        steps=[
            PlanStep(step_id="s1", task="dummy_legacy", operation="specialist_inference"),
        ]
    )
    classified = orchestrator.input_metadata_resolver.classify_assets(["S2_image.tif", "S1_image.tif"])
    bindings = orchestrator._bind_inputs_to_plan(plan, classified, {})
    
    assert bindings["s1"] == ["S2_image.tif", "S1_image.tif"]

def test_execution_engine_uses_bindings():
    orchestrator = SATQueryOrchestrator()
    orchestrator._specialists = {
        "dummy_optical": DummyOpticalSpecialist(),
        "dummy_sar": DummySARSpecialist(),
    }
    plan = EvidencePlan(
        plan_id="plan1",
        task_id="task1",
        query="test",
        steps=[
            PlanStep(step_id="s1", task="dummy_optical", operation="specialist_inference"),
            PlanStep(step_id="s2", task="dummy_sar", operation="specialist_inference")
        ]
    )
    inputs = ["S1_image.tif", "S2_image.tif"]
    classified = orchestrator.input_metadata_resolver.classify_assets(inputs)
    bindings = orchestrator._bind_inputs_to_plan(plan, classified, {})
    
    # We patch execute_step to see what it received
    received_inputs = {}
    original_execute_step = orchestrator.engine.execute_step
    
    def mock_execute_step(step, step_inputs, *args, **kwargs):
        received_inputs[step.step_id] = step_inputs
        from src.executor.execution_result import ExecutionResult
        return ExecutionResult(success=True, step_id=step.step_id, task=step.task, message="ok")
    
    import unittest.mock
    with unittest.mock.patch.object(orchestrator.engine, "execute_step", side_effect=mock_execute_step):
        orchestrator.engine.execute(plan, inputs=inputs, input_bindings=bindings)
    
    assert received_inputs["s1"] == ["S2_image.tif"]
    assert received_inputs["s2"] == ["S1_image.tif"]

def test_execution_engine_fails_on_blocked_binding():
    orchestrator = SATQueryOrchestrator()
    plan = EvidencePlan(
        plan_id="plan1",
        task_id="task1",
        query="test",
        steps=[
            PlanStep(step_id="s1", task="dummy_sar", operation="specialist_inference"),
        ]
    )
    inputs = ["S2_image.tif"]
    # Provide a pre-blocked binding
    bindings = {
        "s1": {"error": "Missing required assets", "reason": "No SAR found."}
    }
    results = orchestrator.engine.execute(plan, inputs=inputs, input_bindings=bindings)
    
    assert len(results) == 1
    assert results[0].success is False
    assert "BLOCKED" in results[0].message
    assert "Missing required assets" in results[0].message
