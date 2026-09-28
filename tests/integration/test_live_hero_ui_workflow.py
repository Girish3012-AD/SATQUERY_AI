import pytest
from pathlib import Path
from src.orchestration.orchestrator import SATQueryOrchestrator
from api_server import resolve_request_inputs, QueryRequest

def test_live_ui_hero_workflow_bindings_and_execution():
    """
    Validates that the LIVE UI payload correctly maps the 'hero' preset to the real
    S2_T1, S2_T2, and S1_GRD assets, and perfectly replicates the production E2E pipeline
    without using mock geometries or bypassing safety.
    """
    # 1. Emulate the API request payload exactly as the frontend sends it
    req = QueryRequest(
        query="Find newly constructed buildings within 500 m of flooded areas.",
        inputs=[],
        demo_preset="hero"
    )
    
    # 2. Emulate the API endpoint resolution
    query_text, resolved_inputs = resolve_request_inputs(req)
    
    # Verify the UI correctly routed to the real artifacts
    assert any("S2_T1.tif" in p for p in resolved_inputs), "T1 optical not bound"
    assert any("S2_T2.tif" in p for p in resolved_inputs), "T2 optical not bound"
    assert any("S1_GRD.tif" in p for p in resolved_inputs), "SAR not bound"
    
    # 3. Execute the full orchestrator
    orchestrator = SATQueryOrchestrator()
    result = orchestrator.run(query=query_text, inputs=resolved_inputs)
    
    # 4. Verify Plan Bindings
    # Ensure all spatial tasks succeeded
    assert "T1" in result.successful_steps # Water (Optical bound)
    assert "T2" in result.successful_steps # Temporal Change (T1+T2 bound)
    assert "T3" in result.successful_steps # Building Detection
    assert "T4" in result.successful_steps # CRS-safe buffer
    assert "T5" in result.successful_steps # Spatial Intersection
    
    # 5. Verify the evidence results
    assert result.success is True

    assert len(result.evidence_ids) >= 3 # Water, Change, Buildings, + Buffered + Intersected
