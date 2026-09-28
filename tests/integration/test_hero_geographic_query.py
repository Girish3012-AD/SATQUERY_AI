import pytest
from src.controller.task_controller import TaskController
from src.planner.evidence_planner import EvidencePlanner

def test_hero_geographic_query_decomposition():
    """
    Test that the orchestrator correctly decomposes the complex hero query
    into a valid Directed Acyclic Graph (DAG) for execution.
    
    Query: 'Find newly constructed buildings within 500 m of flooded areas.'
    """
    query = "Find newly constructed buildings within 500 m of flooded areas."
    
    # 1. Controller parsing
    controller = TaskController()
    spec = controller.build_task_spec(query, input_count=2)
    
    assert spec.task_type == "temporal_analysis"
    assert "building_detection" in spec.required_capabilities
    assert "flood_detection" in spec.required_capabilities
    assert "buffer" in spec.spatial_operations
    assert "intersection" in spec.spatial_operations
    assert spec.parameters.get("distance_m") == 500.0
    
    # 2. Planner DAG generation
    planner = EvidencePlanner()
    plan = planner.create_plan(spec)
    
    steps = {step.step_id: step for step in plan.steps}
    
    # Find specialist detection steps
    flood_step_id = next(s.step_id for s in plan.steps if s.task == "flood_detection")
    building_step_id = next(s.step_id for s in plan.steps if s.task == "building_detection")
    
    # Verify buffer step depends on the FIRST specialist (flood)
    buffer_step = next(s for s in plan.steps if s.task == "buffer")
    assert flood_step_id in buffer_step.depends_on
    
    # Verify intersection depends on the buffer AND the SECOND specialist (buildings)
    intersection_step = next(s for s in plan.steps if s.task == "intersection")
    assert buffer_step.step_id in intersection_step.depends_on
    assert building_step_id in intersection_step.depends_on
    
    # Verify verification is final
    verification_step = plan.steps[-1]
    assert verification_step.task == "verification"
    assert intersection_step.step_id in verification_step.depends_on
