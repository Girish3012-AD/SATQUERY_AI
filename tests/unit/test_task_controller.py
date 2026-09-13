from src.controller import TaskController


def test_simple_vqa():
    controller = TaskController()

    spec = controller.build_task_spec(
        "What type of land cover is present?",
        1,
    )

    assert spec.task_type == "vqa"
    assert "vqa" in spec.required_capabilities
    assert spec.input_count == 1
    assert spec.requires_temporal_pair is False


def test_temporal_building_flood_query():
    controller = TaskController()

    spec = controller.build_task_spec(
        "Find newly constructed buildings within 500 meters of flooded areas.",
        2,
    )

    assert spec.task_type == "temporal_analysis"
    assert spec.requires_temporal_pair is True
    assert "temporal_analysis" in spec.required_capabilities
    assert "building_detection" in spec.required_capabilities
    assert "flood_detection" in spec.required_capabilities
    assert "buffer" in spec.spatial_operations
    assert spec.parameters["distance_m"] == 500.0


def test_kilometer_conversion():
    controller = TaskController()

    spec = controller.build_task_spec(
        "Find buildings within 2 km of the river.",
        1,
    )

    assert spec.parameters["distance_m"] == 2000.0
    assert "building_detection" in spec.required_capabilities
    assert "buffer" in spec.spatial_operations


def test_sar_query():
    controller = TaskController()

    spec = controller.build_task_spec(
        "Identify flooded areas in this SAR image.",
        1,
    )

    assert "sar_analysis" in spec.required_capabilities
    assert "flood_detection" in spec.required_capabilities
    assert "sar" in spec.required_modalities


def test_empty_query_rejected():
    controller = TaskController()

    try:
        controller.build_task_spec("", 1)
    except ValueError as exc:
        assert str(exc) == "Query cannot be empty."
    else:
        raise AssertionError("Expected ValueError")


def test_negative_input_count_rejected():
    controller = TaskController()

    try:
        controller.build_task_spec("Find buildings.", -1)
    except ValueError as exc:
        assert str(exc) == "input_count cannot be negative."
    else:
        raise AssertionError("Expected ValueError")
