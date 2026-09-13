import json

from src.trace.trace_logger import TraceLogger


def test_trace_logger_records_events(tmp_path):
    logger = TraceLogger(output_dir=str(tmp_path))

    logger.log_step(
        1,
        "TaskController",
        "parse_intent",
        {"task_type": "SINGLE_IMAGE_VQA"},
    )

    logger.log_step(
        2,
        "GeoReasonVerifier",
        "verify",
        {"status": "PASS"},
    )

    assert len(logger.events) == 2
    assert logger.events[0]["component"] == "TaskController"
    assert logger.events[1]["action"] == "verify"


def test_trace_logger_exports_json(tmp_path):
    logger = TraceLogger(output_dir=str(tmp_path))

    logger.log_step(
        1,
        "TaskController",
        "parse_intent",
        {"task_type": "vqa"},
    )

    output_file = logger.export_trace("T_TEST_01")

    assert output_file.exists()
    assert output_file.name == "trace_T_TEST_01.json"

    with output_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    assert payload["trace_id"] == "TRACE_T_TEST_01"
    assert payload["total_steps"] == 1
    assert len(payload["execution_trace"]) == 1
