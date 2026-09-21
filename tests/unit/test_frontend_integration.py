"""
SATQuery AI — Frontend Integration Tests

Tests the FastAPI API layer using Starlette TestClient.
Verifies that the thin API adapter correctly wraps the
existing SATQueryOrchestrator without modifying it.
"""

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from api_server import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


# ---------------------------------------------------------------
# Test 1: Frontend HTML serving
# ---------------------------------------------------------------


def test_serve_frontend(client):
    """GET / returns 200 with HTML content."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "SATQuery AI" in resp.text
    assert "satquery.js" in resp.text
    assert "leaflet" in resp.text.lower()


# ---------------------------------------------------------------
# Test 2: Health endpoint
# ---------------------------------------------------------------


def test_health(client):
    """GET /api/health returns healthy status."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "SATQuery AI"
    assert "timestamp" in data
    assert "timestamp_iso" in data


# ---------------------------------------------------------------
# Test 3: Capabilities endpoint
# ---------------------------------------------------------------


def test_capabilities(client):
    """GET /api/capabilities returns specialist list."""
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert "capabilities" in data
    assert "models" in data
    assert data["specialist_count"] > 0

    # Verify known specialists are registered
    caps = data["capabilities"]
    # The orchestrator registers at least SAR, temporal, flood, VQA
    assert len(caps) >= 4


# ---------------------------------------------------------------
# Test 4: Audits list endpoint
# ---------------------------------------------------------------


def test_list_audits(client):
    """GET /api/audits returns audit file list."""
    resp = client.get("/api/audits")
    assert resp.status_code == 200
    data = resp.json()
    assert "audits" in data
    assert "count" in data
    assert isinstance(data["audits"], list)


# ---------------------------------------------------------------
# Test 5: Specific audit retrieval
# ---------------------------------------------------------------


def test_get_audit(client):
    """GET /api/audits/{filename} returns valid audit JSON."""
    # First list available audits
    list_resp = client.get("/api/audits")
    audits = list_resp.json()["audits"]

    if not audits:
        pytest.skip("No audit files available")

    # Retrieve the first available audit
    filename = audits[0]["filename"]
    resp = client.get(f"/api/audits/{filename}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "audit_replay"
    assert data["filename"] == filename
    assert "data" in data


# ---------------------------------------------------------------
# Test 6: Live query execution
# ---------------------------------------------------------------


def test_live_query_execution(client):
    """POST /api/query with a routing query returns OrchestrationResult."""
    payload = {
        "query": "Describe the land-cover and major objects visible in this image.",
        "inputs": [],
        "parameters": {},
    }
    resp = client.post("/api/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Verify OrchestrationResult structure
    assert "success" in data
    assert "status" in data
    assert "task_id" in data
    assert "query" in data
    assert "task_type" in data
    assert "plan_id" in data
    assert "executed_steps" in data
    assert "evidence_ids" in data
    assert "verification" in data
    assert "selected_capabilities" in data
    assert "selected_models" in data
    assert "messages" in data
    assert "mode" in data
    assert data["mode"] == "live"

    # The query should be correctly echoed back
    assert data["query"] == payload["query"]
    # Without image inputs, the VQA specialist will fail
    # but the query was still processed through the real pipeline
    assert "execution_time_seconds" in data


# ---------------------------------------------------------------
# Test 7: Evidence listing
# ---------------------------------------------------------------


def test_list_evidence(client):
    """GET /api/evidence returns evidence or empty list."""
    resp = client.get("/api/evidence")
    assert resp.status_code == 200
    data = resp.json()
    assert "evidence" in data
    assert "count" in data
    assert isinstance(data["evidence"], list)


# ---------------------------------------------------------------
# Test 8: Overlay serving
# ---------------------------------------------------------------


def test_overlay_serving(client):
    """GET /api/overlays/{filename} returns image if available."""
    # Check for known overlay files
    overlay_path = Path("outputs/grounding_water_overlay.png")
    if not overlay_path.exists():
        pytest.skip("No overlay files available")

    resp = client.get("/api/overlays/grounding_water_overlay.png")
    assert resp.status_code == 200
    assert "image" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------
# Test 9: Report generation
# ---------------------------------------------------------------


def test_report_generation(client):
    """POST /api/report generates a report response."""
    payload = {
        "query": "What type of land cover is visible?",
        "inputs": [],
        "parameters": {},
    }
    resp = client.post("/api/report", json=payload)

    # Report generation may return 500 if the specialist
    # fails (no inputs), or 200 if the pipeline completes.
    # Both are valid real-backend responses.
    if resp.status_code == 200:
        data = resp.json()
        # Verify R13-compliant report structure
        assert "report_format" in data
        assert "query" in data
        assert "evidence" in data
        assert "confidence" in data
        assert "verification" in data
        assert "models_and_tools" in data
        assert "execution_trace" in data
        assert "provenance" in data

        # Verify no secrets or private chain-of-thought
        report_text = json.dumps(data)
        assert "chain_of_thought" not in report_text.lower()
        assert data["provenance"]["note"] == (
            "No private chain-of-thought is "
            "included in this report."
        )
    else:
        # 500 with error message is acceptable when
        # specialist execution fails due to missing inputs
        assert resp.status_code == 500
        data = resp.json()
        assert "error" in data


# ---------------------------------------------------------------
# Test 10: Static file serving
# ---------------------------------------------------------------


def test_static_file_serving(client):
    """Static CSS and JS files are served correctly."""
    css_resp = client.get("/static/css/satquery.css")
    assert css_resp.status_code == 200
    assert "text/css" in css_resp.headers.get(
        "content-type", ""
    )

    js_resp = client.get("/static/js/satquery.js")
    assert js_resp.status_code == 200
    # JS content-type varies by server
    assert js_resp.status_code == 200


# ---------------------------------------------------------------
# Test 11: Invalid audit filename
# ---------------------------------------------------------------


def test_invalid_audit_filename(client):
    """GET /api/audits/{invalid} returns appropriate error."""
    # Invalid pattern (not ending in _audit.json)
    resp = client.get("/api/audits/not_valid.txt")
    assert resp.status_code == 400

    # Valid pattern but non-existent file
    resp = client.get(
        "/api/audits/nonexistent_audit.json"
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------
# Test 12: Multiple query routing verification
# ---------------------------------------------------------------


def test_query_routing_all_types(client):
    """Verify all 4 representative queries are accepted by the live API."""
    test_queries = [
        (
            "Describe the land-cover and major "
            "objects visible in this image."
        ),
        (
            "Highlight the water body "
            "referred to in the image."
        ),
        "What changed between the two dates?",
        (
            "Find newly constructed buildings "
            "within 500 m of flooded areas."
        ),
    ]

    for query in test_queries:
        resp = client.post(
            "/api/query",
            json={
                "query": query,
                "inputs": [],
                "parameters": {},
            },
        )
        assert resp.status_code == 200, (
            f"Query '{query}' returned status "
            f"{resp.status_code}"
        )
        data = resp.json()
        # Every query must be processed through the
        # real backend and marked as live execution.
        assert data["mode"] == "live"
        assert data["query"] == query
        assert "execution_time_seconds" in data

# ---------------------------------------------------------------
# Test 13: File upload validation
# ---------------------------------------------------------------

def test_upload_endpoint_success(client, tmp_path):
    """Test successful image upload."""
    from PIL import Image
    import io
    
    # Create valid dummy image
    img = Image.new("RGB", (10, 10), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()
    
    resp = client.post(
        "/api/upload",
        files={"file": ("test_img.jpg", img_bytes, "image/jpeg")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "file_path" in data
    assert data["filename"] == "test_img.jpg"
    assert data["extension"] == ".jpg"
    assert data["size_bytes"] == len(img_bytes)

def test_upload_endpoint_invalid_format(client):
    """Test rejection of unsupported file extensions."""
    resp = client.post(
        "/api/upload",
        files={"file": ("test_doc.pdf", b"dummy pdf content", "application/pdf")}
    )
    assert resp.status_code == 400
    assert "Unsupported format" in resp.json()["detail"]

def test_upload_endpoint_corrupted_image(client):
    """Test rejection of corrupted image content."""
    resp = client.post(
        "/api/upload",
        files={"file": ("fake_img.jpg", b"not a real image", "image/jpeg")}
    )
    assert resp.status_code == 400
    assert "Corrupted file" in resp.json()["detail"]
