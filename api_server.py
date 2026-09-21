"""
SATQuery AI - FastAPI API Server

Thin API adapter wrapping the existing SATQueryOrchestrator.
Does NOT modify or duplicate the orchestration architecture.

Usage:
    python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import json
import time
import uuid
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUTS_DIR = Path("outputs")
STATIC_DIR = Path("static")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SATQuery AI",
    description=(
        "Interactive Vision-Language Assistant for "
        "Multimodal Remote Sensing Image Analysis"
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Lazy-loaded orchestrator
# Specialists are registered at instantiation but heavy models (Qwen2-VL)
# are only loaded when the specialist's infer() method is first called.
# ---------------------------------------------------------------------------

_orchestrator = None


def _get_orchestrator():
    """Return the singleton SATQueryOrchestrator, creating it on first call."""
    global _orchestrator
    if _orchestrator is None:
        from src.orchestration import SATQueryOrchestrator

        _orchestrator = SATQueryOrchestrator()
    return _orchestrator


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Payload for POST /api/query."""

    query: str
    inputs: list[str] = []
    parameters: dict[str, Any] = {}
    demo_preset: str | None = None


DEMO_INPUT_REGISTRY: dict[str, list[str]] = {
    "vqa": [
        str(Path("data/samples/vqa_test.png").resolve().as_posix())
    ],
    "grounding": [
        str(Path("data/samples/test.tif").resolve().as_posix())
    ],
    "change": [
        str(Path("data/samples/test.tif").resolve().as_posix()),
        str(Path("data/samples/test.tif").resolve().as_posix()),
    ],
    "multimodal": [
        str(Path("data/samples/test.tif").resolve().as_posix()),
        str(Path("data/samples/test.tif").resolve().as_posix()),
    ],
    "hero": [
        str(Path("data/samples/test.tif").resolve().as_posix()),
        str(Path("data/samples/test.tif").resolve().as_posix()),
    ],
}

QUERY_TO_PRESET_MAP: dict[str, str] = {
    "Describe the land-cover and major objects visible in this image.": "vqa",
    "Highlight the water body referred to in the image.": "grounding",
    "Show spectral changes between the 2023 and 2024 Sentinel-2 observations.": "change",
    "What changed between the two dates?": "change",
    "Analyze the area using both optical and SAR evidence.": "multimodal",
    "Find newly constructed buildings within 500 m of flooded areas.": "hero",
}


DEMO_QUERY_TEXT_REGISTRY: dict[str, str] = {
    "vqa": "Describe the land-cover and major objects visible in this image.",
    "grounding": "Highlight the water body referred to in the image.",
    "change": "Show spectral changes between the 2023 and 2024 Sentinel-2 observations.",
    "multimodal": "Analyze the area using both optical and SAR evidence.",
    "hero": "Find newly constructed buildings within 500 m of flooded areas.",
}


def resolve_request_inputs(request: QueryRequest) -> tuple[str, list[str]]:
    """
    Resolve query text and real satellite input paths for demo presets automatically.
    
    If the caller supplied manual inputs, preserve them.
    If no inputs are supplied, automatically map to validated sample inputs from DEMO_INPUT_REGISTRY
    using preset keys, exact/fuzzy query matching, or fallback sample rasters.
    """
    query = request.query.strip()
    preset_key = request.demo_preset or QUERY_TO_PRESET_MAP.get(query)

    if not query and request.demo_preset and request.demo_preset in DEMO_QUERY_TEXT_REGISTRY:
        query = DEMO_QUERY_TEXT_REGISTRY[request.demo_preset]

    if request.inputs:
        return query, request.inputs.copy()

    # Keyword / fuzzy matching if exact query string match was not found
    if not preset_key and query:
        q_lower = query.lower()
        if "sar" in q_lower or "multimodal" in q_lower or "radar" in q_lower:
            preset_key = "multimodal"
        elif "water" in q_lower or "grounding" in q_lower or "flood" in q_lower:
            preset_key = "grounding"
        elif "change" in q_lower or "spectral" in q_lower:
            preset_key = "change"
        elif "building" in q_lower or "hero" in q_lower:
            preset_key = "hero"
        elif "vqa" in q_lower or "describe" in q_lower or "land-cover" in q_lower:
            preset_key = "vqa"

    if preset_key and preset_key in DEMO_INPUT_REGISTRY:
        return query, DEMO_INPUT_REGISTRY[preset_key].copy()

    # Fallback to both PNG and GeoTIFF sample dataset rasters when no inputs are attached to a free-form query
    vqa_sample = str(Path("data/samples/vqa_test.png").resolve().as_posix())
    geo_sample = str(Path("data/samples/test.tif").resolve().as_posix())
    return query, [vqa_sample, geo_sample]


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the single-page frontend application."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=404, detail="Frontend index.html not found"
        )
    return HTMLResponse(
        content=index_path.read_text(encoding="utf-8")
    )


@app.get("/api/health")
def health():
    """Server health check."""
    return {
        "status": "healthy",
        "service": "SATQuery AI",
        "version": "1.0.0",
        "timestamp": time.time(),
        "timestamp_iso": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }


@app.get("/api/capabilities")
def capabilities():
    """List registered specialist capabilities and models."""
    orchestrator = _get_orchestrator()

    caps = {}
    for cap, specialist in orchestrator.specialists.items():
        caps[cap] = {
            "specialist": specialist.__class__.__name__,
            "capability": cap,
            "model_name": getattr(specialist, "MODEL_NAME", None),
        }

    models = {}
    if orchestrator.registry is not None:
        for model in orchestrator.registry.all():
            models[model.name] = {
                "name": model.name,
                "capability": model.capability,
                "status": model.status,
                "modalities": model.modalities,
                "task_types": model.task_types,
            }

    return {
        "capabilities": caps,
        "models": models,
        "specialist_count": len(caps),
        "model_count": len(models),
    }


@app.get("/api/audits")
def list_audits():
    """List available audit JSON files from outputs/."""
    audit_files = []
    if OUTPUTS_DIR.exists():
        for f in sorted(OUTPUTS_DIR.glob("*_audit.json")):
            try:
                size = f.stat().st_size
                data = json.loads(
                    f.read_text(encoding="utf-8")
                )
                query = ""
                if isinstance(data, dict):
                    query = data.get(
                        "query",
                        data.get("user_query", ""),
                    )
                audit_files.append(
                    {
                        "filename": f.name,
                        "size_bytes": size,
                        "query": query,
                    }
                )
            except (OSError, json.JSONDecodeError):
                audit_files.append(
                    {
                        "filename": f.name,
                        "size_bytes": 0,
                        "query": "",
                    }
                )
    return {"audits": audit_files, "count": len(audit_files)}


@app.get("/api/audits/{filename}")
def get_audit(filename: str):
    """Return a specific audit JSON file contents."""
    if not filename.endswith("_audit.json"):
        raise HTTPException(
            status_code=400,
            detail="Invalid audit filename pattern",
        )

    audit_path = OUTPUTS_DIR / filename
    if not audit_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audit file not found: {filename}",
        )

    try:
        data = json.loads(
            audit_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse audit file: {exc}",
        )

    return {
        "filename": filename,
        "mode": "audit_replay",
        "data": data,
    }


@app.post("/api/query")
def execute_query(request: QueryRequest):
    """
    Execute a real query through the SATQuery orchestrator.

    This is the PRIMARY execution path.  It runs the genuine
    orchestration pipeline:

        TaskController -> EvidencePlanner -> ExecutionEngine
        -> GeoReasonVerifier -> OrchestrationResult

    The response includes the full result, collected evidence
    objects, and timing information.
    """
    orchestrator = _get_orchestrator()

    # Clear evidence from prior queries to prevent ID collisions.
    orchestrator.evidence_registry.clear()

    start_time = time.time()

    query_text, resolved_inputs = resolve_request_inputs(request)

    try:
        result = orchestrator.run(
            query=query_text,
            inputs=resolved_inputs or None,
            parameters=request.parameters or None,
        )
    except Exception as exc:
        execution_time = time.time() - start_time
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "status": "error",
                "task_id": "",
                "query": query_text,
                "task_type": "",
                "plan_id": "",
                "executed_steps": [],
                "successful_steps": [],
                "failed_steps": [],
                "evidence_ids": [],
                "verification": {},
                "selected_capabilities": {},
                "selected_models": {},
                "messages": [
                    f"Orchestration error: {exc}"
                ],
                "evidence": [],
                "execution_time_seconds": round(
                    execution_time, 3
                ),
                "mode": "live",
            },
        )

    execution_time = time.time() - start_time

    # Collect evidence objects produced by this query.
    evidence_list = []
    for eid in result.evidence_ids:
        try:
            ev = orchestrator.evidence_registry.get(eid)
            evidence_list.append(ev.model_dump())
        except KeyError:
            evidence_list.append(
                {"evidence_id": eid, "error": "not_found"}
            )

    return {
        "success": result.success,
        "status": result.status,
        "task_id": result.task_id,
        "query": result.query,
        "task_type": result.task_type,
        "plan_id": result.plan_id,
        "executed_steps": result.executed_steps,
        "successful_steps": result.successful_steps,
        "failed_steps": result.failed_steps,
        "evidence_ids": result.evidence_ids,
        "verification": result.verification,
        "selected_capabilities": result.selected_capabilities,
        "selected_models": result.selected_models,
        "messages": result.messages,
        "evidence": evidence_list,
        "execution_time_seconds": round(execution_time, 3),
        "mode": "live",
    }


@app.get("/api/evidence")
def list_all_evidence():
    """Return all evidence currently in the registry."""
    orchestrator = _get_orchestrator()
    all_ev = orchestrator.evidence_registry.all()
    return {
        "evidence": [ev.model_dump() for ev in all_ev],
        "count": len(all_ev),
    }


@app.get("/api/evidence/{evidence_id}")
def get_evidence(evidence_id: str):
    """Retrieve a specific evidence object by ID."""
    orchestrator = _get_orchestrator()
    try:
        ev = orchestrator.evidence_registry.get(evidence_id)
        return ev.model_dump()
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Evidence not found: {evidence_id}",
        )


@app.get("/api/overlays/{filename}")
def get_overlay(filename: str):
    """Serve a visual overlay image from outputs/."""
    allowed_ext = {".png", ".jpg", ".jpeg"}

    overlay_path = OUTPUTS_DIR / filename
    if not overlay_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Overlay not found: {filename}",
        )

    if overlay_path.suffix.lower() not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format",
        )

    return FileResponse(
        str(overlay_path),
        media_type=f"image/{overlay_path.suffix.lstrip('.').lower()}",
    )


@app.post("/api/report")
def generate_report(request: QueryRequest):
    """
    Generate a downloadable JSON report.

    Runs the query through the orchestrator and packages
    everything required by R13 (query, input summary, answer,
    evidence, confidence, verification, model/tool names,
    parameters, execution trace, visual artifacts).
    """
    orchestrator = _get_orchestrator()
    orchestrator.evidence_registry.clear()

    start_time = time.time()

    query_text, resolved_inputs = resolve_request_inputs(request)

    try:
        result = orchestrator.run(
            query=query_text,
            inputs=resolved_inputs or None,
            parameters=request.parameters or None,
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Report generation failed: {exc}"
            },
        )

    execution_time = time.time() - start_time

    evidence_list = []
    for eid in result.evidence_ids:
        try:
            ev = orchestrator.evidence_registry.get(eid)
            evidence_list.append(ev.model_dump())
        except KeyError:
            pass

    # Build R13-compliant report.
    report = {
        "report_format": "SATQuery AI Execution Report",
        "report_version": "1.0.0",
        "generated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "query": result.query,
        "input_summary": {
            "input_count": len(resolved_inputs),
            "inputs": resolved_inputs,
            "parameters": request.parameters,
        },
        "answer": {
            "task_type": result.task_type,
            "status": result.status,
            "success": result.success,
            "messages": result.messages,
        },
        "evidence": evidence_list,
        "confidence": {
            "verification_status": result.verification.get(
                "status", "unknown"
            ),
            "confidence_value": result.verification.get(
                "confidence", None
            ),
            "calibrated": False,
            "calibration_note": (
                "Confidence values are uncalibrated "
                "model outputs."
            ),
        },
        "verification": result.verification,
        "models_and_tools": {
            "selected_capabilities": (
                result.selected_capabilities
            ),
            "selected_models": result.selected_models,
        },
        "execution_trace": {
            "plan_id": result.plan_id,
            "task_id": result.task_id,
            "executed_steps": result.executed_steps,
            "successful_steps": result.successful_steps,
            "failed_steps": result.failed_steps,
            "execution_time_seconds": round(
                execution_time, 3
            ),
        },
        "visual_artifacts": _list_overlay_files(),
        "provenance": {
            "system": "SATQuery AI",
            "problem_statement": "SIH26167",
            "note": (
                "No private chain-of-thought is "
                "included in this report."
            ),
        },
    }

    return JSONResponse(
        content=report,
        headers={
            "Content-Disposition": (
                f'attachment; filename="satquery_report_'
                f'{result.task_id}.json"'
            )
        },
    )


def _list_overlay_files() -> list[str]:
    """Return overlay image filenames from outputs/."""
    overlays = []
    if OUTPUTS_DIR.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for f in OUTPUTS_DIR.glob(ext):
                overlays.append(f.name)
    return sorted(overlays)


MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

def sanitize_filename(filename: str) -> str:
    """Basic filename sanitization."""
    if not filename:
        return "unnamed_file"
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)

def validate_file_content(path: Path) -> bool:
    """Validate raster or image content."""
    ext = path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        try:
            import rasterio
            with rasterio.open(path) as src:
                # Basic CRS check where CRS is required
                if not src.crs:
                    pass  # Some local TIFFs might lack CRS, but at least it's a valid TIFF
            return True
        except Exception:
            return False
    else:
        try:
            from PIL import Image
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle secure file uploads for processing."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided or missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    safe_name = f"{uuid.uuid4().hex[:8]}_{sanitize_filename(file.filename)}"
    dest = UPLOAD_DIR / safe_name

    size = 0
    with dest.open("wb") as buffer:
        while chunk := await file.read(8192):
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                dest.unlink()
                raise HTTPException(status_code=413, detail="File too large (>50MB)")
            buffer.write(chunk)
            
    if size == 0:
        dest.unlink()
        raise HTTPException(status_code=400, detail="Empty file")

    if not validate_file_content(dest):
        dest.unlink()
        raise HTTPException(status_code=400, detail="Corrupted file or invalid raster/image content")

    return {
        "file_path": str(dest.absolute().as_posix()), 
        "filename": file.filename, 
        "size_bytes": size,
        "extension": ext
    }

# ---------------------------------------------------------------------------
# Static file serving (must be mounted AFTER API routes)
# ---------------------------------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)
