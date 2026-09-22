/**
 * SATQuery AI — Frontend Application Logic
 *
 * Connects to the FastAPI backend at /api/* endpoints.
 * All data comes from the REAL backend — nothing is hardcoded.
 */

/* ------------------------------------------------------------------ */
/*  Demo query definitions                                            */
/* ------------------------------------------------------------------ */

const DEMO_QUERIES = [
    {
        label: "▶ Run VQA Demo",
        query: "Describe the land-cover and major objects visible in this image.",
        preset: "vqa",
        description: "📷 Attached Input: data/samples/vqa_test.png (Optical Scene)",
    },
    {
        label: "▶ Run Water Grounding Demo",
        query: "Highlight the water body referred to in the image.",
        preset: "grounding",
        description: "📡 Attached Input: data/samples/test.tif (Sentinel-2 GeoTIFF)",
    },
    {
        label: "▶ Run Change Detection Demo",
        query: "Show spectral changes between the 2023 and 2024 Sentinel-2 observations.",
        preset: "change",
        description: "📡 Attached Inputs: T1 & T2 data/samples/test.tif Rasters",
    },
    {
        label: "▶ Run Optical + SAR Demo",
        query: "Analyze the area using both optical and SAR evidence.",
        preset: "multimodal",
        description: "📡 Attached Inputs: Optical Sentinel-2 & C-Band SAR Rasters",
    },
    {
        label: "★ Run Hero Geographic Reasoning",
        query: "Find newly constructed buildings within 500 m of flooded areas.",
        preset: "hero",
        description: "📡 Attached Inputs: Multi-Temporal Rasters & SpaceNet UNet Weights",
    },
];

/* ------------------------------------------------------------------ */
/*  SATQueryApp                                                       */
/* ------------------------------------------------------------------ */

class SATQueryApp {
    constructor() {
        this.map = null;
        this.mapLayers = [];
        this.currentResult = null;
        this.currentMode = "ready"; // ready | live | replay
        this.uploadedFilePath = null;
        this.currentDemoPreset = null;
    }

    /* ---------- Initialization ---------- */

    init() {
        this.loadCapabilities();
        this.loadAuditList();
    }

    /* ---------- Capabilities ---------- */

    async loadCapabilities() {
        try {
            const resp = await fetch("/api/capabilities");
            if (!resp.ok) return;
            const data = await resp.json();
            console.log(
                "SATQuery capabilities:",
                data.specialist_count,
                "specialists,",
                data.model_count,
                "models"
            );
        } catch (e) {
            console.warn("Could not load capabilities:", e);
        }
    }

    /* ---------- Audit list ---------- */

    async loadAuditList() {
        try {
            const resp = await fetch("/api/audits");
            if (!resp.ok) return;
            const data = await resp.json();
            const container = document.getElementById("audit-list");
            if (!container) return;

            if (data.count === 0) {
                container.innerHTML =
                    '<p style="color:#7f8c8d;">No audit files available.</p>';
                return;
            }

            let html = "";
            for (const audit of data.audits) {
                const sizeKB = (audit.size_bytes / 1024).toFixed(1);
                const queryPreview = audit.query
                    ? audit.query.substring(0, 80)
                    : audit.filename;
                html += `<button class="demo-btn" onclick="app.loadAuditReplay('${audit.filename}')">
                    📄 ${queryPreview}
                    <span style="color:#7f8c8d;font-size:0.8em;display:block;">${audit.filename} (${sizeKB} KB)</span>
                </button>`;
            }
            container.innerHTML = html;
        } catch (e) {
            console.warn("Could not load audit list:", e);
        }
    }

    /* ---------- Demo query presets ---------- */

    setDemoQuery(index) {
        if (index < 0 || index >= DEMO_QUERIES.length) return;
        const demo = DEMO_QUERIES[index];
        const textarea = document.getElementById("query-input");
        if (textarea) textarea.value = demo.query;
        this.currentDemoPreset = demo.preset;
        this.uploadedFilePath = null; // Clear manual upload when demo preset is selected
        const statusEl = document.getElementById("upload-status");
        if (statusEl) {
            statusEl.innerHTML = `✨ <span style="color:#2ecc71;font-weight:bold;">${demo.description}</span>`;
        }
    }

    /* ---------- File Upload ---------- */

    async handleFileUpload(event) {
        const file = event.target.files[0];
        const statusEl = document.getElementById("upload-status");
        if (!file) {
            this.uploadedFilePath = null;
            if (statusEl) statusEl.textContent = "No file selected";
            return;
        }

        this.currentDemoPreset = null; // Clear demo preset when manual file is uploaded

        if (statusEl) {
            statusEl.textContent = "Uploading...";
            statusEl.style.color = "#3498db";
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const resp = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data.detail || "Upload failed");
            }

            this.uploadedFilePath = data.file_path;
            if (statusEl) {
                statusEl.innerHTML = `✅ <span style="color:#2ecc71;">Uploaded: ${data.filename} (${(data.size_bytes/1024).toFixed(1)} KB)</span>`;
            }
        } catch (e) {
            this.uploadedFilePath = null;
            if (statusEl) {
                statusEl.innerHTML = `❌ <span style="color:#e74c3c;">${e.message}</span>`;
            }
            event.target.value = ""; // Reset file input
        }
    }

    /* ---------- Live query execution ---------- */

    async submitQuery() {
        const textarea = document.getElementById("query-input");
        const query = textarea ? textarea.value.trim() : "";
        if (!query) {
            alert("Please enter a query.");
            return;
        }

        this.setMode("live");
        this.showLoading(true, "Executing query through real SATQuery pipeline...");
        this.clearResults();

        try {
            const inputs = this.uploadedFilePath ? [this.uploadedFilePath] : [];
            const payload = {
                query: query,
                inputs: inputs,
                parameters: {},
            };
            if (this.currentDemoPreset && !this.uploadedFilePath) {
                payload.demo_preset = this.currentDemoPreset;
            }
            const resp = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = await resp.json();
            this.currentResult = data;
            this.renderResult(data, "live");
        } catch (e) {
            this.renderError("Query execution failed: " + e.message);
        } finally {
            this.showLoading(false);
        }
    }

    /* ---------- Audit replay ---------- */

    async loadAuditReplay(filename) {
        this.setMode("replay");
        this.showLoading(true, "Loading audit replay...");
        this.clearResults();

        try {
            const resp = await fetch("/api/audits/" + encodeURIComponent(filename));
            if (!resp.ok) throw new Error("Audit not found: " + filename);
            const wrapper = await resp.json();
            this.currentResult = wrapper;
            this.renderAuditReplay(wrapper);
        } catch (e) {
            this.renderError("Audit replay failed: " + e.message);
        } finally {
            this.showLoading(false);
        }
    }

    /* ---------- Mode management ---------- */

    setMode(mode) {
        this.currentMode = mode;
        const indicator = document.getElementById("mode-indicator");
        if (!indicator) return;

        if (mode === "live") {
            indicator.innerHTML =
                '<span class="status-badge status-badge--live">⚡ LIVE EXECUTION</span>';
        } else if (mode === "replay") {
            indicator.innerHTML =
                '<span class="status-badge status-badge--replay">📂 AUDIT REPLAY</span>';
        } else {
            indicator.innerHTML =
                '<span class="status-badge status-badge--live">READY</span>';
        }
    }

    /* ---------- Loading ---------- */

    showLoading(show, text) {
        const overlay = document.getElementById("loading-overlay");
        const loadingText = document.getElementById("loading-text");
        if (overlay) overlay.style.display = show ? "flex" : "none";
        if (loadingText && text) loadingText.textContent = text;
    }

    /* ---------- Clear ---------- */

    clearResults() {
        const panels = [
            "scene-panel", "status-panel", "answer-panel",
            "map-panel", "evidence-panel", "confidence-panel",
            "trace-panel", "visual-panel", "report-panel",
        ];
        for (const id of panels) {
            const el = document.getElementById(id);
            if (el) el.classList.add("hidden");
        }
        this.clearMap();
    }

    /* ---------- Error ---------- */

    renderError(msg) {
        this.showPanel("answer-panel");
        const el = document.getElementById("answer-content");
        if (el) {
            el.innerHTML = `<div class="status-badge status-badge--failed">ERROR</div>
                <p style="margin-top:12px;color:#e74c3c;">${this.esc(msg)}</p>`;
        }
    }

    /* ---------- Render live result ---------- */

    renderResult(data, mode) {
        // Panel 2: Scene info
        this.renderScenePanel(data, mode);
        // Panel 3: Execution status
        this.renderStatusPanel(data, mode);
        // Panel 4: Final answer
        this.renderAnswerPanel(data, mode);
        // Panel 5: Map
        this.renderMapPanel(data);
        // Panel 6: Evidence
        this.renderEvidencePanel(data);
        // Panel 7: Confidence
        this.renderConfidencePanel(data);
        // Panel 8: Execution trace
        this.renderTracePanel(data, mode);
        // Panel 9: Visual evidence
        this.renderVisualPanel(data);
        // Panel 10: Report
        this.renderReportPanel(data, mode);
    }

    /* ---------- Render audit replay ---------- */

    renderAuditReplay(wrapper) {
        const data = wrapper.data || {};
        const filename = wrapper.filename || "";

        // Build a normalized view from the audit data
        const normalized = {
            success: data.success !== undefined ? data.success : true,
            status: data.status || data.verification_status || "completed",
            task_id: data.task_id || "",
            query: data.query || data.user_query || "",
            task_type: data.task_type || "",
            plan_id: data.plan_id || "",
            executed_steps: data.executed_steps || data.steps_executed || [],
            successful_steps: data.successful_steps || [],
            failed_steps: data.failed_steps || [],
            evidence_ids: data.evidence_ids || [],
            verification: data.verification || data.georeason_verification || {},
            selected_capabilities: data.selected_capabilities || {},
            selected_models: data.selected_models || {},
            messages: data.messages || [],
            evidence: data.evidence || data.evidence_objects || [],
            execution_time_seconds: data.execution_time_seconds || data.total_time_seconds || 0,
            mode: "audit_replay",
            audit_filename: filename,
            // Pass through raw audit data for trace rendering
            _raw: data,
        };

        this.currentResult = normalized;
        this.renderResult(normalized, "replay");
    }

    /* ---------- Panel 2: Scene info ---------- */

    renderScenePanel(data, mode) {
        this.showPanel("scene-panel");
        const el = document.getElementById("scene-content");
        if (!el) return;

        const modeLabel = mode === "live"
            ? '<span class="status-badge status-badge--live">LIVE</span>'
            : '<span class="status-badge status-badge--replay">REPLAY</span>';

        let html = `<div style="margin-bottom:8px;">${modeLabel}</div>`;
        html += `<div class="evidence-card">`;
        html += `<div class="evidence-card__field"><span class="field-label">Task Type:</span> ${this.esc(data.task_type || "—")}</div>`;
        html += `<div class="evidence-card__field"><span class="field-label">Task ID:</span> <code>${this.esc(data.task_id || "—")}</code></div>`;
        html += `<div class="evidence-card__field"><span class="field-label">Plan ID:</span> <code>${this.esc(data.plan_id || "—")}</code></div>`;

        if (data.selected_models && Object.keys(data.selected_models).length > 0) {
            html += `<div class="evidence-card__field"><span class="field-label">Models:</span>`;
            for (const [cap, model] of Object.entries(data.selected_models)) {
                html += `<br>&nbsp;&nbsp;${this.esc(cap)}: <code>${this.esc(model)}</code>`;
            }
            html += `</div>`;
        }

        if (data.selected_capabilities && Object.keys(data.selected_capabilities).length > 0) {
            html += `<div class="evidence-card__field"><span class="field-label">Specialists:</span>`;
            for (const [cap, spec] of Object.entries(data.selected_capabilities)) {
                html += `<br>&nbsp;&nbsp;${this.esc(cap)}: <code>${this.esc(spec)}</code>`;
            }
            html += `</div>`;
        }

        html += `</div>`;
        el.innerHTML = html;
    }

    /* ---------- Panel 3: Execution status ---------- */

    renderStatusPanel(data, mode) {
        this.showPanel("status-panel");
        const el = document.getElementById("status-content");
        if (!el) return;

        const statusClass = this.statusBadgeClass(data.status);
        const successIcon = data.success ? "✅" : "❌";

        let html = `<div style="margin-bottom:12px;">
            ${successIcon}
            <span class="status-badge ${statusClass}">${this.esc((data.status || "unknown").toUpperCase())}</span>`;

        if (mode === "replay") {
            html += ` <span class="status-badge status-badge--replay">AUDIT REPLAY</span>`;
        }
        html += `</div>`;

        const totalSteps = data.executed_steps ? data.executed_steps.length : 0;
        const passedSteps = data.successful_steps ? data.successful_steps.length : 0;
        const failedSteps = data.failed_steps ? data.failed_steps.length : 0;

        html += `<div class="evidence-card">`;
        html += `<div class="evidence-card__field"><span class="field-label">Steps Executed:</span> ${totalSteps}</div>`;
        html += `<div class="evidence-card__field"><span class="field-label">Passed:</span> <span style="color:#2ecc71;">${passedSteps}</span></div>`;
        if (failedSteps > 0) {
            html += `<div class="evidence-card__field"><span class="field-label">Failed:</span> <span style="color:#e74c3c;">${failedSteps}</span></div>`;
        }
        if (data.execution_time_seconds) {
            html += `<div class="evidence-card__field"><span class="field-label">Execution Time:</span> ${data.execution_time_seconds.toFixed(3)}s</div>`;
        }
        html += `</div>`;

        el.innerHTML = html;
    }

    /* ---------- Panel 4: Final answer ---------- */

    renderAnswerPanel(data, mode) {
        this.showPanel("answer-panel");
        const el = document.getElementById("answer-content");
        if (!el) return;

        const statusClass = this.statusBadgeClass(data.status);
        let html = `<span class="status-badge ${statusClass}">${this.esc((data.status || "unknown").toUpperCase())}</span>`;

        // Extract answer text from evidence or messages
        let answerText = "";
        if (data.evidence && data.evidence.length > 0) {
            for (const ev of data.evidence) {
                if (ev.result && typeof ev.result === "string" && ev.result.length > 10) {
                    answerText = ev.result;
                    break;
                }
                if (ev.result && typeof ev.result === "object") {
                    if (ev.result.answer) answerText = ev.result.answer;
                    else if (ev.result.text) answerText = ev.result.text;
                    else if (ev.result.description) answerText = ev.result.description;
                }
            }
        }

        if (!answerText && data.messages && data.messages.length > 0) {
            answerText = data.messages.join("\n");
        }

        if (!answerText) {
            answerText = data.success
                ? "Query processed successfully. See evidence panel for details."
                : "Query execution encountered issues. See status panel for details.";
        }

        html += `<div style="margin-top:12px;font-size:1.1em;line-height:1.6;">${this.esc(answerText)}</div>`;

        // Show measurements if available
        if (data.evidence) {
            const measurements = data.evidence
                .filter(ev => ev.measurement && Object.keys(ev.measurement).length > 0)
                .map(ev => ev.measurement);

            if (measurements.length > 0) {
                html += `<div style="margin-top:12px;"><strong>Measurements:</strong></div>`;
                for (const m of measurements) {
                    html += `<div class="evidence-card">`;
                    for (const [k, v] of Object.entries(m)) {
                        html += `<div class="evidence-card__field"><span class="field-label">${this.esc(k)}:</span> ${this.esc(String(v))}</div>`;
                    }
                    html += `</div>`;
                }
            }
        }

        el.innerHTML = html;
    }

    /* ---------- Panel 5: Map ---------- */

    renderMapPanel(data) {
        const geometries = [];
        if (data.evidence) {
            for (const ev of data.evidence) {
                if (ev.geometry && ev.geometry.type) {
                    geometries.push({
                        geometry: ev.geometry,
                        task: ev.task || "unknown",
                        evidence_id: ev.evidence_id || "",
                    });
                }
            }
        }

        // Also check raw audit data for geometries
        if (data._raw) {
            this.extractGeometriesFromRaw(data._raw, geometries);
        }

        if (geometries.length === 0) {
            // No geometries to display — show default AOI
            this.showPanel("map-panel");
            this.initMap();
            // Default Rasuwa AOI
            if (this.map) {
                this.map.setView([28.25, 85.25], 10);
            }
            return;
        }

        this.showPanel("map-panel");
        this.initMap();
        this.clearMap();

        const colors = {
            water_detection: "#3498db",
            flood_detection: "#2980b9",
            building_detection: "#e74c3c",
            temporal_analysis: "#f39c12",
            buffer: "#9b59b6",
            intersection: "#2ecc71",
            area: "#1abc9c",
            gis_buffer: "#9b59b6",
            gis_intersection: "#2ecc71",
            gis_area: "#1abc9c",
            vqa: "#3498db",
        };

        const allBounds = [];

        for (const item of geometries) {
            try {
                const color = colors[item.task] || "#3498db";
                const layer = L.geoJSON(item.geometry, {
                    style: {
                        color: color,
                        weight: 2,
                        fillColor: color,
                        fillOpacity: 0.2,
                    },
                    onEachFeature: (feature, layer) => {
                        layer.bindPopup(
                            `<strong>${this.esc(item.task)}</strong><br>` +
                            `<code>${this.esc(item.evidence_id)}</code>`
                        );
                    },
                });
                layer.addTo(this.map);
                this.mapLayers.push(layer);

                const bounds = layer.getBounds();
                if (bounds.isValid()) {
                    allBounds.push(bounds);
                }
            } catch (e) {
                console.warn("Failed to render geometry:", e);
            }
        }

        // Fit map to all geometries
        if (allBounds.length > 0) {
            let combinedBounds = allBounds[0];
            for (let i = 1; i < allBounds.length; i++) {
                combinedBounds.extend(allBounds[i]);
            }
            this.map.fitBounds(combinedBounds, { padding: [20, 20] });
        }
    }

    extractGeometriesFromRaw(raw, geometries) {
        // Recursively search for geometry objects in audit data
        if (!raw || typeof raw !== "object") return;

        if (raw.geometry && raw.geometry.type) {
            geometries.push({
                geometry: raw.geometry,
                task: raw.task || raw.evidence_type || "unknown",
                evidence_id: raw.evidence_id || "",
            });
        }

        if (Array.isArray(raw)) {
            for (const item of raw) {
                this.extractGeometriesFromRaw(item, geometries);
            }
        } else {
            for (const key of Object.keys(raw)) {
                if (key === "geometry") continue; // Already handled
                this.extractGeometriesFromRaw(raw[key], geometries);
            }
        }
    }

    initMap() {
        if (this.map) return;
        const mapEl = document.getElementById("map");
        if (!mapEl) return;

        this.map = L.map("map").setView([28.25, 85.25], 10);
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
            maxZoom: 19,
        }).addTo(this.map);
    }

    clearMap() {
        if (!this.map) return;
        for (const layer of this.mapLayers) {
            this.map.removeLayer(layer);
        }
        this.mapLayers = [];
    }

    /* ---------- Panel 6: Evidence ---------- */

    renderEvidencePanel(data) {
        if (!data.evidence || data.evidence.length === 0) return;
        this.showPanel("evidence-panel");
        const el = document.getElementById("evidence-content");
        if (!el) return;

        let html = `<p style="color:#7f8c8d;margin-bottom:12px;">${data.evidence.length} evidence object(s)</p>`;

        for (const ev of data.evidence) {
            const verifiedBadge = (ev.confidence !== undefined && ev.confidence >= 0.6)
                ? '<span class="status-badge status-badge--verified">VERIFIED</span>'
                : '<span class="status-badge status-badge--low-confidence">UNVERIFIED</span>';

            html += `<div class="evidence-card">`;
            html += `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">`;
            html += `<code class="evidence-card__id">${this.esc(ev.evidence_id || "—")}</code>`;
            html += verifiedBadge;
            html += `</div>`;
            html += `<div class="evidence-card__field"><span class="field-label">Task:</span> ${this.esc(ev.task || "—")}</div>`;
            html += `<div class="evidence-card__field"><span class="field-label">Model:</span> ${this.esc(ev.model || "—")}</div>`;
            if (ev.sensor) html += `<div class="evidence-card__field"><span class="field-label">Sensor:</span> ${this.esc(ev.sensor)}</div>`;
            if (ev.modality) html += `<div class="evidence-card__field"><span class="field-label">Modality:</span> ${this.esc(ev.modality)}</div>`;
            if (ev.confidence !== undefined) {
                html += `<div class="evidence-card__field"><span class="field-label">Confidence:</span> ${(ev.confidence * 100).toFixed(1)}%</div>`;
            }
            if (ev.timestamp) html += `<div class="evidence-card__field"><span class="field-label">Timestamp:</span> ${this.esc(ev.timestamp)}</div>`;
            if (ev.t1_timestamp) html += `<div class="evidence-card__field"><span class="field-label">T1:</span> ${this.esc(ev.t1_timestamp)}</div>`;
            if (ev.t2_timestamp) html += `<div class="evidence-card__field"><span class="field-label">T2:</span> ${this.esc(ev.t2_timestamp)}</div>`;

            if (ev.measurement && Object.keys(ev.measurement).length > 0) {
                html += `<div class="evidence-card__field"><span class="field-label">Measurement:</span></div>`;
                for (const [k, v] of Object.entries(ev.measurement)) {
                    html += `<div class="evidence-card__field" style="padding-left:16px;">${this.esc(k)}: <strong>${this.esc(String(v))}</strong></div>`;
                }
            }

            if (ev.result && typeof ev.result === "string" && ev.result.length > 0) {
                const preview = ev.result.length > 200 ? ev.result.substring(0, 200) + "..." : ev.result;
                html += `<div class="evidence-card__field"><span class="field-label">Result:</span> ${this.esc(preview)}</div>`;
            }

            if (ev.geometry && ev.geometry.type) {
                html += `<div class="evidence-card__field"><span class="field-label">Geometry:</span> ${this.esc(ev.geometry.type)}</div>`;
            }

            html += `</div>`;
        }

        el.innerHTML = html;
    }

    /* ---------- Panel 7: Confidence ---------- */

    renderConfidencePanel(data) {
        this.showPanel("confidence-panel");
        const el = document.getElementById("confidence-content");
        if (!el) return;

        let html = "";

        // Verification status
        const verification = data.verification || {};
        const verStatus = verification.status || data.status || "unknown";
        const verStatusClass = this.statusBadgeClass(verStatus);

        html += `<div style="margin-bottom:12px;">
            <span class="status-badge ${verStatusClass}">${this.esc(verStatus.toUpperCase())}</span>
        </div>`;

        // Confidence value
        const confValue = verification.confidence;
        if (confValue !== undefined && confValue !== null) {
            const confPercent = (confValue * 100).toFixed(1);
            const barColor = confValue >= 0.6 ? "#2ecc71" : confValue >= 0.3 ? "#f39c12" : "#e74c3c";

            html += `<div class="confidence-meter">
                <div class="confidence-meter__fill" style="width:${confPercent}%;background:${barColor};"></div>
            </div>
            <div style="text-align:center;font-size:1.4em;font-weight:bold;margin:8px 0;">${confPercent}%</div>`;
        }

        // Calibration warning — always shown because confidence is uncalibrated
        html += `<div class="calibration-warning">
            ⚠️ <strong>Uncalibrated confidence.</strong> Values are raw model/rule outputs,
            not calibrated posterior probabilities. Do not interpret as true probability.
        </div>`;

        // Verification reasons
        if (verification.reasons && verification.reasons.length > 0) {
            html += `<div style="margin-top:12px;"><strong>Verification Reasons:</strong></div><ul>`;
            for (const reason of verification.reasons) {
                html += `<li>${this.esc(reason)}</li>`;
            }
            html += `</ul>`;
        }

        // Conflicts
        if (verification.conflicts && verification.conflicts.length > 0) {
            html += `<div style="margin-top:12px;color:#e74c3c;"><strong>Conflicts:</strong></div><ul>`;
            for (const conflict of verification.conflicts) {
                html += `<li style="color:#e74c3c;">${this.esc(conflict)}</li>`;
            }
            html += `</ul>`;
        }

        el.innerHTML = html;
    }

    /* ---------- Panel 8: Execution trace ---------- */

    renderTracePanel(data, mode) {
        this.showPanel("trace-panel");
        const el = document.getElementById("trace-content");
        if (!el) return;

        const modeLabel = mode === "live"
            ? '<span class="status-badge status-badge--live">LIVE TRACE</span>'
            : '<span class="status-badge status-badge--replay">REPLAY TRACE</span>';

        let html = `<div style="margin-bottom:12px;">${modeLabel}</div>`;
        html += `<div class="trace-box">`;

        // Build trace from executed steps and messages
        const steps = data.executed_steps || [];
        const successSet = new Set(data.successful_steps || []);
        const failSet = new Set(data.failed_steps || []);

        if (steps.length > 0) {
            for (let i = 0; i < steps.length; i++) {
                const stepId = steps[i];
                const isSuccess = successSet.has(stepId);
                const isFail = failSet.has(stepId);
                const stepClass = isFail ? "trace-step--fail" : (isSuccess ? "trace-step--ok" : "");

                html += `<div class="trace-step ${stepClass}">`;
                html += `<span class="trace-step__number">[${i + 1}]</span> `;
                html += `<span class="trace-step__component">${this.esc(stepId)}</span>`;
                html += ` — ${isSuccess ? "✅ PASS" : (isFail ? "❌ FAIL" : "⏳")}`;
                html += `</div>`;
            }
        }

        // Append messages
        if (data.messages && data.messages.length > 0) {
            html += `<div class="trace-step" style="margin-top:12px;border-top:1px solid #34495e;padding-top:8px;">`;
            html += `<span class="trace-step__component">MESSAGES:</span>`;
            html += `</div>`;
            for (const msg of data.messages) {
                html += `<div class="trace-step"><span class="trace-step__detail">${this.esc(msg)}</span></div>`;
            }
        }

        // For audit replay, also show raw trace if available
        if (data._raw && data._raw.execution_trace) {
            const trace = data._raw.execution_trace;
            if (Array.isArray(trace)) {
                html += `<div class="trace-step" style="margin-top:12px;border-top:1px solid #34495e;padding-top:8px;">`;
                html += `<span class="trace-step__component">DETAILED TRACE:</span>`;
                html += `</div>`;
                for (const event of trace) {
                    html += `<div class="trace-step">`;
                    html += `<span class="trace-step__number">[${event.step || "?"}]</span> `;
                    html += `<span class="trace-step__component">${this.esc(event.component || "")}</span>: `;
                    html += `<span class="trace-step__detail">${this.esc(event.action || "")}</span>`;
                    html += `</div>`;
                }
            }
        }

        if (steps.length === 0 && (!data.messages || data.messages.length === 0)) {
            html += `<div class="trace-step"><span class="trace-step__detail">No execution trace available.</span></div>`;
        }

        html += `</div>`;
        el.innerHTML = html;
    }

    /* ---------- Panel 9: Visual evidence ---------- */

    async renderVisualPanel(data) {
        // Check for available overlay images
        const overlayFiles = [];

        // Known overlay filenames from verified workflows
        const knownOverlays = [
            "grounding_water_overlay.png",
            "hero_reasoning_overlay.png",
        ];

        for (const filename of knownOverlays) {
            try {
                const resp = await fetch("/api/overlays/" + filename, { method: "HEAD" });
                if (resp.ok) overlayFiles.push(filename);
            } catch (e) {
                // Overlay not available
            }
        }

        if (overlayFiles.length === 0) return;

        this.showPanel("visual-panel");
        const el = document.getElementById("visual-content");
        if (!el) return;

        let html = "";
        for (const filename of overlayFiles) {
            const label = filename.replace(/_/g, " ").replace(".png", "").replace(".jpg", "");
            html += `<div style="margin-bottom:16px;">
                <h4>${this.esc(label)}</h4>
                <img class="visual-evidence-img" src="/api/overlays/${filename}"
                     alt="${this.esc(label)}" loading="lazy" />
            </div>`;
        }
        el.innerHTML = html;
    }

    /* ---------- Panel 10: Report ---------- */

    renderReportPanel(data, mode) {
        this.showPanel("report-panel");
        const el = document.getElementById("report-content");
        if (!el) return;

        let html = `<p>Download the execution report as a machine-readable JSON file.</p>`;

        html += `<button class="report-btn" onclick="app.downloadReport()">📥 Download JSON Report</button>`;

        if (data.audit_filename) {
            html += `<button class="report-btn" onclick="app.downloadAuditFile('${data.audit_filename}')" style="margin-left:8px;">
                📄 Download Audit File
            </button>`;
        }

        html += `<div style="margin-top:12px;color:#7f8c8d;font-size:0.85em;">
            Report contains: query, task type, evidence, confidence, verification,
            models/tools, execution trace, and timing. No private chain-of-thought is included.
        </div>`;

        el.innerHTML = html;
    }

    downloadReport() {
        if (!this.currentResult) {
            alert("No query result available.");
            return;
        }

        const report = {
            report_format: "SATQuery AI Execution Report",
            generated_at: new Date().toISOString(),
            mode: this.currentMode,
            query: this.currentResult.query,
            task_type: this.currentResult.task_type,
            status: this.currentResult.status,
            success: this.currentResult.success,
            evidence: this.currentResult.evidence || [],
            verification: this.currentResult.verification || {},
            selected_capabilities: this.currentResult.selected_capabilities || {},
            selected_models: this.currentResult.selected_models || {},
            execution_trace: {
                executed_steps: this.currentResult.executed_steps || [],
                successful_steps: this.currentResult.successful_steps || [],
                failed_steps: this.currentResult.failed_steps || [],
                execution_time_seconds: this.currentResult.execution_time_seconds || 0,
            },
            messages: this.currentResult.messages || [],
            provenance: {
                system: "SATQuery AI",
                problem_statement: "SIH26167",
                calibration_note: "Confidence values are uncalibrated.",
            },
        };

        const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `satquery_report_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    downloadAuditFile(filename) {
        window.open("/api/audits/" + encodeURIComponent(filename), "_blank");
    }

    /* ---------- Helpers ---------- */

    showPanel(id) {
        const el = document.getElementById(id);
        if (el) el.classList.remove("hidden");
    }

    statusBadgeClass(status) {
        if (!status) return "status-badge--failed";
        const s = status.toLowerCase();
        if (s === "verified") return "status-badge--verified";
        if (s === "completed") return "status-badge--verified";
        if (s === "low_confidence") return "status-badge--low-confidence";
        if (s === "failed" || s === "error") return "status-badge--failed";
        if (s === "abstain") return "status-badge--low-confidence";
        if (s === "processing") return "status-badge--processing";
        return "status-badge--live";
    }

    esc(str) {
        if (str === null || str === undefined) return "";
        const div = document.createElement("div");
        div.textContent = String(str);
        return div.innerHTML;
    }
}

/* ------------------------------------------------------------------ */
/*  Bootstrap                                                         */
/* ------------------------------------------------------------------ */

const app = new SATQueryApp();
document.addEventListener("DOMContentLoaded", () => app.init());
