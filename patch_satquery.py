import sys

with open("static/js/satquery.js", "r", encoding="utf-8") as f:
    content = f.read()

import re

# We will replace the following parts:
# 1. renderMapPanel (which includes extractGeometriesFromRaw, initMap, clearMap, etc)
# 2. renderEvidencePanel

map_logic_start = content.find("    renderMapPanel(data) {")
confidence_panel_start = content.find("    /* ---------- Panel 7: Confidence ---------- */")
clearMap_start = content.find("    clearMap() {")

if map_logic_start == -1 or confidence_panel_start == -1 or clearMap_start == -1:
    print("Could not find boundaries")
    sys.exit(1)

new_map_and_evidence_logic = """
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

        if (data._raw) {
            this.extractGeometriesFromRaw(data._raw, geometries);
        }

        if (geometries.length === 0) {
            this.showPanel("map-panel");
            this.initMap();
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
            bi_temporal_water_change: "#f39c12",
            sar_analysis: "#f1c40f",
            buffer: "#9b59b6",
            intersection: "#2ecc71",
            area: "#1abc9c",
            vqa: "#3498db",
        };

        const layerGroups = {
            "Water": L.layerGroup(),
            "Buildings": L.layerGroup(),
            "Change": L.layerGroup(),
            "SAR": L.layerGroup(),
            "Optical": L.layerGroup(),
            "Other": L.layerGroup()
        };

        const allBounds = [];
        this.mapFeaturesByEvidenceId = {};

        for (const item of geometries) {
            try {
                const color = colors[item.task] || "#3498db";
                const layer = L.geoJSON(item.geometry, {
                    style: {
                        color: color,
                        weight: 2,
                        fillColor: color,
                        fillOpacity: 0.4,
                    },
                    onEachFeature: (feature, l) => {
                        l.bindPopup(
                            `<strong>${this.esc(item.task)}</strong><br>` +
                            `<code>${this.esc(item.evidence_id)}</code><br>` +
                            `<button style="margin-top:5px;cursor:pointer;" onclick="app.highlightEvidence('${this.esc(item.evidence_id)}')">View Evidence</button>`
                        );
                        if (item.evidence_id) {
                            if (!this.mapFeaturesByEvidenceId[item.evidence_id]) {
                                this.mapFeaturesByEvidenceId[item.evidence_id] = [];
                            }
                            this.mapFeaturesByEvidenceId[item.evidence_id].push(l);
                        }
                    },
                });

                // Determine group
                let groupName = "Other";
                if (item.task.includes("water") || item.task.includes("flood")) groupName = "Water";
                else if (item.task.includes("building")) groupName = "Buildings";
                else if (item.task.includes("temporal") || item.task.includes("change")) groupName = "Change";
                else if (item.task.includes("sar")) groupName = "SAR";
                else if (item.task.includes("optical") || item.task === "vqa") groupName = "Optical";

                layerGroups[groupName].addLayer(layer);
                this.mapLayers.push(layer);

                const bounds = layer.getBounds();
                if (bounds.isValid()) {
                    allBounds.push(bounds);
                }
            } catch (e) {
                console.warn("Failed to render geometry:", e);
            }
        }

        const activeOverlays = {};
        for (const [name, group] of Object.entries(layerGroups)) {
            if (group.getLayers().length > 0) {
                activeOverlays[name] = group;
                group.addTo(this.map);
            }
        }

        if (Object.keys(activeOverlays).length > 0) {
            this.layerControl = L.control.layers(null, activeOverlays, { collapsed: false }).addTo(this.map);
        }

        if (allBounds.length > 0) {
            let combinedBounds = allBounds[0];
            for (let i = 1; i < allBounds.length; i++) {
                combinedBounds.extend(allBounds[i]);
            }
            this.map.fitBounds(combinedBounds, { padding: [20, 20] });
        }
    }

    extractGeometriesFromRaw(raw, geometries) {
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
                if (key === "geometry") continue;
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
            attribution: '&copy; OpenStreetMap',
            maxZoom: 19,
        }).addTo(this.map);
    }

    clearMap() {
        if (!this.map) return;
        for (const layer of this.mapLayers) {
            this.map.removeLayer(layer);
        }
        this.mapLayers = [];
        if (this.layerControl) {
            this.map.removeControl(this.layerControl);
            this.layerControl = null;
        }
        this.mapFeaturesByEvidenceId = {};
    }

    highlightMapFeature(evidenceId) {
        if (!this.map || !this.mapFeaturesByEvidenceId || !this.mapFeaturesByEvidenceId[evidenceId]) return;
        const layers = this.mapFeaturesByEvidenceId[evidenceId];
        
        const bounds = L.latLngBounds();
        layers.forEach(l => {
            if (l.getBounds) bounds.extend(l.getBounds());
            if (l.openPopup) l.openPopup();
            
            // Temporary highlight effect
            const origColor = l.options.color;
            if (l.setStyle) {
                l.setStyle({ color: '#ff0', weight: 4 });
                setTimeout(() => {
                    if (this.map.hasLayer(l)) l.setStyle({ color: origColor, weight: 2 });
                }, 2000);
            }
        });
        if (bounds.isValid()) this.map.fitBounds(bounds, { padding: [20, 20], maxZoom: 16 });
    }

    highlightEvidence(evidenceId) {
        this.showPanel("evidence-panel");
        const card = document.getElementById(`ev-card-${evidenceId}`);
        if (card) {
            card.scrollIntoView({ behavior: 'smooth', block: 'center' });
            card.style.transition = "background-color 0.5s";
            card.style.backgroundColor = "#fff3cd";
            setTimeout(() => {
                card.style.backgroundColor = "";
            }, 2000);
        }
    }

    /* ---------- Panel 6: Evidence ---------- */

    renderEvidencePanel(data) {
        if (!data.evidence || data.evidence.length === 0) return;
        this.showPanel("evidence-panel");
        const el = document.getElementById("evidence-content");
        if (!el) return;

        let html = `<p style="color:#7f8c8d;margin-bottom:12px;">${data.evidence.length} evidence object(s)</p>`;

        for (const ev of data.evidence) {
            const hasGeometry = ev.geometry && ev.geometry.type;
            const clickHandler = hasGeometry ? `onclick="app.highlightMapFeature('${this.esc(ev.evidence_id)}')" style="cursor:pointer;" title="Click to view on map"` : "";
            
            html += `<div class="evidence-card" id="ev-card-${this.esc(ev.evidence_id || '')}" ${clickHandler}>`;
            html += `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">`;
            html += `<code class="evidence-card__id">${this.esc(ev.evidence_id || "?")}</code>`;
            if (hasGeometry) {
                html += `<span style="font-size:0.8em;color:#2980b9;">&#x1F5FA;&#xFE0F; Map feature</span>`;
            }
            html += `</div>`;
            html += `<div class="evidence-card__field"><span class="field-label">Task:</span> ${this.esc(ev.task || "-")}</div>`;
            html += `<div class="evidence-card__field"><span class="field-label">Model:</span> ${this.esc(ev.model || "-")}</div>`;
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

            if (hasGeometry) {
                html += `<div class="evidence-card__field"><span class="field-label">Geometry:</span> ${this.esc(ev.geometry.type)}</div>`;
            }

            html += `</div>`;
        }

        el.innerHTML = html;
    }

"""

new_content = content[:map_logic_start] + new_map_and_evidence_logic + content[confidence_panel_start:]
with open("static/js/satquery.js", "w", encoding="utf-8") as f:
    f.write(new_content)

print("Replaced map and evidence rendering logic successfully.")
