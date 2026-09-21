# SIH 2026 Recommended Presentation Queries

This document details the 5 recommended demonstration queries for SATQuery AI, explaining their technical execution, specialists involved, GIS operations, verifier behavior, and scientifically safe interpretation.

---

## Query A: Single-Image Vision-Language VQA

**Exact Query**:  
`"Describe the land-cover and major objects visible in this image."`

- **User Intent**: Solicit a natural language visual interpretation of a remote sensing scene.
- **Task Type**: `vqa`
- **Specialists Involved**: `VqaSpecialist` (`Qwen2-VL-2B-Instruct` base + LoRA adapter `qwen2vl_rs_vqa_evidence_grounded_dev`).
- **Evidence Generated**: `Evidence` object (`task="vqa"`, `model="Qwen2-VL-2B-Instruct+LoRA"`, `modality="optical"`).
- **GIS Operation**: Image normalization, 512x512 downsampling for vision processor.
- **Verifier Behavior**: Validates presence of answer text and non-empty result payload; returns `status="verified"` if answer payload exists.
- **Scientifically Safe Interpretation**: Model-generated text description of visual land-cover features. Should be interpreted as a vision-language model output, not a high-precision survey document.

---

## Query B: Single-Image Spatial Grounding

**Exact Query**:  
`"Highlight the water body referred to in the image."`

- **User Intent**: Identify and extract the exact geographic boundaries of a water feature mentioned in natural language.
- **Task Type**: `specialized_analysis`
- **Specialists Involved**: `WaterSpecialist`
- **Evidence Generated**: `Evidence` object (`task="water_detection"`, `sensor="MSI"`, `modality="optical"`, contains WGS84 GeoJSON polygon).
- **GIS Operation**: Calculates Normalized Difference Water Index $\text{NDWI} = (B03 - B08) / (B03 + B08)$, thresholds at $\text{NDWI} \ge 0.0$, applies morphological opening, and vectorizes binary raster mask into Shapely polygons.
- **Verifier Behavior**: Validates existence of GeoJSON polygon geometry and EPSG coordinate metadata; returns `status="verified"`.
- **Scientifically Safe Interpretation**: Identifies **potential water candidates**. NDWI $\ge 0.0$ highlights high-moisture/surface-water features; it is not claimed as "confirmed flooding" without multi-temporal baseline comparison.

---

## Query C: Bi-Temporal Change Detection

**Exact Query**:  
`"Show spectral changes between the 2023 and 2024 Sentinel-2 observations."`

- **User Intent**: Detect surface cover alterations across two distinct observation dates.
- **Task Type**: `temporal_analysis`
- **Specialists Involved**: `TemporalChangeSpecialist`
- **Evidence Generated**: `Evidence` object (`task="temporal_analysis"`, `t1_timestamp="2023-..."`, `t2_timestamp="2024-..."`, GeoJSON change polygons).
- **GIS Operation**: Re-projects $T_1$ and $T_2$ rasters to common grid (`EPSG:32645`), computes bialgebraic band ratio difference, thresholds change intensity, and vectorizes bounding polygons.
- **Verifier Behavior**: Checks that both $T_1$ and $T_2$ ISO timestamps exist and geometry is non-empty; returns `status="verified"`.
- **Scientifically Safe Interpretation**: Detects **candidate change areas** (spectral difference). Does not claim confirmed building construction or deforestation without high-resolution optical/LIDAR verification.

---

## Query D: Multimodal Optical + SAR Reasoning

**Exact Query**:  
`"Analyze the area using both optical and SAR evidence."`

- **User Intent**: Cross-verify surface conditions using complementary optical reflectance and Synthetic Aperture Radar (SAR) backscatter.
- **Task Type**: `multimodal_analysis`
- **Specialists Involved**: `MultimodalSpecialist`, `WaterSpecialist`, `SARSpecialist`
- **Evidence Generated**: Dual `Evidence` objects (`modality="optical"` for Sentinel-2 MSI and `modality="sar"` for Sentinel-1 C-SAR VV/VH backscatter).
- **GIS Operation**: Co-registers Sentinel-1 SAR VV backscatter intensity ($\sigma^0$) with Sentinel-2 NDWI raster. Detects specular reflection (low SAR backscatter) over optical water candidate areas.
- **Verifier Behavior**: Cross-validates optical and SAR evidence IDs; returns `status="verified"`.
- **Scientifically Safe Interpretation**: Multi-sensor evidence fusion. Optical reflectance and SAR backscatter cross-confirmation increases confidence in surface water presence.

---

## Query E: Hero Multi-Step Geographic Reasoning

**Exact Query**:  
`"Find newly constructed buildings within 500 m of flooded areas."`

- **User Intent**: Perform complex spatial-temporal reasoning combining temporal change, water candidate detection, building candidate detection, and spatial proximity filtering.
- **Task Type**: `spatial_analysis`
- **Specialists Involved**: `WaterSpecialist`, `BuildingDetectionSpecialist`, `TemporalChangeSpecialist`
- **Evidence Generated**: Composite multi-evidence set (`E-WATER-001`, `E-BUILDING-001`, `E-CHANGE-001`, `E-BUFFER-001`, `E-INTERSECTION-001`).
- **GIS Operation**:
  1. `WaterSpecialist`: Computes NDWI water candidate mask.
  2. `BuildingDetectionSpecialist`: Runs UNet inference to extract candidate building polygons.
  3. `TemporalChangeSpecialist`: Identifies spectral change areas between $T_1$ and $T_2$.
  4. GIS Spatial Buffer: Generates a 500 m spatial buffer geometry (`shapely.Buffer()`) around water candidate polygons.
  5. GIS Intersection: Performs spatial intersection (`shapely.Intersection()`) between candidate building polygons, temporal change areas, and the 500 m buffer.
- **Verifier Behavior**: Verifies rule dependency graph: ensures buffer geometry is derived from water evidence and building candidates lie within the 500 m buffer. Returns `status="verified"`.
- **Scientifically Safe Interpretation**: Identifies **candidate building structures** located within a 500 m spatial buffer of **water candidate areas** exhibiting **spectral change**. It establishes spatial association, not causal flood damage or legally confirmed construction.
