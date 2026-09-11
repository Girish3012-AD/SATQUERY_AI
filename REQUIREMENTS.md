# SATQuery AI - PS 26167 Requirements

## Problem Statement

Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries.

## Mandatory Capabilities

### R01 - Input Compatibility Checking
The system must validate supported remote-sensing imagery before processing.

Validation must consider:
- file format
- dimensions
- bands
- dtype
- CRS
- bounds
- transform
- resolution
- nodata
- metadata
- modality
- temporal relationships
- optical/SAR relationships

Invalid or incompatible inputs must be rejected with an actionable reason.

### R02 - Remote-Sensing VLM Adaptation
At least one visual/VLM component must be genuinely adapted using BigEarthNet.txt or suitable open-source remote-sensing training data.

Loading a dataset alone does not constitute adaptation.

The adapted checkpoint/adapter must be saved and subsequently used during inference.

### R03 - Single-Image VQA
The system must support natural-language questions about a single remote-sensing image.

Workflow:

Input Validation
-> TaskSpec
-> Evidence Planning
-> VQA Specialist
-> Evidence
-> Verification
-> Answer

### R04 - Captioning OR Text-Guided Grounding
The system must provide at least one of:

1. Image captioning

OR

2. Text-guided grounding

If grounding is selected, actual visual region evidence must be produced.

### R05 - Bi-Temporal Change Understanding
The system must support:

1. Bi-temporal change description

OR

2. Change-based VQA

A change map alone is insufficient.

The system must produce textual change understanding.

### R06 - Optical/Multispectral + SAR Pair Analysis
The system must analyze paired optical/multispectral and SAR imagery.

The workflow must use actual compatible paired inputs.

No fictional fusion capability may be claimed.

### R07 - Agentic Orchestration
The system must automatically select, sequence and execute specialist models/tools according to the user query and input.

The final agentic demonstration must not depend on manual specialist selection.

### R08 - Evidence-Grounded Results
Results must be supported by evidence produced by:
- specialist models
- deterministic GIS operations
- visual evidence
- provenance

No decorative or fabricated evidence is permitted.

### R09 - Confidence and Verification
The system must verify:
- spatial consistency
- temporal consistency
- geometry validity
- evidence completeness
- model confidence
- agreement/consistency
- constraint satisfaction

The system must support:
- PASS
- LOW CONFIDENCE
- CONFLICT
- ABSTAIN

### R10 - Deterministic GIS
Exact geographic calculations must be performed by deterministic GIS tools.

Required capabilities include:
- area
- distance
- buffer
- intersection
- containment
- spatial relationships
- CRS/unit handling

The LLM must not perform exact geographic calculations.

### R11 - Auditable Execution Trace
Every completed request must produce an execution summary containing observable facts such as:
- task
- input relationship
- specialist/model/tool
- permitted parameters
- status
- confidence/result
- evidence IDs
- timing

Private chain-of-thought must never be exposed.

### R12 - Visual Evidence
The system must be able to display evidence such as:
- bounding boxes
- masks
- change regions
- buffers
- intersections
- relevant regions

Every displayed measurement or region must have provenance.

### R13 - Downloadable Report
The system must generate a downloadable report containing:
- query
- input summary
- answer
- evidence
- confidence
- verification
- model/tool names
- parameters
- execution trace
- visual artifacts

Secrets must not be included.

### R14 - Web GUI
The system must provide a usable web interface supporting:
- image upload
- compatibility status
- natural-language query
- execution progress
- answer
- confidence
- evidence
- visual overlays
- report download

### R15 - Public Benchmark Evaluation
The system must be evaluated using prescribed public benchmark resources where applicable.

Required resources include:
- VRSBench
- RSVQA
- CDVQA
- BigEarthNet.txt where appropriate

Official splits and procedures must be respected.

No train/test leakage is permitted.

### R16 - Cartosat-2S + RISAT Readiness
The architecture must remain ready for ISRO/SAC Cartosat-2S + RISAT evaluation data when available.

The architecture must not be hard-coded only for Sentinel-1/Sentinel-2.

No official result may be claimed without actual authorized data.

# Golden Queries

## G01
What type of land cover is visible?

## G02
Are there visible roads or built-up areas?

## G03
Describe the main scene and important objects.

## G04
Where are the buildings?

## G05
What changed between the two dates?

## G06
Did built-up area increase between the two images?

## G07
What features are jointly supported by the optical and SAR observations?

## G08
What is the area of the detected region?

## G09
Find newly constructed buildings within 500 meters of flooded areas.

## G10
Analyze this unsupported or incompatible file.

# Non-Negotiable Rules

1. No hallucinated capabilities.
2. No fake adaptation.
3. No fake evidence.
4. No LLM arithmetic for exact geographic measurements.
5. No manual specialist selection in the final agentic demonstration.
6. No private chain-of-thought in reports.
7. No benchmark data leakage.
8. No Sentinel-only hard-coding.
9. No model freeze before evaluation.
10. Every major failure mode must have a negative test and user-readable response.

# Definition of Implemented

A component is considered implemented only when:

- Real dependency/model/checkpoint is reproducibly available.
- Real input is validated and accepted.
- Real inference/tool execution completes.
- Output is converted to the common schema.
- Required evidence is produced.
- Verification/confidence behavior is defined.
- At least one automated test passes.
- The specialist is reachable through the registry/router.
- The component participates in an end-to-end workflow.
- Limitations and required data are documented.
