# Remote-Sensing Adaptation Report
## SATQuery AI — SIH 2026 PS26167

**Status**: `remote_sensing_adapted = False`
**Adapter Role**: Evidence-grounded language answer generator

---

## 1. Base Model

| Parameter | Value |
|-----------|-------|
| **Base Model** | Qwen2-VL-2B-Instruct |
| **Snapshot** | `895c3a49bc3fa70a340399125c650a463535e71c` |
| **Source** | HuggingFace `Qwen/Qwen2-VL-2B-Instruct` |
| **Architecture** | Vision-Language Model (VLM) with visual encoder + language decoder |

---

## 2. Adaptation Method

| Parameter | Value |
|-----------|-------|
| **Method** | PEFT LoRA (v0.20.0) |
| **Task Type** | CAUSAL_LM |
| **LoRA Rank (r)** | 8 |
| **LoRA Alpha** | 16 |
| **LoRA Dropout** | 0.05 |
| **Target Modules** | `q_proj`, `v_proj` (language decoder only) |
| **Visual Encoder** | **FROZEN** (`visual_encoder_trainable: false`) |
| **Trainable Parameters** | 1,089,536 (0.049% of total) |
| **Bias** | none |

**Critical Observation**: Only the language decoder's query and value projections were fine-tuned.
The visual encoder received **zero gradient pressure** during training. This means the adapter
learned to process text tokens, not visual features.

---

## 3. Dataset

| Parameter | Value |
|-----------|-------|
| **Source** | SpaceNet 4 building footprints |
| **AOI** | Single patch: `Atlanta_743501_3721539` |
| **Train Examples** | 628 |
| **Validation Examples** | 156 |
| **Total** | 784 |
| **Generation Method** | Synthetic template QA from building mask pixel fractions |
| **Development Only** | Yes |
| **Independent Benchmark** | No |

**Not Used**:
- BigEarthNet (mentioned only in requirements, zero code references)
- RSVQA academic benchmark (name reused internally for own synthetic dataset)
- VRSBench
- CDVQA

**Geographic Coverage**: Single scene from Atlanta, Georgia, USA.
No varied sensors, terrains, atmospheric conditions, or geographic regions.

---

## 4. Training Evidence

| Metric | Value |
|--------|-------|
| **Epochs** | 1 |
| **Batch Size** | 1 |
| **Gradient Accumulation** | 4 |
| **Learning Rate** | 1e-4 |
| **Max Sequence Length** | 512 |
| **Initial Validation Loss** | 1.1696 |
| **Final Validation Loss** | 0.0019 |
| **Improvement** | 99.84% |

Training scripts located at:
- `src/training/train_qwen_rs_vqa_evidence_grounded.py`
- `src/training/build_evidence_grounded_rs_vqa_dataset.py`
- `src/training/rs_vqa_evidence_grounded_data.py`
- `src/training/rs_vqa_evidence_grounded_collator.py`

---

## 5. Checkpoint

| Parameter | Value |
|-----------|-------|
| **Path** | `outputs/checkpoints/qwen2vl_rs_vqa_evidence_grounded_dev/` |
| **Files** | `adapter_config.json`, `adapter_model.safetensors`, `training_metadata.json` |
| **Phase** | 9.4H.8B.6 |
| **Warning** | Included in `training_metadata.json`: "This adapter is a development evidence-grounded adaptation. Validation loss improvement does not establish visual reasoning or production readiness." |

---

## 6. Inference Loading Path

```
VqaSpecialist.__init__()
  → self.adapter_path = "outputs/checkpoints/qwen2vl_rs_vqa_evidence_grounded_dev"

VqaSpecialist._load_model()
  → AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
  → PeftModel.from_pretrained(base_model, adapter_path)

VqaSpecialist.infer()
  → Formats prompt with REMOTE-SENSING EVIDENCE block (if available)
  → Passes image + text to Qwen2-VL
  → Returns Evidence with confidence, provenance, metadata
```

Code location: `src/executor/vqa_specialist.py` (652 lines)

---

## 7. Evaluation Evidence

### 7.1 Normal Accuracy (on own synthetic validation set)
| Metric | Value |
|--------|-------|
| Exact Match Accuracy | 98.08% (153/156) |
| Semantic Match Accuracy | 75.00% (117/156) |
| Mean Latency | 0.879s (GPU), ~228s (CPU) |

### 7.2 Evidence Grounding Sensitivity
| Metric | Value |
|--------|-------|
| Test | Remove/perturb structured evidence text from prompt |
| Examples | 24 |
| Predictions Changed | 24/24 (100%) |
| Semantic Status Changes | 22/24 (91.7%) |
| **Conclusion** | Adapter IS responsive to text evidence changes |

### 7.3 Visual Sensitivity (Critical)
| Metric | Value |
|--------|-------|
| Test | Replace real satellite image with blank/noise image |
| Examples | 24 |
| Real vs Blank Changed | **0/24 (0.0%)** |
| Real vs Noise Changed | **0/24 (0.0%)** |
| **Conclusion** | **Adapter does NOT use visual information** |

### 7.4 Scientific Conclusion (from internal audit)

> **Supported Claim**: "The adapter functions as an evidence-grounded remote-sensing answer generator."
>
> **Unsupported Claim**: "The experiment does not establish independent visual remote-sensing reasoning or a genuinely vision-sensitive remote-sensing VLM adaptation."
>
> **`visual_rs_adaptation_established: false`**

Source: `outputs/checkpoints/qwen2vl_rs_vqa_evidence_grounded_dev/experiment_audit_9_4H_8B_9.json`

---

## 8. Limitations

1. **No Visual RS Reasoning**: The adapter produces identical outputs when satellite images are replaced with blank/noise, proving it does not process visual features for decision-making.

2. **Text Template Reformatter**: The adapter learned to extract numerical evidence from the structured text prompt and template it into natural language answers. This is a language skill, not a remote-sensing skill.

3. **Single AOI Overfitting**: All training data from one Atlanta scene patch. No geographic, atmospheric, or sensor diversity.

4. **Frozen Visual Encoder**: Zero gradient pressure on visual features during training. The LoRA targets (`q_proj`, `v_proj`) are language decoder components only.

5. **No Independent Benchmark**: Not evaluated on BigEarthNet, RSVQA, VRSBench, CDVQA, or any public RS benchmark.

6. **Development Only**: The checkpoint's own metadata explicitly states `development_only: true`.

---

## 9. Decision

```
remote_sensing_adapted = False
```

**Rationale**: The internal counterfactual audit (`experiment_audit_9_4H_8B_9.json`) conclusively
proved 0% visual sensitivity. The adapter functions as an evidence-grounded language reformatter.
This is a legitimate and useful capability for SATQuery's evidence-conditioned pipeline, but it is
NOT remote-sensing visual adaptation.

**What the adapter IS**: An evidence-grounded answer generator that reformats structured
remote-sensing evidence (NDWI values, building counts, area measurements) into natural language
responses. This is valuable in the SATQuery pipeline where upstream deterministic specialists
(FloodSpecialist, BuildingDetectionSpecialist, ChangeSpecialist) produce the actual RS analysis
and the VQA specialist synthesizes their output.

**Code locations where claim was corrected**:
- `src/orchestration/orchestrator.py` line 155: `remote_sensing_adapted: False`
- `src/orchestration/orchestrator.py` line 116: `remote_sensing_adapted: False`
- `src/executor/vqa_specialist.py` line 619: `remote_sensing_adapted: False`

---

## 10. Required Research for Future `remote_sensing_adapted = True`

From the internal audit's `next_required_research`:

1. Introduce training targets that require image discrimination
2. Add positive/negative visual counterfactual pairs
3. Add image-only evaluation with evidence removed
4. Add evidence-image consistency tests
5. Evaluate on scenes independent from the single development AOI
6. Only then reconsider `remote_sensing_adapted = True`
