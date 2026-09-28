# Remote-Sensing VLM Adaptation Blocker

## Exact Missing Dependencies / Resources
- **Missing Dataset:** No substantial or labeled remote-sensing specific visual QA dataset (e.g., RSVQA, RSICD, generic EO pretraining images) exists locally in this repository for fine-tuning.
- **Compute Constraints:** The current local environment lacks the GPU VRAM capacity necessary to execute LoRA/PEFT fine-tuning on a 2B+ parameter Vision Language Model (e.g., Qwen2-VL-2B-Instruct) within a reasonable timeframe and memory footprint.
- **Model Checkpoints:** No fine-tuned adapter weights were successfully completed or verified against counterfactual evaluation.

## What Is Already Available
- **Base Model Loading:** The codebase contains a `vqa_specialist.py` pipeline capable of loading `Qwen/Qwen2-VL-2B-Instruct` and passing multimodal text/image tensors.
- **Evaluation Framework:** `test_multimodal_evidence_alignment.py` and `REMOTE_SENSING_ADAPTATION_REPORT.md` (from previous audits) demonstrate an established pipeline to measure visual sensitivity via visual counterfactuals (where it previously scored 0/24 changed predictions).

## What Experiment Remains
A full genuine remote-sensing LoRA adaptation experiment:
1. Initialize PEFT/LoRA wrapper around the Qwen2-VL-2B-Instruct base model.
2. Prepare a real remote-sensing image/text dataset and split into train/val.
3. Train the adapter weights targeting visual embedding interpretation to break the model's reliance on prior text biases.
4. Evaluate the held-out validation set.
5. Execute the visual counterfactual suite to verify that modifying the image actually modifies the VLM's generated answer.

## Exact Command to Execute Later
Once datasets and GPU compute are available:
```bash
# Example training script that would need to be created/executed
python scripts/train_rs_vqa_lora.py \
    --model_name "Qwen/Qwen2-VL-2B-Instruct" \
    --dataset_path "data/rsvqa_lr" \
    --output_dir "outputs/checkpoints/qwen2_vl_rs_adapter" \
    --epochs 3 --batch_size 4 --fp16
```

## Exact Acceptance Criteria
1. The adapter successfully loads into `VQASpecialist`.
2. The visual counterfactual evaluation shows a statistically significant number of changed predictions (e.g., > 12/24) when the visual input is altered.
3. The evaluation proves the model uses visual evidence rather than language priors, proving *genuine* visual sensitivity.
