from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.training.building_dataset import SpaceNetBuildingDataset
from src.training.building_loss import BuildingSegmentationLoss
from src.training.building_model import BuildingUNet
from src.training.metrics import binary_segmentation_metrics


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(
    dataset: SpaceNetBuildingDataset,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """
    Create a deterministic development train/validation split.

    IMPORTANT:
    This is a patch-level split from one scene and is therefore
    preliminary. It is not a scene-level generalization benchmark.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    total = len(dataset)

    if total < 2:
        raise ValueError(
            "Dataset must contain at least two samples."
        )

    indices = list(range(total))

    rng = random.Random(seed)
    rng.shuffle(indices)

    validation_size = max(
        1,
        int(round(total * validation_fraction)),
    )

    train_indices = indices[:-validation_size]
    validation_indices = indices[-validation_size:]

    if not train_indices or not validation_indices:
        raise ValueError(
            "Train/validation split produced an empty partition."
        )

    return (
        Subset(dataset, train_indices),
        Subset(dataset, validation_indices),
    )


def _run_epoch(
    model: BuildingUNet,
    loader: DataLoader,
    loss_fn: BuildingSegmentationLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    batch_count = 0

    metric_totals = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    for images, masks in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)

            loss = loss_fn(
                logits,
                masks,
            )

            if training:
                loss.backward()
                optimizer.step()

        total_loss += float(
            loss.detach().item()
        )

        metrics = binary_segmentation_metrics(
            logits.detach(),
            masks.detach(),
        )

        for name in metric_totals:
            metric_totals[name] += metrics[name]

        batch_count += 1

    if batch_count == 0:
        raise RuntimeError(
            "DataLoader produced zero batches."
        )

    result = {
        "loss": total_loss / batch_count,
    }

    for name in metric_totals:
        result[name] = (
            metric_totals[name] / batch_count
        )

    return result



def create_building_aware_sampler(
    dataset: Subset,
    positive_weight: float = 2.0,
    negative_weight: float = 1.0,
) -> WeightedRandomSampler:
    """
    Create a weighted sampler that gives building-containing
    training patches higher sampling probability.

    The sampler only affects training data. Validation data
    must never use this sampler.

    Args:
        dataset:
            Training subset of SpaceNetBuildingDataset.
        positive_weight:
            Relative sampling weight for patches containing
            at least one building pixel.
        negative_weight:
            Sampling weight for empty patches.

    Returns:
        WeightedRandomSampler configured for the dataset.
    """
    if positive_weight <= 0:
        raise ValueError("positive_weight must be > 0")

    if negative_weight <= 0:
        raise ValueError("negative_weight must be > 0")

    weights = []

    for index in dataset.indices:
        _, mask = dataset.dataset[index]

        has_building = bool(mask.sum().item() > 0)

        if has_building:
            weights.append(float(positive_weight))
        else:
            weights.append(float(negative_weight))

    if not weights:
        raise ValueError("Cannot create sampler for empty dataset")

    generator = torch.Generator()
    generator.manual_seed(42)

    return WeightedRandomSampler(
        weights=torch.tensor(
            weights,
            dtype=torch.double,
        ),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def train_building_model(
    manifest_path: str,
    output_checkpoint: str,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train the lightweight SpaceNet building segmentation model.

    This function intentionally records the best validation IoU
    checkpoint rather than merely saving the final epoch.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive."
        )

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = SpaceNetBuildingDataset(
        manifest_path
    )

    train_dataset, validation_dataset = split_dataset(
        dataset,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = BuildingUNet(
        in_channels=4,
        base_channels=16,
    ).to(device)

    loss_fn = BuildingSegmentationLoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    output_path = Path(output_checkpoint)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_iou = -1.0
    best_epoch = 0

    history: list[dict[str, Any]] = []

    print("=== Building Model Training ===")
    print("Device:", device)
    print("Total samples:", len(dataset))
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))
    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
        )

        validation_metrics = _run_epoch(
            model=model,
            loader=validation_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=None,
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }

        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_iou={train_metrics['iou']:.4f} | "
            f"val_loss={validation_metrics['loss']:.4f} | "
            f"val_iou={validation_metrics['iou']:.4f} | "
            f"val_dice={validation_metrics['dice']:.4f}"
        )

        if validation_metrics["iou"] > best_iou:
            best_iou = validation_metrics["iou"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "BuildingUNet",
                    "in_channels": 4,
                    "base_channels": 16,
                    "epoch": epoch,
                    "validation_iou": best_iou,
                    "validation_metrics": validation_metrics,
                    "seed": seed,
                },
                output_path,
            )

            print(
                f"  Saved best checkpoint: {output_path}"
            )

    return {
        "device": str(device),
        "total_samples": len(dataset),
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "best_epoch": best_epoch,
        "best_validation_iou": best_iou,
        "checkpoint": str(output_path),
        "history": history,
    }
