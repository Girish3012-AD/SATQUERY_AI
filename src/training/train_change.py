from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.training.change_dataset import ChangePairDataset
from src.training.change_loss import ChangeLoss
from src.training.change_metrics import binary_change_metrics
from src.training.change_model import ChangeUNet


SEED = 20260913
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DATASET_ROOT = (
    "data/remote_sensing/change_dev"
)

CHECKPOINT = (
    "outputs/checkpoints/"
    "change_unet_synthetic_dev.pt"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_indices(
    dataset_size: int,
    validation_fraction: float = 0.2,
) -> tuple[list[int], list[int]]:

    indices = np.arange(dataset_size)

    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)

    split = int(
        dataset_size * (1.0 - validation_fraction)
    )

    train_indices = indices[:split].tolist()
    validation_indices = indices[split:].tolist()

    return train_indices, validation_indices


def run_epoch(
    model: ChangeUNet,
    loader: DataLoader,
    loss_fn: ChangeLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, dict[str, float]]:

    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    batches = 0

    metric_values = []

    for batch in loader:

        before = batch["before"].to(
            device,
            non_blocking=True,
        )

        after = batch["after"].to(
            device,
            non_blocking=True,
        )

        target = batch["mask"].to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):

            logits = model(
                before,
                after,
            )

            loss = loss_fn(
                logits,
                target,
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

        total_loss += float(loss.item())
        batches += 1

        with torch.no_grad():
            metric_values.append(
                binary_change_metrics(
                    logits.detach(),
                    target,
                )
            )

    mean_metrics = {}

    if metric_values:
        keys = metric_values[0].keys()

        for key in keys:
            mean_metrics[key] = float(
                np.mean([
                    metrics[key]
                    for metrics in metric_values
                ])
            )

    return (
        total_loss / max(batches, 1),
        mean_metrics,
    )


def train() -> dict:
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=== CHANGEUNET DEVELOPMENT TRAINING ===")
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    dataset = ChangePairDataset(
        DATASET_ROOT
    )

    train_indices, val_indices = split_indices(
        len(dataset)
    )

    train_dataset = Subset(
        dataset,
        train_indices,
    )

    val_dataset = Subset(
        dataset,
        val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = ChangeUNet(
        in_channels=4,
        base_channels=16,
    ).to(device)

    loss_fn = ChangeLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Dataset:", len(dataset))
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Parameters:", parameter_count)

    best_iou = -1.0
    best_epoch = -1
    best_metrics = None

    checkpoint_path = Path(CHECKPOINT)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )

        with torch.no_grad():
            val_loss, val_metrics = run_epoch(
                model,
                val_loader,
                loss_fn,
                None,
                device,
            )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"precision={val_metrics['precision']:.4f} | "
            f"recall={val_metrics['recall']:.4f}"
        )

        if val_metrics["iou"] > best_iou:

            best_iou = val_metrics["iou"]
            best_epoch = epoch
            best_metrics = val_metrics

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model": "ChangeUNet",
                    "in_channels": 4,
                    "base_channels": 16,
                    "epoch": epoch,
                    "val_iou": best_iou,
                    "val_metrics": val_metrics,
                    "seed": SEED,
                    "dataset": "SpaceNet4-derived synthetic development data",
                    "development_only": True,
                    "label_method": (
                        "controlled_synthetic_spectral_change"
                    ),
                    "training_scope": (
                        "single-scene development split"
                    ),
                },
                checkpoint_path,
            )

    print("\n=== TRAINING COMPLETE ===")
    print("Best epoch:", best_epoch)
    print("Best validation IoU:", best_iou)
    print("Best validation metrics:", best_metrics)
    print("Checkpoint:", checkpoint_path.resolve())
    print(
        "Checkpoint size:",
        checkpoint_path.stat().st_size,
        "bytes",
    )

    return {
        "best_epoch": best_epoch,
        "best_iou": best_iou,
        "checkpoint": str(
            checkpoint_path.resolve()
        ),
    }


if __name__ == "__main__":
    train()
