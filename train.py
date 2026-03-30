# -*- coding: utf-8 -*-
"""HHANet training script with full pipeline: train / validate / test."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from hhanet import build_hhanet
from dataset import SegmentationDataset, get_train_transform, get_val_transform
from utils import (
    AverageMeter,
    BceDiceLoss,
    MetricTracker,
    count_params,
    dice_coef,
    get_logger,
    iou_score,
    set_seed,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HHANet for medical image segmentation")
    # model
    p.add_argument("--variant", type=str, default="tiny", choices=["tiny", "base"])
    p.add_argument("--num_classes", type=int, default=1)
    p.add_argument("--in_ch", type=int, default=3)
    p.add_argument("--img_size", type=int, default=256)
    # data
    p.add_argument("--dataset_dir", type=str, required=True, help="Root directory of datasets")
    p.add_argument("--dataset_name", type=str, required=True,
                    help="Dataset name (ACDC, EchoNet-Dynamic, Fetal_HC, Synapse, FIVES, Kvasir-SEG, tn3k, etc.)")
    # training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # checkpoint
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--pretrained", type=str, default=None, help="Path to pretrained weights")
    p.add_argument("--legacy_ckpt", action="store_true", help="Remap old HappyNet keys when loading pretrained")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Train / Validate / Test
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    loss_meter = AverageMeter()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device) if masks.ndim == 3 else masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")
    return loss_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    phase: str = "Val",
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]", leave=False)
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device) if masks.ndim == 3 else masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        loss_meter.update(loss.item(), images.size(0))
        dice_meter.update(dice_coef(preds, masks), images.size(0))
        iou_meter.update(iou_score(preds, masks), images.size(0))

    return {"loss": loss_meter.avg, "dice": dice_meter.avg, "miou": iou_meter.avg}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # directories
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, "log")
    logger = get_logger("hhanet_train", log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- data ----
    train_dataset = SegmentationDataset(
        args.dataset_dir, args.dataset_name, split="train",
        image_size=args.img_size, transform=get_train_transform(args.img_size),
    )
    val_dataset = SegmentationDataset(
        args.dataset_dir, args.dataset_name, split="val", image_size=args.img_size,
    )
    test_dataset = SegmentationDataset(
        args.dataset_dir, args.dataset_name, split="test", image_size=args.img_size,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    logger.info(f"Dataset: {args.dataset_name} | train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")

    # ---- model ----
    model = build_hhanet(
        variant=args.variant,
        num_classes=args.num_classes,
        in_ch=args.in_ch,
        pretrained=args.pretrained,
        legacy_ckpt=args.legacy_ckpt,
        img_size=args.img_size,
    )
    model = model.to(device)
    logger.info(f"HHANet-{args.variant.capitalize()} | params: {count_params(model) / 1e6:.2f}M")

    # ---- loss / optimizer / scheduler ----
    criterion = BceDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # ---- resume ----
    start_epoch = 1
    dice_tracker = MetricTracker(strategy="maximize")
    iou_tracker = MetricTracker(strategy="maximize")

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        if "best_dice" in ckpt:
            dice_tracker.best_metric = ckpt["best_dice"]
        if "best_miou" in ckpt:
            iou_tracker.best_metric = ckpt["best_miou"]
        logger.info(f"Resumed from epoch {ckpt['epoch']}")

    # ---- training loop ----
    logger.info("Start training")
    for epoch in range(start_epoch, args.epochs + 1):
        torch.cuda.empty_cache()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        metrics = evaluate(model, val_loader, criterion, device, epoch, phase="Val")
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch}/{args.epochs} | lr={lr_now:.2e} | "
            f"train_loss={train_loss:.4f} | val_loss={metrics['loss']:.4f} | "
            f"dice={metrics['dice']:.4f} | miou={metrics['miou']:.4f}"
        )

        # save checkpoint helper
        def _save(tag: str) -> None:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_dice": dice_tracker.best_metric,
                    "best_miou": iou_tracker.best_metric,
                    "loss": metrics["loss"],
                },
                os.path.join(args.save_dir, f"{tag}.pth"),
            )

        if dice_tracker(metrics["dice"], epoch):
            _save("best_dice")
            logger.info(f"  -> New best Dice: {metrics['dice']:.4f}")

        if iou_tracker(metrics["miou"], epoch):
            _save("best_miou")
            logger.info(f"  -> New best mIoU: {metrics['miou']:.4f}")

        _save("latest")

    # ---- test with best dice checkpoint ----
    best_path = os.path.join(args.save_dir, "best_dice.pth")
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate(model, test_loader, criterion, device, epoch=ckpt["epoch"], phase="Test")
        logger.info(
            f"Test (best_dice @ epoch {ckpt['epoch']}) | "
            f"loss={test_metrics['loss']:.4f} | dice={test_metrics['dice']:.4f} | miou={test_metrics['miou']:.4f}"
        )


if __name__ == "__main__":
    main()
