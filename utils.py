# -*- coding: utf-8 -*-
"""Training utilities: metrics, losses, logging, and helpers."""

from __future__ import annotations

import os
import math
import random
import logging
import logging.handlers
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def str2bool(v: str) -> bool:
    if v.lower() in ("true", "1"):
        return True
    elif v.lower() in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    info_name = os.path.join(log_dir, f"{name}.info.log")
    handler = logging.handlers.TimedRotatingFileHandler(
        info_name, when="D", encoding="utf-8",
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"),
    )
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------------------------------------------------------------------
# Metric tracker (best-so-far)
# ---------------------------------------------------------------------------

class MetricTracker:
    """Track best metric value across epochs.

    Args:
        strategy: ``"minimize"`` or ``"maximize"``.
    """

    def __init__(self, strategy: str = "minimize") -> None:
        assert strategy in ("minimize", "maximize")
        self.strategy = strategy
        self.best_metric: float | None = None
        self.best_epoch: int | None = None

    def __call__(self, metric: float, epoch: int) -> bool:
        if self.best_metric is None:
            self.best_metric = metric
            self.best_epoch = epoch
            return True
        improved = (
            metric <= self.best_metric
            if self.strategy == "minimize"
            else metric >= self.best_metric
        )
        if improved:
            self.best_metric = metric
            self.best_epoch = epoch
        return improved


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b = pred.size(0)
        return self.bceloss(pred.view(b, -1), target.view(b, -1))


class DiceLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = 1.0
        b = pred.size(0)
        pred_ = pred.view(b, -1)
        target_ = target.view(b, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (
            pred_.sum(1) + target_.sum(1) + smooth
        )
        return 1 - dice_score.sum() / b


class BceDiceLoss(nn.Module):
    """BCE + Dice combined loss."""

    def __init__(self, wb: float = 1.0, wd: float = 1.0) -> None:
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.wb * self.bce(pred, target) + self.wd * self.dice(pred, target)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def iou_score(output: torch.Tensor, target: torch.Tensor) -> float:
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    return float((intersection + smooth) / (union + smooth))


@torch.no_grad()
def dice_coef(output: torch.Tensor, target: torch.Tensor) -> float:
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return float(
        (2.0 * intersection + smooth) / (output.sum() + target.sum() + smooth)
    )
