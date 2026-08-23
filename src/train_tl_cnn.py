#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Traffic-Light Color CNN (PyTorch, simple & practical)
- Dataset: ImageFolder-style folder with subdirs red/yellow/green/unknown
- Default model: ResNet18 (pretrained) with last layer replaced
- Handles class imbalance via WeightedRandomSampler
- Saves: best_model.pth, class_indices.json, training_log.csv
- Also supports single-image prediction: add --predict "path_to_image"

本版改动：
- 默认 batch_size=32，epochs=40（配合 early stopping）
- SimpleCNN 中加入 BatchNorm2d + Dropout(0.3)
- 加入 Early Stopping（监控 val_loss，patience=5）
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple
import time
import csv
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models
from PIL import Image

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def compute_class_weights(targets: List[int], num_classes: int) -> torch.Tensor:
    counts = [0]*num_classes
    for t in targets:
        counts[t] += 1
    total = sum(counts)
    weights = [total/(c if c > 0 else 1) for c in counts]
    s = sum(weights)
    weights = [w/s*num_classes for w in weights]
    return torch.tensor(weights, dtype=torch.float32)

def build_weighted_sampler(targets: List[int]) -> WeightedRandomSampler:
    from collections import Counter
    cnt = Counter(targets)
    sample_weights = [1.0/max(1, cnt[t]) for t in targets]
    return WeightedRandomSampler(weights=sample_weights,
                                 num_samples=len(sample_weights),
                                 replacement=True)

def split_indices_stratified(targets: List[int], train_ratio=0.8, seed=42):
    from collections import defaultdict
    buckets = defaultdict(list)
    for idx, t in enumerate(targets):
        buckets[t].append(idx)
    rnd = random.Random(seed)
    train_idx, val_idx = [], []
    for t, idxs in buckets.items():
        rnd.shuffle(idxs)
        k = int(len(idxs)*train_ratio)
        train_idx += idxs[:k]
        val_idx += idxs[k:]
    rnd.shuffle(train_idx)
    rnd.shuffle(val_idx)
    return train_idx, val_idx

class SimpleCNN(nn.Module):
    """轻量 CNN：Conv + BN + ReLU + Pool * 3 + 全连接 + Dropout"""
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def make_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m
    elif name == "mobilenetv3":
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    elif name == "simple":
        return SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name} (choose resnet18|mobilenetv3|simple)")

def accuracy_top1(logits, targets):
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()

def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            acc = accuracy_top1(logits, labels)
        bs = imgs.size(0)
        loss_sum += loss.item() * bs
        acc_sum += acc * bs
        n += bs
    return loss_sum / n, acc_sum / n

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        acc = accuracy_top1(logits, labels)
        bs = imgs.size(0)
        loss_sum += loss.item() * bs
        acc_sum += acc * bs
        n += bs
    return loss_sum / n, acc_sum / n

def main():
    parser = argparse.ArgumentParser(
        description="Train a CNN for traffic-light color classification."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--out-dir", type=str, required=False, default="cnn_out"
    )
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)  # 修改为 32
    parser.add_argument("--epochs", type=int, default=40)      # 训练上限，配合 early stopping
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "mobilenetv3", "simple"],
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Single-image prediction using saved model in out-dir.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    # Early stopping 参数
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs without val_loss improvement).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # ---------- PREDICT MODE ----------
    if args.predict is not None:
        model_path = out_dir / "best_model.pth"
        mapping_path = out_dir / "class_indices.json"
        assert model_path.exists(), f"Model not found: {model_path}"
        assert mapping_path.exists(), f"Mapping not found: {mapping_path}"
        with open(mapping_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(idx_to_class)

        model = make_model(
            args.model,
            num_classes=num_classes,
            pretrained=not args.no_pretrained,
        )
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()

        tfm = transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        img = Image.open(args.predict).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1).squeeze(0)
            pred_idx = int(prob.argmax().item())
            pred_class = idx_to_class[pred_idx]
            conf = float(prob[pred_idx].item())
        print(f"[PREDICT] {args.predict} -> {pred_class} ({conf:.3f})")
        return

    # ---------- TRAIN MODE ----------
    if args.data_dir is None:
        parser.error("--data-dir is required in training mode")
    data_dir = Path(args.data_dir)
    assert data_dir.exists(), (
        f"Data dir not found: {data_dir} "
        f"(expect subfolders red/yellow/green/unknown)"
    )

    # 数据增强（训练）与验证变换
    train_tfm = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
            ),
            transforms.RandomAffine(
                degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    val_tfm = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # 整体数据集（只初始化一次 ImageFolder，再用 indices 切分）
    full_ds = datasets.ImageFolder(root=str(data_dir))
    class_to_idx = full_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print("[Info] Classes:", idx_to_class)

    # 目标标签，用于分层划分 & class weights
    targets = [y for _, y in full_ds.samples]
    train_idx, val_idx = split_indices_stratified(
        targets, train_ratio=0.85, seed=args.seed
    )

    # 重新封装 train / val 为 Subset + 各自的 transform
    train_base = datasets.ImageFolder(root=str(data_dir), transform=train_tfm)
    val_base = datasets.ImageFolder(root=str(data_dir), transform=val_tfm)
    train_ds = Subset(train_base, train_idx)
    val_ds = Subset(val_base, val_idx)

    # Weighted sampler for imbalanced classes
    train_targets = [targets[i] for i in train_idx]
    sampler = build_weighted_sampler(train_targets)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_classes = len(class_to_idx)
    model = make_model(
        args.model,
        num_classes=num_classes,
        pretrained=(not args.no_pretrained),
    )
    model.to(device)

    class_weights = compute_class_weights(
        targets, num_classes=num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    # 日志文件
    log_csv = out_dir / "training_log.csv"
    with log_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        )

    # 保存类别映射
    with (out_dir / "class_indices.json").open(
        "w", encoding="utf-8"
    ) as fjs:
        json.dump(class_to_idx, fjs, ensure_ascii=False, indent=2)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    bad_epochs = 0
    patience = max(1, args.patience)
    best_path = out_dir / "best_model.pth"

    print(
        f"[Train] max_epochs={args.epochs}, batch_size={args.batch_size}, "
        f"early_stopping_patience={patience}"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, device, optimizer, criterion
        )
        va_loss, va_acc = evaluate(
            model, val_loader, device, criterion
        )
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        print(
            f"[Epoch {epoch:02d}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} "
            f"| val_loss={va_loss:.4f} acc={va_acc:.3f} "
            f"| lr={lr_now:.2e} | {dt:.1f}s"
        )

        # 写入 CSV 日志
        with log_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{tr_loss:.6f}",
                    f"{tr_acc:.6f}",
                    f"{va_loss:.6f}",
                    f"{va_acc:.6f}",
                    f"{lr_now:.6e}",
                ]
            )

        # 保存 best model（按 val_acc），并用于 early stopping 监控 val_loss
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)
            print(
                f"  [*] Saved new best (by acc) to: {best_path}  "
                f"(val_acc={best_val_acc:.3f})"
            )

        # Early stopping：以 val_loss 作为主要监控指标
        if va_loss < best_val_loss - 1e-4:
            best_val_loss = va_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            print(
                f"  [EarlyStopping] no val_loss improvement for "
                f"{bad_epochs}/{patience} epoch(s)."
            )
            if bad_epochs >= patience:
                print(
                    f"[EarlyStopping] Stop training at epoch {epoch:02d}. "
                    f"Best val_loss={best_val_loss:.4f}, "
                    f"best val_acc={best_val_acc:.3f}"
                )
                break

    print(f"[DONE] Best val acc: {best_val_acc:.3f}")
    print(f"Artifacts saved to: {out_dir}")
    print("  - best_model.pth")
    print("  - class_indices.json")
    print("  - training_log.csv")

if __name__ == "__main__":
    main()
