"""
PureForest Deep Learning Training Pipeline: 4-Channel EfficientNetV2-S.

This module implements the training, validation, and checkpointing pipeline
for adapting the EfficientNetV2-S convolutional neural network to 4-channel
Very High Resolution (VHR) aerial imagery (Near-Infrared, Red, Green, Blue)
for 13-class monospecific tree species classification.

Key Architectural & Training Features:
    1. 4-Channel Input Convolution:
       - Modifies the first convolutional layer (features.0.0) from 3 to 4 input channels.
       - Pre-trained RGB weights are preserved, and Red channel weights are cloned into
         the NIR channel weights to initialize vegetation-sensitive filters effectively.
    2. Data Augmentations:
       - Geometric: Random horizontal/vertical flips and arbitrary rotations (180°).
       - Radiometric: Color jitter applied across the visible RGB subset.
       - Regularization: Random Erasing (Cutout) and Mixup Augmentation (alpha=0.2).
    3. Optimization Strategy:
       - Differential Learning Rates: Backbone parameters train at 0.1x base LR (1e-5),
         while the newly initialized 4-channel input and classifier head train at 1.0x (1e-4).
       - AdamW optimizer with weight decay of 1e-2.
       - Cosine Annealing learning rate schedule.
       - Cross-Entropy Loss with label smoothing (epsilon=0.1).

Usage:
    Train from scratch for 20 epochs:
        $ python3 train_efficientnet.py --epochs 20

    Resume training from existing checkpoint:
        $ python3 train_efficientnet.py --epochs 20 --resume --checkpoint checkpoint.pth
"""

import os
import sys
import argparse
import random
import datetime
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np

# ----------------- Configurations & Constants -----------------
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-4
NUM_WORKERS: int = 4
DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
DATA_DIR: str = BASE_DIR
LOG_FILE_PATH: str = os.path.join(BASE_DIR, "training_log.txt")


class PureForestDataset(Dataset):
    """PyTorch Dataset for ingesting 4-Channel TIFF images from the PureForest dataset.

    Permutes spectral channels from PureForest TIFF standard [NIR, Red, Green, Blue] to
    the standard deep learning order [Red, Green, Blue, NIR] and normalizes pixel
    intensities to [0.0, 1.0].
    """

    def __init__(self, file_list: List[Tuple[str, int]], transform: Optional[Any] = None) -> None:
        """Initialize the PureForest dataset instance.

        Args:
            file_list: List of tuples containing (file_path, integer_class_id).
            transform: Optional callable transform pipeline to apply to image tensors.
        """
        self.file_list = file_list
        self.transform = transform

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load and process a single 4-channel image and its ground truth class label.

        Args:
            idx: Integer index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: 4-channel image tensor of shape (4, 250, 250) and integer label.
        """
        img_path, label = self.file_list[idx]
        try:
            with Image.open(img_path) as img:
                arr = np.array(img).astype(np.float32)
                # Permute channels: [0: NIR, 1: Red, 2: Green, 3: Blue] -> [Red, Green, Blue, NIR]
                arr = arr[:, :, [1, 2, 3, 0]]
                # Normalize pixel values from [0, 255] to [0.0, 1.0]
                arr = arr / 255.0
                tensor_img = torch.from_numpy(arr).permute(2, 0, 1)

                if self.transform:
                    tensor_img = self.transform(tensor_img)
                return tensor_img, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(4, 250, 250, dtype=torch.float32), label


class JointTransform:
    """Custom data augmentation pipeline supporting multi-spectral 4-channel imagery."""

    def __init__(
        self,
        geometric_transform: Optional[transforms.Compose] = None,
        color_jitter: Optional[transforms.ColorJitter] = None,
        random_erasing: Optional[transforms.RandomErasing] = None
    ) -> None:
        """Initialize joint transform pipeline.

        Args:
            geometric_transform: Transformations applied to all 4 channels simultaneously.
            color_jitter: Radiometric jitter applied exclusively to RGB channels (0:3).
            random_erasing: Cutout / rectangular erasing applied to the full 4-channel tensor.
        """
        self.geometric_transform = geometric_transform
        self.color_jitter = color_jitter
        self.random_erasing = random_erasing

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply sequential transformations to a 4-channel tensor.

        Args:
            tensor: Input image tensor of shape (4, H, W).

        Returns:
            torch.Tensor: Augmented image tensor.
        """
        if self.geometric_transform:
            tensor = self.geometric_transform(tensor)
        if self.color_jitter:
            # Color jitter only operates meaningfully on standard 3-channel RGB
            rgb = tensor[0:3, :, :]
            rgb = self.color_jitter(rgb)
            tensor = torch.cat([rgb, tensor[3:4, :, :]], dim=0)
        if self.random_erasing:
            tensor = self.random_erasing(tensor)
        return tensor


def get_efficientnet_v2_model(num_classes: int = 13) -> nn.Module:
    """Instantiate and adapt pre-trained EfficientNetV2-S for 4-channel input and target classes.

    Replaces the initial 3-channel convolutional stem with a 4-channel convolution. Weights for
    the NIR channel are warm-started by copying weights from the Red channel.

    Args:
        num_classes: Number of target output classes (default: 13).

    Returns:
        nn.Module: Adapted EfficientNetV2-S model.

    Example:
        >>> model = get_efficientnet_v2_model(num_classes=13)
        >>> print(model.classifier[1].out_features)
        13
    """
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)

    original_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )

    with torch.no_grad():
        # Copy pre-trained RGB weights directly
        new_conv.weight[:, 0:3, :, :] = original_conv.weight
        # Initialize NIR channel (channel 3) with Red channel weights (channel 1)
        new_conv.weight[:, 3, :, :] = original_conv.weight[:, 1, :, :]
        if original_conv.bias is not None:
            new_conv.bias = original_conv.bias

    model.features[0][0] = new_conv
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model


def collect_dataset_files(base_dir: str, split_name: str) -> List[Tuple[str, int]]:
    """Scan all species directories for patch files belonging to a specific dataset split.

    Args:
        base_dir: Base workspace directory.
        split_name: Split folder to search ('train', 'val', 'test').

    Returns:
        List[Tuple[str, int]]: List of (filepath, class_id) tuples.
    """
    files: List[Tuple[str, int]] = []
    if not os.path.exists(base_dir):
        return files

    species_dirs = [d for d in os.listdir(base_dir) if d.startswith("imagery-")]
    for s_dir in species_dirs:
        split_path = os.path.join(base_dir, s_dir, split_name)
        if os.path.exists(split_path):
            for f in os.listdir(split_path):
                if f.endswith(".tiff") or f.endswith(".tif"):
                    parts = f.split("-")
                    if len(parts) >= 3 and parts[2].startswith("C"):
                        class_id = int(parts[2][1:])
                        files.append((os.path.join(split_path, f), class_id))
    return files


def clear_line() -> None:
    """Utility to clear the current terminal line for clean in-place progress output."""
    sys.stdout.write("\r" + " " * 90 + "\r")
    sys.stdout.flush()


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Perform Mixup data augmentation on input batch.

    Computes convex combinations of pairs of examples and their labels.

    Args:
        x: Input batch tensor of shape (B, C, H, W).
        y: Ground truth target labels tensor of shape (B,).
        alpha: Beta distribution parameter controlling interpolation strength.

    Returns:
        Tuple containing (mixed_x, y_a, y_b, lambda_weight).
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, float(lam)


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """Compute interpolated Mixup loss using ground truth label pairs."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def load_optimizer_state_dict_flexible(
    optimizer: torch.optim.Optimizer,
    checkpoint_optimizer_state_dict: Dict[str, Any],
    model: nn.Module
) -> None:
    """Safely restore optimizer state dict when parameter groups or split architecture changes."""
    checkpoint_param_ids: List[int] = []
    for gp in checkpoint_optimizer_state_dict["param_groups"]:
        checkpoint_param_ids.extend(gp["params"])

    saved_state = checkpoint_optimizer_state_dict["state"]

    if len(checkpoint_optimizer_state_dict["param_groups"]) == 1:
        checkpoint_ordered_params = list(model.parameters())
    else:
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if "classifier" in name or "features.0.0" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        checkpoint_ordered_params = backbone_params + head_params

    param_to_state = {}
    for i, p_id in enumerate(checkpoint_param_ids):
        if p_id in saved_state:
            param_to_state[checkpoint_ordered_params[i]] = saved_state[p_id]

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in param_to_state:
                optimizer.state[p] = param_to_state[p]


def main() -> None:
    """Execute the deep learning training and validation loop with logging and checkpointing."""
    parser = argparse.ArgumentParser(description="Train 4-channel EfficientNetV2 on PureForest dataset.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs to execute.")
    parser.add_argument("--resume", action="store_true", help="Resume training from an existing checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Path to checkpoint state file.")
    parser.add_argument("--best-model", type=str, default="efficientnet_v2_forest.pth", help="Path to save best weights-only model.")
    parser.add_argument("--reset-best-acc", action="store_true", help="Reset best validation accuracy tracking to 0.0%.")
    args = parser.parse_args()

    checkpoint_path = os.path.join(BASE_DIR, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    best_model_path = os.path.join(BASE_DIR, args.best_model) if not os.path.isabs(args.best_model) else args.best_model

    session_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE_PATH, "a") as f:
        f.write("\n=========================================\n")
        f.write(f"Session Started: {session_start_time}\n")
        f.write(f"Target Epochs: {args.epochs} | Resume Flag: {args.resume} | Device: {DEVICE}\n")
        f.write("=========================================\n")

    print(f"Device: {DEVICE}")
    print(f"Log File: {LOG_FILE_PATH}")

    # Configure augmentation pipeline
    geometric_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=180),
    ])
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    random_erasing = transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), value=0)
    train_transform = JointTransform(geometric_transform, color_jitter, random_erasing)

    train_files = collect_dataset_files(DATA_DIR, "train")
    val_files = collect_dataset_files(DATA_DIR, "val")
    print(f"Found {len(train_files)} training images and {len(val_files)} validation images.")

    if not train_files:
        print("No training images found in dataset splits. Please verify data paths.")
        return

    train_dataset = PureForestDataset(train_files, transform=train_transform)
    val_dataset = PureForestDataset(val_files)

    use_pin_memory = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=use_pin_memory)

    # Initialize model, loss, and differential optimizer
    model = get_efficientnet_v2_model(num_classes=13).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "classifier" in name or "features.0.0" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LEARNING_RATE * 0.1},
        {"params": head_params, "lr": LEARNING_RATE}
    ], weight_decay=1e-2)

    start_epoch = 0
    best_acc = 0.0

    if args.resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        load_optimizer_state_dict_flexible(optimizer, checkpoint["optimizer_state_dict"], model)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint.get("best_acc", 0.0)
        if args.reset_best_acc:
            print("Resetting best validation accuracy tracking to 0.0%")
            best_acc = 0.0
        print(f"Resuming from Epoch {start_epoch} (Best validation accuracy: {best_acc:.2f}%)")

        with open(LOG_FILE_PATH, "a") as f:
            f.write(f"Resumed from checkpoint: Epoch {start_epoch} | Prev Best Acc: {best_acc:.2f}%\n")
    elif args.resume:
        print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")

    end_epoch = start_epoch + args.epochs
    print(f"Training range: Epoch {start_epoch + 1} to Epoch {end_epoch}\n")

    for group in optimizer.param_groups:
        if 'initial_lr' not in group:
            group['initial_lr'] = LEARNING_RATE

    last_epoch = start_epoch - 1 if start_epoch > 0 else -1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=end_epoch, eta_min=1e-6, last_epoch=last_epoch)

    for epoch in range(start_epoch, end_epoch):
        epoch_start = datetime.datetime.now()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        total_batches = len(train_loader)

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            use_mixup = random.random() < 0.5
            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            percent = 100 * (i + 1) / total_batches
            sys.stdout.write(f"\rTraining Epoch {epoch+1}/{end_epoch} | Progress: {percent:.1f}% ({i+1}/{total_batches})")
            sys.stdout.flush()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct_train / total_train) * 100

        # Validation loop
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        total_val_batches = len(val_loader)

        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                val_percent = 100 * (i + 1) / total_val_batches
                sys.stdout.write(f"\rValidation Epoch {epoch+1}/{end_epoch} | Progress: {val_percent:.1f}% ({i+1}/{total_val_batches})")
                sys.stdout.flush()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_acc = (correct_val / total_val) * 100

        epoch_end = datetime.datetime.now()
        duration = epoch_end - epoch_start
        seconds = int(duration.total_seconds())
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        duration_str = f"{h}h " if h > 0 else ""
        duration_str += f"{m}m " if (m > 0 or h > 0) else ""
        duration_str += f"{s}s"

        timestamp_str = epoch_end.strftime("%Y-%m-%d %H:%M:%S")

        clear_line()
        print(f"[Epoch {epoch+1:02d}/{end_epoch:02d}] Train Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}% | Val Loss: {val_epoch_loss:.4f} - Acc: {val_acc:.2f}% | Time: {duration_str}", end="")

        # Save checkpoint dictionary
        checkpoint_state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc
        }
        torch.save(checkpoint_state, checkpoint_path)

        is_best = False
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(" (Saved Best Model)")
            is_best = True
        else:
            print(" (Checkpoint Saved)")

        log_entry = (
            f"[{timestamp_str}] Epoch {epoch+1:02d}/{end_epoch:02d} | "
            f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | "
            f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Time: {duration_str}"
        )
        if is_best:
            log_entry += " [NEW BEST MODEL SAVED]"

        with open(LOG_FILE_PATH, "a") as f:
            f.write(log_entry + "\n")

        scheduler.step()


if __name__ == "__main__":
    main()
