"""Train the synthetic BCG cardiac-component denoising model."""

import argparse
import json
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchinfo import summary
from tqdm import tqdm

from Dataset.BCG_Dataset import BCGSynthesisDataset
from Model.LSTM import LSTM_BCGFilter_Pre
from Model.Loss_Function import MorletCWTLoss
from logger import get_logger


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "Dataset" / "BCG" / "Synthesis"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_DATA_ROOT / "training")
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_DATA_ROOT / "validation")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "weight" / "BCG_HeartFilter" / "run")
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--step-size", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--input-length", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, help="Useful for smoke tests")
    parser.add_argument("--max-val-samples", type=int, help="Useful for smoke tests")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args(argv)


def resolve_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited_dataset(dataset, maximum):
    if maximum is None or maximum >= len(dataset):
        return dataset
    return Subset(dataset, range(maximum))


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_loss, train_losses, val_losses):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        checkpoint["epoch"] + 1,
        checkpoint["best_val_loss"],
        checkpoint.get("train_losses", checkpoint.get("total_loss_list", [])),
        checkpoint.get("val_losses", checkpoint.get("val_loss_list", [])),
    )


def combined_loss(prediction, target, cwt_loss):
    return 0.5 * cwt_loss(prediction, target) + 0.5 * torch.mean((prediction - target) ** 2)


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for signals, targets in tqdm(loader, desc="Validation", leave=False):
            signals = signals.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            total += combined_loss(model(signals), targets, criterion).item()
    if not len(loader):
        raise ValueError("Validation loader is empty")
    return total / len(loader)


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, device,
          output_dir, epochs, start_epoch=0, best_val_loss=math.inf,
          train_losses=None, val_losses=None, logger=None):
    train_losses = [] if train_losses is None else train_losses
    val_losses = [] if val_losses is None else val_losses
    best_train_loss = min(train_losses, default=math.inf)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for signals, targets in tqdm(train_loader, desc=f"Training {epoch + 1}/{epochs}", leave=False):
            signals = signals.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = combined_loss(model(signals), targets, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        if not len(train_loader):
            raise ValueError("Training loader is empty")
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), output_dir / "best_train_model.pth")

        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_val_model.pth")
            logger.info("New best validation model: epoch=%d loss=%.6f", epoch + 1, val_loss)

        logger.info("Epoch %d/%d lr=%.8g train=%.6f val=%.6f", epoch + 1, epochs,
                    optimizer.param_groups[0]["lr"], train_loss, val_loss)
        scheduler.step()
        save_checkpoint(output_dir / "checkpoint.pth", epoch, model, optimizer, scheduler,
                        best_val_loss, train_losses, val_losses)
    return train_losses, val_losses, best_val_loss


def main(argv=None):
    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(str(args.output_dir / "train.log"), name="bcg-training")

    train_dataset = limited_dataset(BCGSynthesisDataset(str(args.train_dir)), args.max_train_samples)
    val_dataset = limited_dataset(BCGSynthesisDataset(str(args.val_dir)), args.max_val_samples)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                          pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False,
                              generator=generator, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_options)

    model = LSTM_BCGFilter_Pre(args.input_length, 1, args.hidden_size, 1,
                               args.dropout, args.num_layers, True, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    criterion = MorletCWTLoss(fs=100, fmin=0.7, fmax=10.0, num_freqs=48,
                              kernel_size=401, sigma=0.3, log_compress=True).to(device)

    start_epoch, best_val_loss, train_losses, val_losses = 0, math.inf, [], []
    checkpoint_path = args.output_dir / "checkpoint.pth"
    if args.resume and checkpoint_path.is_file():
        start_epoch, best_val_loss, train_losses, val_losses = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, device)
        logger.info("Resuming at epoch %d", start_epoch + 1)
    elif args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location=device, weights_only=True))
        logger.info("Loaded pretrained weights: %s", args.pretrained)

    config = {**vars(args), "device": str(device)}
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value.resolve())
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_dir / "model_summary.txt").write_text(
        str(summary(model, input_size=(1, 1, args.input_length), device=device, verbose=0)),
        encoding="utf-8",
    )
    logger.info("device=%s train_samples=%d val_samples=%d", device, len(train_dataset), len(val_dataset))

    train_losses, val_losses, best_val_loss = train(
        model, train_loader, val_loader, optimizer, scheduler, criterion, device,
        args.output_dir, args.epochs, start_epoch, best_val_loss, train_losses, val_losses, logger)

    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "loss.png")
    plt.close()
    logger.info("Training complete. Best validation loss: %.6f", best_val_loss)


if __name__ == "__main__":
    main()
