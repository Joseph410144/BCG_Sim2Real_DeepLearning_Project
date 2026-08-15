"""Two-stage training for the residual algorithm-aware BCG filter."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from Dataset.BCG_Dataset import BCGSynthesisDataset, RealBCGHeartRateDataset
from Dataset.metadata import parse_real_recording_name
from Dataset.splits import leave_one_subject_out_folds
from Model.AlgorithmAwareFilter import AlgorithmAwareResidualFilter
from Model.AlgorithmAwareLoss import (
    HarmonicHeartRateLoss, HeartV6SurrogateLoss, SpectralReconstructionLoss,
    real_algorithm_aware_loss, synthetic_filter_loss,
)
from train_BCG_HeartFilter import DEFAULT_DATA_ROOT, PROJECT_ROOT, resolve_device, seed_everything


DEFAULT_REAL_DATA = PROJECT_ROOT.parent / "Dataset" / "BCG" / "DeepLearningData" / "BCG_ECG_10sec"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-train-dir", type=Path, default=DEFAULT_DATA_ROOT / "training")
    parser.add_argument("--synthetic-val-dir", type=Path, default=DEFAULT_DATA_ROOT / "validation")
    parser.add_argument("--real-data-dir", type=Path, default=DEFAULT_REAL_DATA)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "weight" / "AlgorithmAwareFilter" / "run")
    parser.add_argument("--fold", type=int, default=0, choices=range(10))
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=2e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--initial-gate", type=float, default=-2.0)
    parser.add_argument("--identity-weight", type=float, default=0.01)
    parser.add_argument("--residual-tv-weight", type=float, default=0.005)
    parser.add_argument("--real-objective", choices=("heart-v6", "envelope"),
                        default="heart-v6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-synthetic-train", type=int)
    parser.add_argument("--max-synthetic-val", type=int)
    parser.add_argument("--max-real-train", type=int)
    parser.add_argument("--max-real-val", type=int)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    return parser.parse_args(argv)


def limited(dataset, count):
    return dataset if count is None or count >= len(dataset) else Subset(dataset, range(count))


def subject_balanced_limited(dataset, count, seed):
    """Limit real data without silently selecting only early subject folders."""
    if count is None or count >= len(dataset):
        return dataset
    groups = {}
    for index, filename in enumerate(dataset.signals):
        subject_id = parse_real_recording_name(filename).subject_id
        groups.setdefault(subject_id, []).append(index)
    rng = np.random.default_rng(seed)
    selected = []
    base, remainder = divmod(count, len(groups))
    for offset, subject_id in enumerate(sorted(groups)):
        candidates = np.asarray(groups[subject_id])
        take = min(len(candidates), base + (offset < remainder))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    # Redistribute quota if a short subject could not supply its share.
    if len(selected) < count:
        remaining = np.setdiff1d(np.arange(len(dataset)), np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(remaining, size=count - len(selected), replace=False).tolist())
    rng.shuffle(selected)
    return Subset(dataset, selected)


def loader(dataset, args, shuffle):
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, drop_last=False, generator=generator)


def average_components(sums, batches):
    return {key: value / batches for key, value in sums.items()}


def synthetic_epoch(model, data_loader, spectral_loss, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums = {"total": 0.0, "time": 0.0, "spectral": 0.0}
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, targets in tqdm(data_loader, desc="Synthetic train" if training else "Synthetic val", leave=False):
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            loss, components = synthetic_filter_loss(output, targets, spectral_loss)
            if training:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            sums["total"] += loss.item()
            for key, value in components.items(): sums[key] += value.item()
    return average_components(sums, len(data_loader))


def real_epoch(model, data_loader, harmonic_loss, device, optimizer=None,
               identity_weight=0.01, residual_tv_weight=0.005):
    training = optimizer is not None
    model.train(training)
    sums = {"total": 0.0, "hr": 0.0, "identity": 0.0, "residual_tv": 0.0}
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, target_bpm, _, _ in tqdm(data_loader, desc="Real train" if training else "Real val", leave=False):
            inputs = inputs.to(device, dtype=torch.float32)
            target_bpm = target_bpm.to(device, dtype=torch.float32)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            loss, components = real_algorithm_aware_loss(
                output, inputs, target_bpm, harmonic_loss,
                identity_weight=identity_weight, residual_tv_weight=residual_tv_weight,
            )
            if training:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            sums["total"] += loss.item()
            for key, value in components.items(): sums[key] += value.item()
    return average_components(sums, len(data_loader))


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = AlgorithmAwareResidualFilter(
        args.channels, args.blocks, initial_gate=args.initial_gate
    ).to(device)
    spectral_loss = SpectralReconstructionLoss().to(device)
    harmonic_loss = (
        HeartV6SurrogateLoss() if args.real_objective == "heart-v6"
        else HarmonicHeartRateLoss()
    ).to(device)
    history = {"synthetic": [], "real": []}

    folds = leave_one_subject_out_folds(range(1, 11))
    fold = folds[args.fold]
    config = {**vars(args), "device": str(device), "subjects": fold.to_dict()}
    for key, value in config.items():
        if isinstance(value, Path): config[key] = str(value.resolve())
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if not args.skip_pretrain and args.pretrain_epochs:
        train_data = limited(BCGSynthesisDataset(str(args.synthetic_train_dir)), args.max_synthetic_train)
        val_data = limited(BCGSynthesisDataset(str(args.synthetic_val_dir)), args.max_synthetic_val)
        train_loader, val_loader = loader(train_data, args, True), loader(val_data, args, False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        best = float("inf")
        for epoch in range(args.pretrain_epochs):
            train_metrics = synthetic_epoch(model, train_loader, spectral_loss, device, optimizer)
            val_metrics = synthetic_epoch(model, val_loader, spectral_loss, device)
            history["synthetic"].append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
            if val_metrics["total"] < best:
                best = val_metrics["total"]
                torch.save(model.state_dict(), args.output_dir / "best_synthetic.pth")
            print(f"synthetic {epoch + 1}/{args.pretrain_epochs}: train={train_metrics['total']:.4f} val={val_metrics['total']:.4f}")
        model.load_state_dict(torch.load(args.output_dir / "best_synthetic.pth", map_location=device, weights_only=True))

    if not args.skip_finetune and args.finetune_epochs:
        real_train = subject_balanced_limited(
            RealBCGHeartRateDataset(str(args.real_data_dir), fold.train_subjects),
            args.max_real_train, args.seed,
        )
        real_val = subject_balanced_limited(
            RealBCGHeartRateDataset(str(args.real_data_dir), fold.val_subjects),
            args.max_real_val, args.seed + 1,
        )
        train_loader, val_loader = loader(real_train, args, True), loader(real_val, args, False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_learning_rate, weight_decay=1e-4)
        best = float("inf")
        for epoch in range(args.finetune_epochs):
            train_metrics = real_epoch(
                model, train_loader, harmonic_loss, device, optimizer,
                args.identity_weight, args.residual_tv_weight,
            )
            val_metrics = real_epoch(
                model, val_loader, harmonic_loss, device,
                identity_weight=args.identity_weight,
                residual_tv_weight=args.residual_tv_weight,
            )
            history["real"].append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
            if val_metrics["total"] < best:
                best = val_metrics["total"]
                torch.save(model.state_dict(), args.output_dir / "best_real.pth")
            print(f"real {epoch + 1}/{args.finetune_epochs}: train={train_metrics['total']:.4f} val={val_metrics['total']:.4f}")

    torch.save(model.state_dict(), args.output_dir / "last_model.pth")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
