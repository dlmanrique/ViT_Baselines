"""
Frame-wise ViT baseline for SAR-RARP50 gesture recognition (8 classes, 10 Hz).

Each annotated frame is classified on its own by an ImageNet-pretrained ViT
from timm, fine-tuned end to end with cross-entropy on the train split. The
checkpoint with the best validation --monitor metric is then evaluated on the
val and the official test split.

Usage (from any folder, Base_Leo environment):
    python main.py
    python main.py --epochs 30 --lr 1e-4 --batch_size 128 --exp_name vitb_lr1e-4
    python main.py --eval_only --checkpoint outputs/<run>/best.pt

Outputs in <output_dir>/<exp_name>/ (with --eval_only, the checkpoint folder):
    config.json      arguments, environment and dataset sizes
    train.log        training log (eval.log with --eval_only)
    history.csv      one row per epoch: train loss/accuracy and validation metrics
    best.pt, last.pt model weights (best validation --monitor, last epoch)
    <split>_metrics.json, <split>_per_class.csv, <split>_per_video.csv,
    <split>_confusion_matrix.csv, <split>_predictions.csv (one row per frame
    with the predicted class and the class probabilities), for val and test.
"""

import os
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from metrics import compute_metrics
from models.vit import build_vit, vit_data_config
from utils import (CLASS_NAMES, NUM_CLASSES, SPLITS, GestureFrameDataset, build_loader, build_transforms,
                   check_frames, class_weights, find_frame_dirs, load_split, param_groups, parse_args, save_json,
                   set_seed, setup_logger, warmup_cosine)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, amp, grad_clip, epoch, log_every, logger):
    """Train for one epoch and return the mean training loss and accuracy."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for step, (images, labels) in enumerate(tqdm(loader, desc=f'train {epoch}', leave=False), start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(images)
        loss = criterion(logits.float(), labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
        if step % log_every == 0:
            logger.info(f'epoch {epoch} step {step}/{len(loader)} | loss {total_loss / seen:.4f} '
                        f'acc {correct / seen:.4f} | lr {scheduler.get_last_lr()[0]:.2e}')
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model, loader, device, amp, desc):
    """Return the class probabilities (num_frames x num_classes) of every frame, in loader order."""
    model.eval()
    probs = []
    for images, _ in tqdm(loader, desc=desc, leave=False):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(images.to(device, non_blocking=True))
        probs.append(logits.float().softmax(dim=1).cpu())
    return torch.cat(probs).numpy()


def evaluate(model, loader, df, device, amp, split):
    """Predict a split and return (cross-entropy loss, metrics dict, probabilities)."""
    probs = predict(model, loader, device, amp, split)
    labels = df['annotation'].to_numpy()
    loss = float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean())
    metrics = compute_metrics(df['video_id'], df['frame_id'], labels, probs.argmax(axis=1),
                              NUM_CLASSES, CLASS_NAMES)
    return loss, metrics, probs


def save_split_results(run_dir, split, df, loss, metrics, probs, checkpoint_path):
    """Write the metrics, per-class / per-video tables, confusion matrix and per-frame predictions of a split."""
    save_json({'split': split, 'checkpoint': checkpoint_path, 'loss': loss, **metrics['scalars']},
              os.path.join(run_dir, f'{split}_metrics.json'))
    metrics['per_class'].to_csv(os.path.join(run_dir, f'{split}_per_class.csv'), index=False)
    metrics['per_video'].to_csv(os.path.join(run_dir, f'{split}_per_video.csv'), index=False)
    pd.DataFrame(metrics['confusion_matrix'], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        os.path.join(run_dir, f'{split}_confusion_matrix.csv'))

    predictions = df[['video_id', 'frame_id', 'annotation']].copy()
    predictions['prediction'] = probs.argmax(axis=1)
    for c in range(NUM_CLASSES):
        predictions[f'prob_{c}'] = probs[:, c].round(5)
    predictions.to_csv(os.path.join(run_dir, f'{split}_predictions.csv'), index=False)


def format_metrics(scalars):
    """Return the main metrics of a split as one log string."""
    return (f"acc {scalars['accuracy']:.4f} | f1_macro {scalars['f1_macro']:.4f} | "
            f"F1@10 {scalars['f1@10']:.4f} F1@25 {scalars['f1@25']:.4f} F1@50 {scalars['f1@50']:.4f} | "
            f"edit {scalars['edit']:.4f} | score {scalars['action_score']:.4f}")


def evaluate_and_save(model, loaders, splits, device, amp, run_dir, checkpoint_path, logger):
    """Evaluate the current model on val and test and save the results of both."""
    for split in ('val', 'test'):
        loss, metrics, probs = evaluate(model, loaders[split], splits[split], device, amp, split)
        save_split_results(run_dir, split, splits[split], loss, metrics, probs, checkpoint_path)
        logger.info(f'{split:<5} | loss {loss:.4f} | {format_metrics(metrics["scalars"])}')


def main():
    """Entry point: train the frame-wise ViT (or load a checkpoint) and evaluate it on val and test."""
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    device = torch.device(args.device)
    amp = not args.no_amp and device.type == 'cuda'

    if args.eval_only:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        args.model, args.img_size = checkpoint['args']['model'], checkpoint['args']['img_size']
        run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        logger = setup_logger(os.path.join(run_dir, 'eval.log'))
    else:
        run_dir = os.path.join(os.path.abspath(args.output_dir), args.exp_name)
        os.makedirs(run_dir, exist_ok=False)  # never write a new run into an existing folder
        logger = setup_logger(os.path.join(run_dir, 'train.log'))
    logger.info(f'Run folder: {run_dir}')

    # Data
    frame_dirs = find_frame_dirs(args.sar_root)
    splits = {split: load_split(args.splits_dir, split) for split in SPLITS}
    for split, df in splits.items():
        check_frames(df, frame_dirs)
        logger.info(f'{split}: {len(df)} frames from {df["video_id"].nunique()} videos')

    # Model and loaders
    model = build_vit(args.model, NUM_CLASSES, pretrained=not (args.no_pretrained or args.eval_only),
                      img_size=args.img_size, drop_rate=args.dropout, drop_path_rate=args.drop_path)
    data_config = vit_data_config(model)
    train_transform, eval_transform = build_transforms(args.img_size, data_config, args.crop_scale_min,
                                                       args.color_jitter)
    loaders = {
        split: build_loader(GestureFrameDataset(df, frame_dirs, train_transform if split == 'train' else eval_transform),
                            args.batch_size if split == 'train' else args.eval_batch_size,
                            train=split == 'train', num_workers=args.num_workers, seed=args.seed)
        for split, df in splits.items()
    }

    if args.eval_only:
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        logger.info(f'Evaluating {args.checkpoint} (epoch {checkpoint["epoch"]})')
        evaluate_and_save(model, loaders, splits, device, amp, run_dir, os.path.abspath(args.checkpoint), logger)
        return

    model.to(device)
    save_json({
        'args': vars(args),
        'data_config': data_config,
        'num_frames': {split: len(df) for split, df in splits.items()},
        'num_videos': {split: int(df['video_id'].nunique()) for split, df in splits.items()},
        'torch': torch.__version__,
        'timm': timm.__version__,
        'cuda_device': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
    }, os.path.join(run_dir, 'config.json'))
    logger.info(f'Model {args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters | '
                f'mean {data_config["mean"]} std {data_config["std"]}')

    # Optimization
    weights = class_weights(splits['train']['annotation'].to_numpy(), NUM_CLASSES, args.class_weighting)
    if weights is not None:
        logger.info(f'Class weights ({args.class_weighting}): {np.round(weights.numpy(), 3).tolist()}')
        weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)
    steps_per_epoch = len(loaders['train'])
    scheduler = LambdaLR(optimizer, warmup_cosine(args.epochs * steps_per_epoch,
                                                  round(args.warmup_epochs * steps_per_epoch), args.lr, args.min_lr))

    # Training
    best_path, last_path = os.path.join(run_dir, 'best.pt'), os.path.join(run_dir, 'last.pt')
    best_value, history = -np.inf, []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, loaders['train'], criterion, optimizer, scheduler, device,
                                                amp, args.grad_clip, epoch, args.log_every, logger)
        val_loss, val_metrics, _ = evaluate(model, loaders['val'], splits['val'], device, amp, 'val')
        scalars = val_metrics['scalars']

        history.append({'epoch': epoch, 'lr': scheduler.get_last_lr()[0], 'train_loss': train_loss,
                        'train_acc': train_acc, 'val_loss': val_loss,
                        **{f'val_{k}': v for k, v in scalars.items()}, 'seconds': round(time.time() - start, 1)})
        pd.DataFrame(history).to_csv(os.path.join(run_dir, 'history.csv'), index=False)

        checkpoint = {'model': model.state_dict(), 'epoch': epoch, 'val_metrics': scalars, 'args': vars(args)}
        torch.save(checkpoint, last_path)
        is_best = scalars[args.monitor] > best_value
        if is_best:
            best_value = scalars[args.monitor]
            torch.save(checkpoint, best_path)
        logger.info(f'epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | '
                    f'val loss {val_loss:.4f} | {format_metrics(scalars)} | {time.time() - start:.0f}s'
                    + (f' | new best {args.monitor}' if is_best else ''))

    # Final evaluation with the best checkpoint
    best = torch.load(best_path, map_location='cpu')
    model.load_state_dict(best['model'])
    logger.info(f'Best epoch {best["epoch"]} ({args.monitor} {best_value:.4f}); evaluating on val and test')
    evaluate_and_save(model, loaders, splits, device, amp, run_dir, best_path, logger)


if __name__ == '__main__':
    main()
