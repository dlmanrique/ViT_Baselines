"""
Frame-wise ViT baseline for SAR-RARP50, for the task selected with --task:
    gesture_recognition   8 gesture classes at 10 Hz (multiclass metrics)
    error_detection       normal / error at 5 Hz (binary metrics: AUC, AP, ...)

Each annotated frame is classified on its own by an ImageNet-pretrained ViT
from timm, fine-tuned end to end with cross-entropy on the train split. The
checkpoint with the best validation monitor metric of the task (frame-wise
accuracy for gestures, AP for errors) is then evaluated on the test split.

Training logs everything to Weights & Biases, in a run named after its start
date and time (YYYY-MM-DD_HH-MM-SS):
    train/loss, train/lr          every --log_every steps
    train_epoch/*, val/*          every epoch (loss, all metrics of the task,
    val_class_f1/*                per-class F1 and val/best_<monitor> so far)
    best_val/*, test/*            at the end, for the best checkpoint: all metrics (run
                                  summary) and per-class and per-video tables.
The only file written is the best checkpoint, <output_dir>/<run name>/best.pt,
next to the local wandb run folder (<output_dir>/wandb).

Sweeps: the arguments are read back from wandb.config after wandb.init, so a
`wandb agent` can override any of them (e.g. --lr). Use val/best_<monitor> as
the sweep metric (goal: maximize): val/best_accuracy for gestures, val/best_ap
for errors.

Inference (--eval_only): loads the ViT in --checkpoint, evaluates it on the val
and test splits of --task and prints every metric of the task in the terminal.
Nothing is logged to wandb and no file is written.

Usage (from any folder, Base_Leo environment):
    python main.py --task gesture_recognition            # training, after `wandb login`
    python main.py --task error_detection --epochs 30 --lr 1e-4
    python main.py --eval_only --task error_detection --checkpoint outputs/<run name>/best.pt
"""

import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from metrics import compute_metrics
from models.vit import build_vit, vit_data_config
from utils import (SPLITS, TASKS, FrameDataset, build_loader, build_transforms, check_frames, class_weights,
                   find_frame_dirs, load_split, param_groups, parse_args, plot_timelines, set_seed, warmup_cosine)

NUM_PLOT_VIDEOS = 3  # random test videos drawn in the inference timeline plot


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, amp, grad_clip, epoch, log_every, step):
    """Train for one epoch, logging the loss to wandb every log_every steps.

    Returns the mean training loss and accuracy of the epoch and the global step.
    """
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    window_loss, window_steps = 0.0, 0
    for images, labels in tqdm(loader, desc=f'train {epoch}', leave=False):
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
        step += 1

        loss_value = loss.item()
        total_loss += loss_value * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
        window_loss += loss_value
        window_steps += 1
        if step % log_every == 0:
            wandb.log({'step': step, 'train/loss': window_loss / window_steps,
                       'train/lr': scheduler.get_last_lr()[0]})
            window_loss, window_steps = 0.0, 0
    return total_loss / seen, correct / seen, step


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


def evaluate(model, loader, df, device, amp, split, task):
    """Predict a split and return (cross-entropy loss, metrics dict of the task type, class probabilities)."""
    probs = predict(model, loader, device, amp, split)
    labels = df['annotation'].to_numpy()
    loss = float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean())
    metrics = compute_metrics(task['type'], df['video_id'], df['frame_id'], labels, probs, task['class_names'])
    return loss, metrics, probs


def load_data(args, task, split_names):
    """Find the frame folders and load and check the requested splits of the task.

    Returns ({video_id: frames folder}, {split: DataFrame}).
    """
    splits_dir = args.splits_dir or task['splits_dir']
    frame_dirs = find_frame_dirs(args.sar_root)
    splits = {}
    for split in split_names:
        splits[split] = load_split(splits_dir, task, split)
        check_frames(splits[split], frame_dirs)
        print(f'{split}: {len(splits[split])} frames from {splits[split]["video_id"].nunique()} videos')
    return frame_dirs, splits


def log_split_results(run, prefix, loss, metrics):
    """Store the metrics of the best checkpoint in the run summary and log its per-class and per-video tables.

    prefix is 'best_val' or 'test', kept apart from the per-epoch val/* charts.
    """
    for name, value in {'loss': loss, **metrics['scalars']}.items():
        run.summary[f'{prefix}/{name}'] = value
    wandb.log({
        f'{prefix}/per_class': wandb.Table(dataframe=metrics['per_class']),
        f'{prefix}/per_video': wandb.Table(dataframe=metrics['per_video']),
    })


def format_metrics(scalars, task_type):
    """Return the main metrics of a split as one console string."""
    if task_type == 'binary':
        return (f"AUC {scalars['auc']:.4f} | AP {scalars['ap']:.4f} | acc {scalars['accuracy']:.4f} | "
                f"precision {scalars['precision']:.4f} recall {scalars['recall']:.4f} F1 {scalars['f1']:.4f}")
    return (f"acc {scalars['accuracy']:.4f} | f1_macro {scalars['f1_macro']:.4f} | "
            f"F1@10 {scalars['f1@10']:.4f} F1@25 {scalars['f1@25']:.4f} F1@50 {scalars['f1@50']:.4f} | "
            f"edit {scalars['edit']:.4f} | score {scalars['action_score']:.4f}")


def print_split_results(split, loss, metrics, class_names):
    """Print every metric of a split: scalars, per-class and per-video tables and the confusion matrix."""
    print(f'\n{"=" * 30} {split} {"=" * 30}')
    for name, value in {'loss': loss, **metrics['scalars']}.items():
        print(f'{name:<20}{value:.4f}')
    print('\nPer class:')
    print(metrics['per_class'].to_string(index=False, float_format='%.4f'))
    print('\nPer video:')
    print(metrics['per_video'].to_string(index=False, float_format='%.4f'))
    print('\nConfusion matrix (rows: ground truth, columns: predicted class id):')
    print(pd.DataFrame(metrics['confusion_matrix'], index=[f'{c} {name}' for c, name in enumerate(class_names)],
                       columns=range(len(class_names))).to_string())


def run_inference(args):
    """Evaluate the ViT in --checkpoint on the val and test splits of --task and print the metrics."""
    task = TASKS[args.task]
    class_names = task['class_names']
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    trained_args = checkpoint['args']
    trained_task = trained_args.get('task', 'gesture_recognition')  # checkpoints saved before --task existed
    if trained_task != args.task:
        raise ValueError(f'{args.checkpoint} was trained for {trained_task}, not for {args.task}')

    device = torch.device(args.device)
    amp = not args.no_amp and device.type == 'cuda'
    torch.backends.cudnn.benchmark = True

    model = build_vit(trained_args['model'], len(class_names), pretrained=False, img_size=trained_args['img_size'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    _, eval_transform = build_transforms(trained_args['img_size'], vit_data_config(model), args.crop_scale_min,
                                         args.color_jitter)
    monitor = task['monitor']
    print(f'Checkpoint: {os.path.abspath(args.checkpoint)}')
    print(f'Task {args.task} | model {trained_args["model"]} | epoch {checkpoint["epoch"]} | '
          f'val {monitor} during training {checkpoint["val_metrics"][monitor]:.4f}')

    frame_dirs, splits = load_data(args, task, ('val', 'test'))
    for split, df in splits.items():
        loader = build_loader(FrameDataset(df, frame_dirs, eval_transform), args.eval_batch_size, train=False,
                              num_workers=args.num_workers, seed=args.seed)
        loss, metrics, probs = evaluate(model, loader, df, device, amp, split, task)
        print_split_results(split, loss, metrics, class_names)

        if split == 'test' and 'class_colors' in task:
            video_ids = sorted(random.Random(args.seed).sample(sorted(df['video_id'].unique()), NUM_PLOT_VIDEOS))
            path = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), f'test_timelines_{args.task}.png')
            plot_timelines(df, probs.argmax(axis=1), metrics['per_video'], class_names, task['class_colors'],
                           video_ids, path, f'Test set: ground truth vs frame-wise ViT ({args.task})')
            print(f'\nTimelines of {video_ids} saved to {path}')


def run_training(args):
    """Train the frame-wise ViT on --task, keep the best checkpoint and evaluate it, logging everything to wandb."""
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=vars(args),
                     mode=args.wandb_mode, dir=output_dir, job_type='train')
    args = argparse.Namespace(**dict(run.config))  # a sweep agent may have overridden some arguments
    wandb.define_metric('step')
    wandb.define_metric('train/*', step_metric='step')
    wandb.define_metric('epoch')
    for prefix in ('train_epoch', 'val', 'val_class_f1'):
        wandb.define_metric(f'{prefix}/*', step_metric='epoch')
    print(f'wandb run {run_name} ({run.id})')

    task = TASKS[args.task]
    class_names, monitor = task['class_names'], task['monitor']
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    device = torch.device(args.device)
    amp = not args.no_amp and device.type == 'cuda'

    # Data
    frame_dirs, splits = load_data(args, task, SPLITS)

    # Model and loaders
    model = build_vit(args.model, len(class_names), pretrained=not args.no_pretrained, img_size=args.img_size,
                      drop_rate=args.dropout, drop_path_rate=args.drop_path)
    data_config = vit_data_config(model)
    train_transform, eval_transform = build_transforms(args.img_size, data_config, args.crop_scale_min,
                                                       args.color_jitter)
    loaders = {
        split: build_loader(FrameDataset(df, frame_dirs, train_transform if split == 'train' else eval_transform),
                            args.batch_size if split == 'train' else args.eval_batch_size,
                            train=split == 'train', num_workers=args.num_workers, seed=args.seed)
        for split, df in splits.items()
    }
    run.config.update({
        'data_config': data_config,
        'num_frames': {split: len(df) for split, df in splits.items()},
        'num_videos': {split: int(df['video_id'].nunique()) for split, df in splits.items()},
        'torch_version': torch.__version__,
        'timm_version': timm.__version__,
        'cuda_device': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
    })
    model.to(device)
    print(f'Model {args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters | '
          f'mean {data_config["mean"]} std {data_config["std"]}')

    # Optimization
    weights = class_weights(splits['train']['annotation'].to_numpy(), len(class_names), args.class_weighting)
    if weights is not None:
        print(f'Class weights ({args.class_weighting}): {np.round(weights.numpy(), 3).tolist()}')
        weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)
    steps_per_epoch = len(loaders['train'])
    scheduler = LambdaLR(optimizer, warmup_cosine(args.epochs * steps_per_epoch,
                                                  round(args.warmup_epochs * steps_per_epoch), args.lr, args.min_lr))

    # Training: only the best checkpoint by the validation monitor metric of the task is saved
    best_path = os.path.join(output_dir, run_name, 'best.pt')
    best_value, best_epoch, best_val, step = -np.inf, 0, None, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc, step = train_one_epoch(model, loaders['train'], criterion, optimizer, scheduler,
                                                      device, amp, args.grad_clip, epoch, args.log_every, step)
        val_loss, val_metrics, _ = evaluate(model, loaders['val'], splits['val'], device, amp, 'val', task)
        scalars = val_metrics['scalars']

        is_best = scalars[monitor] > best_value
        if is_best:
            best_value, best_epoch = scalars[monitor], epoch
            best_val = (val_loss, val_metrics)
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_metrics': scalars, 'args': vars(args),
                        'wandb_run': run_name, 'wandb_id': run.id}, best_path)

        wandb.log({
            'epoch': epoch,
            'train_epoch/loss': train_loss,
            'train_epoch/accuracy': train_acc,
            'val/loss': val_loss,
            **{f'val/{name}': value for name, value in scalars.items()},
            f'val/best_{monitor}': best_value,
            **{f'val_class_f1/{row.class_name}': row.f1 for row in val_metrics['per_class'].itertuples()},
        })
        print(f'epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | '
              f'val loss {val_loss:.4f} | {format_metrics(scalars, task["type"])} | {time.time() - start:.0f}s'
              + (f' | new best {monitor}' if is_best else ''))

    # Final evaluation of the best checkpoint
    run.summary['best_epoch'] = best_epoch
    run.summary['best_checkpoint'] = best_path
    log_split_results(run, 'best_val', *best_val)
    model.load_state_dict(torch.load(best_path, map_location='cpu')['model'])
    test_loss, test_metrics, _ = evaluate(model, loaders['test'], splits['test'], device, amp, 'test', task)
    log_split_results(run, 'test', test_loss, test_metrics)
    print(f'best epoch {best_epoch} (val {monitor} {best_value:.4f}) | test loss {test_loss:.4f} | '
          f'{format_metrics(test_metrics["scalars"], task["type"])}')
    wandb.finish()


def main():
    """Entry point: inference with --eval_only, training otherwise."""
    args = parse_args()
    if args.eval_only:
        run_inference(args)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
