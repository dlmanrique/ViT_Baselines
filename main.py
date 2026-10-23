"""
Video baselines for the task selected with --task, in two steps:
    gesture_recognition   SAR-RARP50, 8 gesture classes at 10 Hz (multiclass metrics)
    error_detection       SAR-RARP50, normal / error at 5 Hz (binary metrics: AUC, AP, ...)
    triplet_recognition   CholecT45, 100 <instrument, verb, target> triplets at 1 fps, multi-label
                          (video-wise AP of ivtmetrics); one run per cross-validation fold (--fold)

1. Frame-wise ViT (training, the default). Each annotated frame is classified
   on its own by an ImageNet-pretrained ViT from timm, fine-tuned end to end on
   the train split with cross-entropy, or with binary cross-entropy over the
   100 triplets for the multi-label task. The checkpoint with the best
   validation monitor metric of the task (frame-wise accuracy for gestures, AP
   for errors, AP_IVT for triplets) is then evaluated on the test split.
   Triplet recognition follows the official CholecT45 5-fold cross-validation:
   --fold k is the test split, 5 videos of the other folds are the validation
   split and the remaining 31 videos are trained on; the paper results are the
   mean and std of the test AP over the 5 folds.

2. Temporal module (--temporal_model, models/temporal.py) on the frozen ViT of
   --checkpoint, whose task, fold and model it takes (--task and
   --fold are optional and must match the checkpoint when given). Before
   training, the ViT embeds every frame of the three splits once, without
   gradients and with its evaluation transform; the pre-logits embeddings stay
   in GPU memory for the run (about 0.5 GB for gestures) and are never written
   to disk. The module maps the embeddings of consecutive frames of a video to
   per-frame logits of the task classes:
       tcm    MuST Temporal Consistency Module on chunks of --chunk_size frames
              overlapping by --chunk_overlap (0: whole videos); the probabilities
              of a frame are averaged over the chunks that contain it.
       mrtt   CurConMix+ Multi-Resolution Temporal Transformer on whole videos,
              pooling sizes --ks (--no_frame_branch drops its frame-wise branch).
       frame  the MRTT frame-wise branch alone (linear probe, no temporal context).
   Sequence sizes default per task (TASKS temporal in utils.py). The three
   modules share a plain training recipe (AdamW, warmup and cosine LR, the
   loss of the task on every frame) and the checkpoint selection, metrics and
   test evaluation of the ViT.

Both steps log everything to Weights & Biases, in a run named after its start
date and time (YYYY-MM-DD_HH-MM-SS), with job type 'train' (ViT) or
'temporal', tagged with the task, the fold for triplet recognition and the
temporal module, and placed in the --wandb_group group if given
(run_triplet_cv.sh groups its 5 folds); the run config has every argument:
    train/loss, train/lr          every --log_every steps
    train_epoch/*, val/*          every epoch (loss, all metrics of the task and
    val_class_f1/*, mrtt/*        val/best_<monitor> so far; per-class F1 of the ViT,
                                  fusion weights of the MRTT)
    best_val/*, test/*            at the end, for the best checkpoint: all metrics (run
                                  summary) and per-class and per-video tables
    vit_val/*, vit_test/*         temporal runs: the frozen ViT alone, from the same
                                  embedding pass (matches the test of its training run)
The only file written is the best checkpoint, <output_dir>/<run name>/best.pt
(the temporal module only for temporal runs), next to the local wandb run
folder (<output_dir>/wandb).

Sweeps: the arguments are read back from wandb.config after wandb.init, so a
`wandb agent` can override any of them (e.g. --lr). Use val/best_<monitor> as
the sweep metric (goal: maximize): val/best_accuracy for gestures, val/best_ap
for errors, val/best_ap_ivt for triplets.

Inference (--eval_only): loads the ViT in --checkpoint, evaluates it on the val
and test splits of --task (and --fold) and prints every metric of the task in
the terminal. Nothing is logged to wandb and no file is written.

Usage (from any folder, Base_Leo environment):
    python main.py --task gesture_recognition            # ViT training, after `wandb login`
    python main.py --task error_detection --epochs 30 --lr 1e-4
    python main.py --task triplet_recognition --fold 1   # repeat for folds 1..5
    python main.py --task gesture_recognition --checkpoint outputs/<ViT run>/best.pt --temporal_model tcm
    python main.py --eval_only --task error_detection --checkpoint outputs/<run name>/best.pt
"""

import argparse
import json
import math
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

from metrics import TRIPLET_COMPONENTS, compute_metrics
from models.temporal import build_temporal_model
from models.vit import build_vit, vit_data_config
from utils import (EXPORT_TASK_NAMES, SPLITS, TASKS, VIDEO_FPS, FrameDataset, build_loader, build_transforms,
                   check_frames, class_weights, file_sha256, find_frame_dirs, gather_sequences, load_cholect45_split,
                   load_split, param_groups, parse_args, plot_timelines, resolve_task, resolve_temporal_args, set_seed,
                   smooth_probs, split_labels, split_sequences, split_video_ids, warmup_cosine)

NUM_PLOT_VIDEOS = 3  # random test videos drawn in the inference timeline plot


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, amp, grad_clip, epoch, log_every, step):
    """Train for one epoch, logging the loss to wandb every log_every steps.

    Returns the mean training loss of the epoch, its frame accuracy (None for
    multi-label targets) and the global step.
    """
    model.train()
    total_loss, correct, seen, multilabel = 0.0, 0, 0, False
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
        multilabel = labels.ndim == 2
        if not multilabel:
            correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
        window_loss += loss_value
        window_steps += 1
        if step % log_every == 0:
            wandb.log({'step': step, 'train/loss': window_loss / window_steps,
                       'train/lr': scheduler.get_last_lr()[0]})
            window_loss, window_steps = 0.0, 0
    return total_loss / seen, None if multilabel else correct / seen, step


def class_probs(logits, multilabel):
    """Class probabilities of logits (classes on the last dimension): softmax, or a sigmoid per class if multilabel."""
    return logits.sigmoid() if multilabel else logits.softmax(dim=-1)


@torch.no_grad()
def predict_logits(model, loader, device, amp, desc):
    """Return the logits (num_frames x num_classes) of every frame, as a float32 tensor on the CPU in loader order."""
    model.eval()
    logits = []
    for images, _ in tqdm(loader, desc=desc, leave=False):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits.append(model(images.to(device, non_blocking=True)).float().cpu())
    return torch.cat(logits)


def predict(model, loader, device, amp, desc, multilabel):
    """Return the class probabilities (num_frames x num_classes) of every frame, in loader order."""
    return class_probs(predict_logits(model, loader, device, amp, desc), multilabel).numpy()


def split_results(probs, df, task):
    """Return (loss, metrics dict of the task type) of the class probabilities of the frames of a split.

    The loss is the cross-entropy of the true class, or for multi-label tasks
    the binary cross-entropy averaged over frames and classes.
    """
    labels = split_labels(df, task)
    if task['type'] == 'multilabel':
        p = np.clip(probs.astype(np.float64), 1e-12, 1 - 1e-12)
        loss = float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())
    else:
        loss = float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean())
    metrics = compute_metrics(task['type'], df['video_id'], df['frame_id'], labels, probs, task['class_names'],
                              task.get('component_map'))
    return loss, metrics


def evaluate(model, loader, df, device, amp, split, task):
    """Predict a split and return (loss, metrics dict of the task type, class probabilities), see split_results."""
    probs = predict(model, loader, device, amp, split, task['type'] == 'multilabel')
    return (*split_results(probs, df, task), probs)


def load_data(args, task, split_names):
    """Load and check the requested splits of the task.

    Returns {split: DataFrame} with the video_id, frame_id, image path and labels of every frame.
    """
    if task['dataset'] == 'sarrarp50':
        splits_dir = args.splits_dir or task['splits_dir']
        frame_dirs = find_frame_dirs(args.sar_root)
    splits = {}
    for split in split_names:
        if task['dataset'] == 'cholect45':
            splits[split] = load_cholect45_split(args.cholect_root, args.fold, split)
        else:
            splits[split] = load_split(splits_dir, task, split, frame_dirs)
        check_frames(splits[split])
        videos = splits[split]['video_id'].unique()
        print(f'{split}: {len(splits[split])} frames from {len(videos)} videos ({", ".join(videos)})')
    return splits


def load_vit(checkpoint, num_classes, device):
    """Return the ViT of a training checkpoint (loaded with torch.load) on device, in eval mode."""
    trained_args = checkpoint['args']
    if trained_args.get('temporal_model'):
        raise ValueError(f'Expected a ViT checkpoint, got a {trained_args["temporal_model"]} temporal module')
    model = build_vit(trained_args['model'], num_classes, pretrained=False, img_size=trained_args['img_size'])
    model.load_state_dict(checkpoint['model'])
    return model.eval().to(device)


def init_wandb(args, job_type, tags, epoch_prefixes):
    """Start the wandb run of a training, named after its start date and time, and define its x axes.

    Returns (run, run name, arguments read back from the run config: a sweep
    agent may have overridden some).
    """
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, group=args.wandb_group,
                     tags=tags, config=vars(args), mode=args.wandb_mode, dir=output_dir, job_type=job_type)
    args = argparse.Namespace(**dict(run.config))
    wandb.define_metric('step')
    wandb.define_metric('train/*', step_metric='step')
    wandb.define_metric('epoch')
    for prefix in epoch_prefixes:
        wandb.define_metric(f'{prefix}/*', step_metric='epoch')
    print(f'wandb run {run_name} ({run.id})')
    return run, run_name, args


def log_split_results(run, prefix, loss, metrics):
    """Store the metrics of a split in the run summary and log its per-class and per-video tables.

    prefix is 'best_val', 'test', 'vit_val' or 'vit_test', kept apart from the per-epoch val/* charts.
    """
    for name, value in {'loss': loss, **metrics['scalars']}.items():
        run.summary[f'{prefix}/{name}'] = value
    wandb.log({
        f'{prefix}/per_class': wandb.Table(dataframe=metrics['per_class']),
        f'{prefix}/per_video': wandb.Table(dataframe=metrics['per_video']),
    })


def format_metrics(scalars, task_type):
    """Return the main metrics of a split as one console string."""
    if task_type == 'multilabel':
        return ' | '.join(f'AP_{component.upper()} {scalars[f"ap_{component}"]:.4f}'
                          for component in TRIPLET_COMPONENTS)
    if task_type == 'binary':
        return (f"AUC {scalars['auc']:.4f} | AP {scalars['ap']:.4f} | acc {scalars['accuracy']:.4f} | "
                f"precision {scalars['precision']:.4f} recall {scalars['recall']:.4f} F1 {scalars['f1']:.4f}")
    return (f"acc {scalars['accuracy']:.4f} | f1_macro {scalars['f1_macro']:.4f} | "
            f"F1@10 {scalars['f1@10']:.4f} F1@25 {scalars['f1@25']:.4f} F1@50 {scalars['f1@50']:.4f} | "
            f"edit {scalars['edit']:.4f} | score {scalars['action_score']:.4f}")


def print_split_results(split, loss, metrics, class_names):
    """Print every metric of a split: scalars, per-class and per-video tables and the confusion matrix (if any)."""
    print(f'\n{"=" * 30} {split} {"=" * 30}')
    for name, value in {'loss': loss, **metrics['scalars']}.items():
        print(f'{name:<20}{value:.4f}')
    print('\nPer class:')
    print(metrics['per_class'].to_string(index=False, float_format='%.4f'))
    print('\nPer video:')
    print(metrics['per_video'].to_string(index=False, float_format='%.4f'))
    if 'confusion_matrix' in metrics:
        print('\nConfusion matrix (rows: ground truth, columns: predicted class id):')
        print(pd.DataFrame(metrics['confusion_matrix'], index=[f'{c} {name}' for c, name in enumerate(class_names)],
                           columns=range(len(class_names))).to_string())


def export_logits(directory, df, logits, task, args, checkpoint):
    """Write the test logits of every video to <directory>/test/<video_id>.npz, plus <directory>/model.json.

    The files follow the schema of the kinematics late-fusion package: video_id,
    the native frame_ids of the split rows, logits and probabilities: (T x 8)
    softmax for gestures, (T,) binary log-odds z_error - z_normal and their
    sigmoid for errors, (T x 100) sigmoid per triplet for triplets. model.json
    records the task, class order, frame rates, val/test videos, the videos the
    model was fitted on (its training split; the validation split only selected
    the checkpoint) and the checkpoint hashes.
    """
    multilabel, binary = task['type'] == 'multilabel', task['type'] == 'binary'
    if binary:
        logits = logits[:, 1] - logits[:, 0]
        probs = 1 / (1 + np.exp(-logits))
    else:
        probs = class_probs(torch.from_numpy(logits), multilabel).numpy()
    for video_id, rows in df.groupby('video_id').indices.items():
        path = os.path.join(directory, 'test', f'{video_id}.npz')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, video_id=np.array(video_id), frame_ids=df['frame_id'].to_numpy()[rows],
                            logits=logits[rows].astype(np.float32), probabilities=probs[rows].astype(np.float32))

    cholect = task['dataset'] == 'cholect45'
    trained_args = checkpoint['args']
    train_videos = split_video_ids(args, task, 'train')
    meta = {
        'schema_version': 1,
        'model_id': checkpoint.get('wandb_run'),
        'task': EXPORT_TASK_NAMES[args.task],
        'fold': args.fold if cholect else None,
        'class_order': list(range(len(task['class_names']))),
        'class_names': task['class_names'],
        'hz': task['fps'],
        'native_fps': task['fps'] if cholect else VIDEO_FPS,  # unit of frame_ids
        'split_videos': {split: split_video_ids(args, task, split) for split in ('val', 'test')},
        'fit_videos_by_split': {'val': train_videos, 'test': train_videos},
        'fit_manifest_verified': True,
        'checkpoint': os.path.abspath(args.checkpoint),
        'checkpoint_sha256': file_sha256(args.checkpoint),
        'logits_kind': 'multilabel' if multilabel else 'binary_log_odds' if binary else 'multiclass',
        'model': (f'frozen {trained_args["model"]} + {trained_args["temporal_model"]}'
                  if trained_args.get('temporal_model') else f'frame-wise {trained_args["model"]}'),
        'selection': f'validation {task["monitor"]} (best epoch {checkpoint["epoch"]})',
        'scope': 'Fitted on the training split (the ViT and, for temporal modules, the temporal module on the '
                 'frozen ViT); the validation split only selected the checkpoint. ImageNet pretraining is upstream.',
    }
    if trained_args.get('temporal_model'):
        meta['vit_checkpoint'] = trained_args['checkpoint']
        meta['vit_checkpoint_sha256'] = file_sha256(trained_args['checkpoint'])
    with open(os.path.join(directory, 'model.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\nTest logits of {df["video_id"].nunique()} videos written to {os.path.abspath(directory)}')


def run_inference(args):
    """Evaluate a trained ViT, or a temporal module on its frozen ViT, on the val and test splits and print the metrics.

    --checkpoint may be a ViT checkpoint or a temporal-module checkpoint of
    --temporal_model training (whose frozen ViT is loaded from its own
    --checkpoint). With --export_logits, the test logits are also written to
    that folder (export_logits).
    """
    task = resolve_task(args)
    class_names, multilabel = task['class_names'], task['type'] == 'multilabel'
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    trained_args = checkpoint['args']
    trained_task = trained_args.get('task', 'gesture_recognition')  # checkpoints saved before --task existed
    if trained_task != args.task:
        raise ValueError(f'{args.checkpoint} was trained for {trained_task}, not for {args.task}')
    if task['dataset'] == 'cholect45' and trained_args['fold'] != args.fold:
        raise ValueError(f'{args.checkpoint} was trained with test fold {trained_args["fold"]}, not --fold {args.fold}')

    device = torch.device(args.device)
    amp = not args.no_amp and device.type == 'cuda'
    torch.backends.cudnn.benchmark = True

    temporal_model = trained_args.get('temporal_model')
    vit_checkpoint = torch.load(trained_args['checkpoint'], map_location='cpu') if temporal_model else checkpoint
    vit = load_vit(vit_checkpoint, len(class_names), device)
    _, eval_transform = build_transforms(vit_checkpoint['args']['img_size'], vit_data_config(vit),
                                         args.crop_scale_min, args.color_jitter)
    if temporal_model:
        model = build_temporal_model(temporal_model, vit.num_features, len(class_names), trained_args['num_layers'],
                                     trained_args['num_heads'], trained_args['ffn_dim'], trained_args['dropout'],
                                     trained_args['tcm_dim'], trained_args['ks'], not trained_args['no_frame_branch'])
        model.load_state_dict(checkpoint['model'])
        model.to(device)
    monitor = task['monitor']
    print(f'Checkpoint: {os.path.abspath(args.checkpoint)}')
    print(f'Task {args.task} | model {trained_args["model"]}' + (f' + {temporal_model}' if temporal_model else '')
          + f' | epoch {checkpoint["epoch"]} | val {monitor} during training {checkpoint["val_metrics"][monitor]:.4f}')

    splits = load_data(args, task, ('val', 'test'))
    for split, df in splits.items():
        if temporal_model:
            embeddings, _ = embed_frames(vit, df, split_labels(df, task), eval_transform, args, device, amp,
                                         f'embed {split}')
            sequences = split_sequences(df, trained_args['chunk_size'], trained_args['chunk_overlap'])
            probs = predict_sequences(model, embeddings, sequences, trained_args['batch_size'], multilabel, device,
                                      amp)
            logits = predict_sequences(model, embeddings, sequences, trained_args['batch_size'], multilabel, device,
                                       amp, logits=True)
        else:
            loader = build_loader(FrameDataset(df, split_labels(df, task), eval_transform), args.eval_batch_size,
                                  train=False, num_workers=args.num_workers, seed=args.seed)
            logits = predict_logits(vit, loader, device, amp, split)
            probs = class_probs(logits, multilabel).numpy()
            logits = logits.numpy()
        loss, metrics = split_results(probs, df, task)
        print_split_results(split, loss, metrics, class_names)
        if split == 'test' and args.export_logits:
            export_logits(args.export_logits, df, logits, task, args, checkpoint)

        if args.smooth_seconds:
            print(f'\n{split}: ViT probabilities averaged over centered windows ({task["fps"]} frames per second)')
            print(f'  {"none":>13} | {format_metrics(metrics["scalars"], task["type"])}')
            for seconds in args.smooth_seconds:
                window = max(1, round(seconds * task['fps']))
                smoothed_metrics = split_results(smooth_probs(probs, df, window), df, task)[1]
                print(f'  {seconds:5g} s ({window:>3}) | {format_metrics(smoothed_metrics["scalars"], task["type"])}')

        if split == 'test' and 'class_colors' in task:
            video_ids = sorted(random.Random(args.seed).sample(sorted(df['video_id'].unique()), NUM_PLOT_VIDEOS))
            path = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), f'test_timelines_{args.task}.png')
            model_name = f'frozen ViT + {temporal_model}' if temporal_model else 'frame-wise ViT'
            plot_timelines(df, probs.argmax(axis=1), metrics['per_video'], class_names, task['class_colors'],
                           video_ids, path, f'Test set: ground truth vs {model_name} ({args.task})')
            print(f'\nTimelines of {video_ids} saved to {path}')


def run_training(args):
    """Train the frame-wise ViT on --task, keep the best checkpoint and evaluate it, logging everything to wandb."""
    tags = [args.task] + ([f'fold{args.fold}'] if TASKS[args.task]['dataset'] == 'cholect45' else [])
    run, run_name, args = init_wandb(args, 'train', tags, ('train_epoch', 'val', 'val_class_f1'))
    output_dir = os.path.abspath(args.output_dir)

    task = resolve_task(args)
    class_names, monitor = task['class_names'], task['monitor']
    multilabel = task['type'] == 'multilabel'
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    device = torch.device(args.device)
    amp = not args.no_amp and device.type == 'cuda'

    # Data
    splits = load_data(args, task, SPLITS)

    # Model and loaders
    model = build_vit(args.model, len(class_names), pretrained=not args.no_pretrained, img_size=args.img_size,
                      drop_rate=args.dropout, drop_path_rate=args.drop_path)
    data_config = vit_data_config(model)
    train_transform, eval_transform = build_transforms(args.img_size, data_config, args.crop_scale_min,
                                                       args.color_jitter)
    loaders = {
        split: build_loader(FrameDataset(df, split_labels(df, task),
                                         train_transform if split == 'train' else eval_transform),
                            args.batch_size if split == 'train' else args.eval_batch_size,
                            train=split == 'train', num_workers=args.num_workers, seed=args.seed)
        for split, df in splits.items()
    }
    run.config.update({
        'data_config': data_config,
        'num_frames': {split: len(df) for split, df in splits.items()},
        'num_videos': {split: int(df['video_id'].nunique()) for split, df in splits.items()},
        'videos': {split: df['video_id'].unique().tolist() for split, df in splits.items()},
        'torch_version': torch.__version__,
        'timm_version': timm.__version__,
        'cuda_device': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
    })
    model.to(device)
    print(f'Model {args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters | '
          f'mean {data_config["mean"]} std {data_config["std"]}')

    # Optimization
    if multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else:
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

        # The 100 per-triplet APs are only logged in the final per-class tables
        class_f1 = ({} if multilabel else
                    {f'val_class_f1/{row.class_name}': row.f1 for row in val_metrics['per_class'].itertuples()})
        wandb.log({
            'epoch': epoch,
            'train_epoch/loss': train_loss,
            **({} if train_acc is None else {'train_epoch/accuracy': train_acc}),
            'val/loss': val_loss,
            **{f'val/{name}': value for name, value in scalars.items()},
            f'val/best_{monitor}': best_value,
            **class_f1,
        })
        train_acc_text = '' if train_acc is None else f' acc {train_acc:.4f}'
        print(f'epoch {epoch}/{args.epochs} | train loss {train_loss:.4f}{train_acc_text} | '
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


# ---------------------------------------------------------------------------
# Second step: temporal module on the frozen ViT
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_frames(vit, df, labels, transform, args, device, amp, desc):
    """Embed every frame of a split with the frozen ViT, in row order.

    Returns (pre-logits embeddings, ViT logits): float32 tensors on device of
    shape (num_frames x embed_dim) and (num_frames x num_classes).
    """
    loader = build_loader(FrameDataset(df, labels, transform), args.eval_batch_size, train=False,
                          num_workers=args.num_workers, seed=args.seed)
    embeddings, logits = [], []
    for images, _ in tqdm(loader, desc=desc, leave=False):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            features = vit.forward_head(vit.forward_features(images.to(device, non_blocking=True)), pre_logits=True)
            logits.append(vit.head(features).float())
        # The pooled token is a view of all the tokens of the batch (B x 197 x C): keep a copy of it only
        embeddings.append(features.float().clone())
    return torch.cat(embeddings), torch.cat(logits)


def train_temporal_epoch(model, embeddings, labels, sequences, criterion, optimizer, scheduler, args, device, amp,
                         epoch, step, generator):
    """Train the temporal module on the sequences in random order, logging the loss every log_every steps.

    The loss of a batch is computed on its non-padded frames. Returns the mean
    training loss per frame of the epoch and the global step.
    """
    model.train()
    order = torch.randperm(len(sequences), generator=generator).tolist()
    total_loss, total_frames = 0.0, 0
    window_loss, window_steps = 0.0, 0
    for first in tqdm(range(0, len(order), args.batch_size), desc=f'train {epoch}', leave=False):
        batch = [sequences[i] for i in order[first:first + args.batch_size]]
        x, padding = gather_sequences(embeddings, batch)
        y, _ = gather_sequences(labels, batch)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x, padding_mask=padding)
        valid = ~padding
        loss = criterion(logits.float()[valid], y[valid])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        loss_value, frames = loss.item(), int(valid.sum())
        total_loss += loss_value * frames
        total_frames += frames
        window_loss += loss_value
        window_steps += 1
        if step % args.log_every == 0:
            wandb.log({'step': step, 'train/loss': window_loss / window_steps,
                       'train/lr': scheduler.get_last_lr()[0]})
            window_loss, window_steps = 0.0, 0
    return total_loss / total_frames, step


@torch.no_grad()
def predict_sequences(model, embeddings, sequences, batch_size, multilabel, device, amp, logits=False):
    """Return the (num_frames x num_classes) class probabilities of a split from its sequences, in row order.

    The probabilities (or, with logits=True, the logits) of a frame are
    averaged over the sequences that contain it (overlapping TCM chunks); with
    whole videos each frame has one.
    """
    model.eval()
    scores, counts = None, torch.zeros(len(embeddings), device=device)
    for first in range(0, len(sequences), batch_size):
        batch = sequences[first:first + batch_size]
        x, padding = gather_sequences(embeddings, batch)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            batch_scores = model(x, padding_mask=padding).float()
        if not logits:
            batch_scores = class_probs(batch_scores, multilabel)
        if scores is None:
            scores = torch.zeros(len(embeddings), batch_scores.shape[-1], device=device)
        for i, (start, length) in enumerate(batch):
            scores[start:start + length] += batch_scores[i, :length]
            counts[start:start + length] += 1
    return (scores / counts.unsqueeze(1)).cpu().numpy()


def run_temporal(args):
    """Train --temporal_model on the frozen ViT of --checkpoint, keep the best checkpoint and evaluate it."""
    vit_checkpoint = torch.load(args.checkpoint, map_location='cpu')
    vit_args = vit_checkpoint['args']
    # Task, fold and ViT come from the checkpoint, so the run config shows the ones used;
    # a --task or --fold given on the command line must match it
    trained_task = vit_args.get('task', 'gesture_recognition')  # checkpoints saved before --task existed
    if args.task not in (None, trained_task):
        raise ValueError(f'{args.checkpoint} was trained for {trained_task}, not for --task {args.task}')
    if TASKS[trained_task]['dataset'] == 'cholect45' and args.fold not in (None, vit_args['fold']):
        raise ValueError(f'{args.checkpoint} was trained with test fold {vit_args["fold"]}, not --fold {args.fold}')
    args.task, args.fold = trained_task, vit_args.get('fold', 1)
    args.model, args.img_size = vit_args['model'], vit_args['img_size']
    args.checkpoint = os.path.abspath(args.checkpoint)
    resolve_temporal_args(args)
    tags = ([args.task] + ([f'fold{args.fold}'] if TASKS[args.task]['dataset'] == 'cholect45' else [])
            + [args.temporal_model, 'temporal'])
    run, run_name, args = init_wandb(args, 'temporal', tags, ('train_epoch', 'val', 'mrtt'))
    output_dir = os.path.abspath(args.output_dir)

    task = resolve_task(args)
    class_names, monitor = task['class_names'], task['monitor']
    multilabel = task['type'] == 'multilabel'
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    device = torch.device(args.device)
    amp = not args.no_amp and device.type == 'cuda'

    # Data and frozen ViT: every frame is embedded once, with the evaluation transform
    splits = load_data(args, task, SPLITS)
    label_arrays = {split: split_labels(df, task) for split, df in splits.items()}
    labels = {split: torch.as_tensor(array, device=device) for split, array in label_arrays.items()}
    vit = load_vit(vit_checkpoint, len(class_names), device).requires_grad_(False)
    _, eval_transform = build_transforms(args.img_size, vit_data_config(vit), args.crop_scale_min, args.color_jitter)
    embeddings, vit_results = {}, {}
    for split in SPLITS:
        embeddings[split], vit_logits = embed_frames(vit, splits[split], label_arrays[split], eval_transform, args,
                                                     device, amp, f'embed {split}')
        if split != 'train':
            vit_results[split] = split_results(class_probs(vit_logits, multilabel).cpu().numpy(), splits[split],
                                               task)
    del vit  # only its embeddings are needed from here on
    print(f'Frozen ViT {args.model} of {args.checkpoint} (training run {vit_checkpoint.get("wandb_run")}, '
          f'val {monitor} during training {vit_checkpoint["val_metrics"][monitor]:.4f})')
    for split in ('val', 'test'):
        print(f'  ViT {split}: {format_metrics(vit_results[split][1]["scalars"], task["type"])}')

    # Temporal module
    sequences = {split: split_sequences(df, args.chunk_size, args.chunk_overlap) for split, df in splits.items()}
    embed_dim = embeddings['val'].shape[1]
    model = build_temporal_model(args.temporal_model, embed_dim, len(class_names), args.num_layers, args.num_heads,
                                 args.ffn_dim, args.dropout, args.tcm_dim, args.ks, not args.no_frame_branch)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    run.config.update({
        'vit_run': vit_checkpoint.get('wandb_run'),
        'vit_wandb_id': vit_checkpoint.get('wandb_id'),
        'embed_dim': embed_dim,
        'num_params': num_params,
        'num_frames': {split: len(df) for split, df in splits.items()},
        'num_sequences': {split: len(seqs) for split, seqs in sequences.items()},
        'videos': {split: df['video_id'].unique().tolist() for split, df in splits.items()},
        'torch_version': torch.__version__,
        'timm_version': timm.__version__,
        'cuda_device': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
    })
    print(f'Temporal module {args.temporal_model}: {num_params / 1e6:.1f}M parameters | sequences '
          + ', '.join(f'{split} {len(seqs)}' for split, seqs in sequences.items())
          + (f' (chunks of {args.chunk_size}, overlap {args.chunk_overlap})' if args.chunk_size else ' (whole videos)')
          + (f' | ks {args.ks}' if args.ks else ''))

    # Optimization
    criterion = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)
    steps_per_epoch = math.ceil(len(sequences['train']) / args.batch_size)
    scheduler = LambdaLR(optimizer, warmup_cosine(args.epochs * steps_per_epoch,
                                                  round(args.warmup_epochs * steps_per_epoch), args.lr, args.min_lr))
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    # Training: only the best checkpoint by the validation monitor metric of the task is saved
    best_path = os.path.join(output_dir, run_name, 'best.pt')
    best_value, best_epoch, best_val, step = -np.inf, 0, None, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, step = train_temporal_epoch(model, embeddings['train'], labels['train'], sequences['train'],
                                                criterion, optimizer, scheduler, args, device, amp, epoch, step,
                                                generator)
        probs = predict_sequences(model, embeddings['val'], sequences['val'], args.batch_size, multilabel, device, amp)
        val_loss, val_metrics = split_results(probs, splits['val'], task)
        scalars = val_metrics['scalars']

        is_best = scalars[monitor] > best_value
        if is_best:
            best_value, best_epoch = scalars[monitor], epoch
            best_val = (val_loss, val_metrics)
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_metrics': scalars, 'args': vars(args),
                        'wandb_run': run_name, 'wandb_id': run.id}, best_path)

        fusion = ({f'mrtt/{name}': value for name, value in model.fusion_weights().items()}
                  if args.temporal_model == 'mrtt' else {})
        wandb.log({
            'epoch': epoch,
            'train_epoch/loss': train_loss,
            'val/loss': val_loss,
            **{f'val/{name}': value for name, value in scalars.items()},
            f'val/best_{monitor}': best_value,
            **fusion,
        })
        print(f'epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | '
              f'{format_metrics(scalars, task["type"])} | {time.time() - start:.0f}s'
              + (f' | new best {monitor}' if is_best else ''))

    # Final evaluation of the best checkpoint, next to the frozen ViT alone
    run.summary['best_epoch'] = best_epoch
    run.summary['best_checkpoint'] = best_path
    log_split_results(run, 'best_val', *best_val)
    model.load_state_dict(torch.load(best_path, map_location='cpu')['model'])
    probs = predict_sequences(model, embeddings['test'], sequences['test'], args.batch_size, multilabel, device, amp)
    test_loss, test_metrics = split_results(probs, splits['test'], task)
    log_split_results(run, 'test', test_loss, test_metrics)
    for split in ('val', 'test'):
        log_split_results(run, f'vit_{split}', *vit_results[split])
    print(f'best epoch {best_epoch} (val {monitor} {best_value:.4f}) | test loss {test_loss:.4f} | '
          f'{format_metrics(test_metrics["scalars"], task["type"])}')
    print(f'frozen ViT alone | test loss {vit_results["test"][0]:.4f} | '
          f'{format_metrics(vit_results["test"][1]["scalars"], task["type"])}')
    wandb.finish()


def main():
    """Entry point: inference with --eval_only, temporal module with --temporal_model, ViT training otherwise."""
    args = parse_args()
    if args.eval_only:
        run_inference(args)
    elif args.temporal_model:
        run_temporal(args)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
