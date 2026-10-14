"""
Shared code for the frame-wise ViT baselines on SAR-RARP50.

- Command line arguments (parse_args) and default paths.
- Task registry (TASKS): split folder, CSV name and class names of each task
  selected with --task.
- Frame dataset: one sample per annotated frame, read from the JPEGs
  written by SAR-RARP50/extract_frames_sarrarp50.py
  (<video_dir>/frames/<frame_id:09d>.jpg).
- Image transforms, data loaders, class weights, optimizer parameter groups,
  LR schedule and seeding.
- Timeline plot of ground-truth vs predicted classes per video (inference).

Default paths are resolved from this file's folder (ViT_Baselines/), so the
run does not depend on the current working directory.
"""

import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from metrics import segment_intervals

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAR_ROOT = os.path.join(SCRIPT_DIR, 'Datasets', 'SAR-RARP50')
PREPROCESS_DIR = os.path.join(SCRIPT_DIR, 'Datasets', 'preprocess_sarrarp50')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')

# One entry per task:
#   splits_dir   folder with the split CSVs ({split}_sarrarp50_{csv_suffix}.csv)
#   type         metric set of metrics.compute_metrics ('multiclass' or 'binary')
#   monitor      validation metric that selects the best checkpoint
#   class_names  index = annotation value
#   class_colors (optional) one color per class; enables the test timeline plot of the inference
TASKS = {
    'gesture_recognition': {
        'splits_dir': os.path.join(PREPROCESS_DIR, 'gesture_recognition'),
        'csv_suffix': 'gestures',
        'type': 'multiclass',
        'monitor': 'accuracy',
        'class_names': [
            'Other',
            'Picking-up needle',
            'Positioning needle tip',
            'Pushing needle',
            'Pulling needle',
            'Tying knot',
            'Cutting suture',
            'Returning needle',
        ],
        # Gray for Other; the hues are ordered so that the gestures that follow each other in
        # time (>= 1% of the transitions) stay distinguishable: worst pair OKLab Delta E 12.3
        # under protan/deutan simulation and 17.8 with normal vision.
        'class_colors': ['#898781', '#eda100', '#008300', '#e87ba4', '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'],
    },
    'error_detection': {
        'splits_dir': os.path.join(PREPROCESS_DIR, 'error_detection'),
        'csv_suffix': 'errors',
        'type': 'binary',
        'monitor': 'ap',
        'class_names': ['Normal', 'Error'],
    },
}
SPLITS = ('train', 'val', 'test')
FRAMES_DIR = 'frames'
FRAME_ASPECT = 16 / 9  # extracted frames are 960x540
VIDEO_FPS = 60  # frame_id is the frame index in the 60 fps video_left.avi

# Plot styling (same tokens as the preprocess_sarrarp50 figures)
SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
BASELINE = '#c3c2b7'


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    """Parse the command line arguments of main.py."""
    parser = argparse.ArgumentParser(
        description='Frame-wise ViT (ImageNet weights) for SAR-RARP50: training or inference (--eval_only).')

    data = parser.add_argument_group('data')
    data.add_argument('--task', choices=list(TASKS), default='gesture_recognition',
                      help='Task: selects the split CSVs and the class names.')
    data.add_argument('--sar_root', default=DEFAULT_SAR_ROOT,
                      help='SAR-RARP50 root containing <split>/video_xx/frames/*.jpg.')
    data.add_argument('--splits_dir', default=None,
                      help="Folder with the task's {train,val,test} CSVs (default: preprocess_sarrarp50/<task>).")
    data.add_argument('--num_workers', type=int, default=8, help='Data loader worker processes.')

    model = parser.add_argument_group('model')
    model.add_argument('--model', default='vit_base_patch16_224.augreg2_in21k_ft_in1k',
                       help='timm model name, including the pretrained tag.')
    model.add_argument('--no_pretrained', action='store_true', help='Start from random weights instead of ImageNet.')
    model.add_argument('--img_size', type=int, default=224,
                       help='Input size; full frames are resized to img_size x img_size.')
    model.add_argument('--drop_path', type=float, default=0.1, help='Stochastic depth rate.')
    model.add_argument('--dropout', type=float, default=0.0, help='Dropout before the classification head.')

    train = parser.add_argument_group('training')
    train.add_argument('--epochs', type=int, default=15)
    train.add_argument('--batch_size', type=int, default=64)
    train.add_argument('--eval_batch_size', type=int, default=128)
    train.add_argument('--lr', type=float, default=5e-5, help='Peak learning rate (AdamW).')
    train.add_argument('--min_lr', type=float, default=1e-6, help='Final learning rate of the cosine schedule.')
    train.add_argument('--weight_decay', type=float, default=0.05)
    train.add_argument('--warmup_epochs', type=float, default=1.0, help='Linear warmup length in epochs.')
    train.add_argument('--grad_clip', type=float, default=1.0, help='Max gradient norm (0 disables clipping).')
    train.add_argument('--label_smoothing', type=float, default=0.0)
    train.add_argument('--class_weighting', choices=['none', 'inverse', 'inverse_sqrt'], default='none',
                       help='Cross-entropy class weights from the training frame counts.')
    train.add_argument('--crop_scale_min', type=float, default=0.8,
                       help='Min area fraction of the training random crop (1 disables cropping).')
    train.add_argument('--color_jitter', type=float, default=0.2,
                       help='Brightness/contrast/saturation jitter for training (0 disables it).')
    train.add_argument('--no_amp', action='store_true', help='Disable bfloat16 autocast.')
    train.add_argument('--log_every', type=int, default=50, help='Log the training loss to wandb every N steps.')

    run = parser.add_argument_group('run')
    run.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                     help='Folder with one subfolder per run (<run name>/best.pt) and the local wandb files.')
    run.add_argument('--device', default='cuda:0')
    run.add_argument('--seed', type=int, default=42)
    run.add_argument('--eval_only', action='store_true',
                     help='Inference: evaluate --checkpoint on the val and test splits of --task and print the '
                          'metrics (no training, no wandb).')
    run.add_argument('--checkpoint', default=None, help='Path of the trained ViT (best.pt) used with --eval_only.')

    logging_group = parser.add_argument_group('wandb')
    logging_group.add_argument('--wandb_project', default='ViT_Baselines')
    logging_group.add_argument('--wandb_entity', default='dl-manrique',
                               help='wandb user or team that owns the project (default: personal account).')
    logging_group.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online',
                               help='"disabled" runs without wandb, e.g. for a quick code test.')

    args = parser.parse_args(argv)
    if args.eval_only and not args.checkpoint:
        parser.error('--eval_only requires --checkpoint')
    if not 0 < args.crop_scale_min <= 1:
        parser.error('--crop_scale_min must be in (0, 1]')
    return args


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def find_frame_dirs(sar_root):
    """Return {video_id: frames folder} for every video folder under sar_root that has a frames/ subfolder."""
    frame_dirs = {}
    for dirpath, dirnames, _ in os.walk(sar_root):
        if FRAMES_DIR in dirnames:
            video_id = os.path.basename(dirpath)
            if video_id in frame_dirs:
                raise ValueError(f'Duplicated video id {video_id}: {frame_dirs[video_id]} and {dirpath}')
            frame_dirs[video_id] = os.path.join(dirpath, FRAMES_DIR)
            dirnames[:] = []  # do not walk into frames/ or segmentation/
    if not frame_dirs:
        raise FileNotFoundError(f'No {FRAMES_DIR}/ folders found under {sar_root}')
    return frame_dirs


def load_split(splits_dir, task, split):
    """Read {split}_sarrarp50_{csv_suffix}.csv of a task (video_id, frame_id, annotation), sorted by video and frame.

    task is an entry of TASKS.
    """
    path = os.path.join(splits_dir, f"{split}_sarrarp50_{task['csv_suffix']}.csv")
    num_classes = len(task['class_names'])
    df = pd.read_csv(path)
    invalid = ~df['annotation'].between(0, num_classes - 1)
    if invalid.any():
        raise ValueError(f'{path}: {int(invalid.sum())} rows with a label outside [0, {num_classes - 1}]')
    return df.sort_values(['video_id', 'frame_id']).reset_index(drop=True)


def check_frames(df, frame_dirs):
    """Raise if a video of df has no frames folder or a frame JPEG is missing."""
    missing_videos = sorted(set(df['video_id']) - set(frame_dirs))
    if missing_videos:
        raise FileNotFoundError(f'No frames folder for videos {missing_videos}')
    missing = []
    for video_id, video in df.groupby('video_id'):
        available = set(os.listdir(frame_dirs[video_id]))
        missing += [f'{video_id}/{f:09d}.jpg' for f in video['frame_id'] if f'{f:09d}.jpg' not in available]
    if missing:
        raise FileNotFoundError(f'{len(missing)} frames missing, e.g. {missing[:5]}')


class FrameDataset(Dataset):
    """One sample per row of a split CSV: (transformed frame, annotation)."""

    def __init__(self, df, frame_dirs, transform):
        self.paths = [os.path.join(frame_dirs[video_id], f'{frame_id:09d}.jpg')
                      for video_id, frame_id in zip(df['video_id'], df['frame_id'])]
        self.labels = df['annotation'].to_numpy(dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            image = img.convert('RGB')
        return self.transform(image), self.labels[idx]


def build_transforms(img_size, data_config, crop_scale_min, color_jitter):
    """Return (train transform, eval transform).

    Both resize the full 16:9 frame to img_size x img_size, so no part of the
    field of view is cut at evaluation. Training adds a random crop covering at
    least crop_scale_min of the frame (aspect ratio around 16:9) and color
    jitter. Normalization comes from the pretrained weights (data_config).
    """
    size = (img_size, img_size)
    interpolation = (T.InterpolationMode.BICUBIC if data_config.get('interpolation') == 'bicubic'
                     else T.InterpolationMode.BILINEAR)
    normalize = [T.ToTensor(), T.Normalize(mean=data_config['mean'], std=data_config['std'])]

    if crop_scale_min < 1:
        train_ops = [T.RandomResizedCrop(size, scale=(crop_scale_min, 1.0),
                                         ratio=(0.75 * FRAME_ASPECT, FRAME_ASPECT / 0.75),
                                         interpolation=interpolation)]
    else:
        train_ops = [T.Resize(size, interpolation=interpolation)]
    if color_jitter > 0:
        train_ops.append(T.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter))

    train_transform = T.Compose(train_ops + normalize)
    eval_transform = T.Compose([T.Resize(size, interpolation=interpolation)] + normalize)
    return train_transform, eval_transform


def seed_worker(worker_id):
    """Seed numpy and random in each data loader worker from the torch worker seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(dataset, batch_size, train, num_workers, seed):
    """Return a DataLoader; the training loader shuffles, drops the last partial batch and keeps its workers alive."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=train and num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def class_weights(labels, num_classes, mode):
    """Return cross-entropy class weights (mean 1) from the training labels, or None if mode is 'none'."""
    if mode == 'none':
        return None
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    if (counts == 0).any():
        raise ValueError(f'Classes without training frames: {np.flatnonzero(counts == 0).tolist()}')
    weights = 1.0 / counts if mode == 'inverse' else 1.0 / np.sqrt(counts)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def param_groups(model, weight_decay):
    """Split the parameters into a weight-decay group and a no-decay group (biases, norms, pos_embed, cls_token)."""
    skip = set(model.no_weight_decay()) if hasattr(model, 'no_weight_decay') else set()
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name in skip:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}]


def warmup_cosine(total_steps, warmup_steps, lr, min_lr):
    """Return a LambdaLR factor: linear warmup to lr, then cosine decay to min_lr at total_steps."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return (min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))) / lr
    return factor


def set_seed(seed):
    """Seed python, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_timelines(df, preds, per_video, class_names, class_colors, video_ids, path, title):
    """Save one panel per video with the ground-truth and predicted class of every frame as colored ribbons.

    df holds the split rows (video_id, frame_id, annotation) in the same order
    as preds; per_video is the per-video metrics table, used in the panel
    titles. The x axis is the video time in minutes (frame_id / 60 fps).
    """
    rows = df.assign(pred=preds)
    scores = per_video.set_index('video_id')
    fig = Figure(figsize=(12, 1.7 * len(video_ids) + 1.2), facecolor=SURFACE, layout='constrained')
    axes = fig.subplots(len(video_ids), 1, squeeze=False)[:, 0]
    for ax, video_id in zip(axes, video_ids):
        video = rows[rows['video_id'] == video_id].sort_values('frame_id')
        frames = video['frame_id'].to_numpy()
        step = np.diff(frames).min()
        for y, column in ((1.0, 'annotation'), (0.0, 'pred')):
            labels, starts, ends = segment_intervals(video[column].to_numpy())
            start_min = frames[starts] / VIDEO_FPS / 60
            width_min = (frames[ends - 1] + step - frames[starts]) / VIDEO_FPS / 60
            for c, color in enumerate(class_colors):
                mask = labels == c
                if mask.any():
                    ax.broken_barh(list(zip(start_min[mask], width_min[mask])), (y, 0.8), facecolors=color,
                                   linewidth=0)
        score = scores.loc[video_id]
        ax.set_title(f"{video_id}   accuracy {score['accuracy']:.3f} · F1@10 {score['f1@10']:.3f} · "
                     f"edit {score['edit']:.3f}", loc='left', fontsize=10, color=INK_PRIMARY)
        ax.set_xlim(0, (frames[-1] + step) / VIDEO_FPS / 60)
        ax.set_ylim(-0.1, 1.9)
        ax.set_yticks([1.4, 0.4], ['Ground truth', 'Prediction'])
        ax.set_facecolor(SURFACE)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(BASELINE)
        ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0)
    axes[-1].set_xlabel('time (min)', color=INK_SECONDARY)
    fig.suptitle(title, x=0.01, ha='left', color=INK_PRIMARY, fontsize=12, fontweight='bold')
    fig.legend(handles=[Patch(facecolor=color, label=name) for name, color in zip(class_names, class_colors)],
               loc='outside lower center', ncol=4, frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
