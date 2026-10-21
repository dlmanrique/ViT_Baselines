"""
Shared code for the frame-wise ViT baselines on SAR-RARP50 and CholecT45.

- Command line arguments (parse_args) with the training defaults of the two
  steps (STEP_DEFAULTS: frame-wise ViT, temporal module) and default paths.
- Task registry (TASKS): dataset, split folder, CSV name, metric type, class
  names and temporal-module sequence sizes of each task selected with --task.
- Split loading: one row per annotated frame with its video, frame id, image
  path and labels.
    SAR-RARP50: split CSVs of preprocess_sarrarp50 and the JPEGs written by
      SAR-RARP50/extract_frames_sarrarp50.py (<video_dir>/frames/<frame_id:09d>.jpg).
    CholecT45: the 45 CholecT45 videos of the official CholecT50 release
      (<cholect_root>/data/VIDxx/<frame_id:06d>.png, multi-hot labels in
      triplet/VIDxx.txt), split by the official 5-fold cross-validation.
- Frame dataset, image transforms, data loaders, class weights, optimizer
  parameter groups, LR schedule and seeding.
- Frame sequences of the temporal module: chunks or whole videos of a split,
  gathered into padded batches.
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
from models.temporal import TEMPORAL_MODELS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAR_ROOT = os.path.join(SCRIPT_DIR, 'Datasets', 'SAR-RARP50')
PREPROCESS_DIR = os.path.join(SCRIPT_DIR, 'Datasets', 'preprocess_sarrarp50')
DEFAULT_CHOLECT_ROOT = os.path.join(SCRIPT_DIR, 'Datasets', 'CholecT50')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')

# One entry per task:
#   dataset      'sarrarp50' or 'cholect45'
#   splits_dir   (sarrarp50) folder with the split CSVs ({split}_sarrarp50_{csv_suffix}.csv)
#   type         metric set of metrics.compute_metrics ('multiclass', 'binary' or 'multilabel')
#   monitor      validation metric that selects the best checkpoint
#   class_names  index = annotation value (None: read from the dataset by resolve_task)
#   class_colors (optional) one color per class; enables the test timeline plot of the inference
#   temporal     second-step sequence sizes in frames: TCM chunks of chunk_size frames overlapping by
#                chunk_overlap (70 s overlapping by 60 s, the MuST Cholec80 setting at 1 fps) and MRTT pooling
#                sizes ks (0.4-0.7 token per median event, the CurConMix+ CholecT45 ratio; median event:
#                5.8 s for gestures, 1.2 s for errors, 10 s for triplets)
TASKS = {
    'gesture_recognition': {
        'dataset': 'sarrarp50',
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
        'temporal': {'chunk_size': 700, 'chunk_overlap': 600, 'ks': [25, 30, 35, 40]},  # 10 Hz
    },
    'error_detection': {
        'dataset': 'sarrarp50',
        'splits_dir': os.path.join(PREPROCESS_DIR, 'error_detection'),
        'csv_suffix': 'errors',
        'type': 'binary',
        'monitor': 'ap',
        'class_names': ['Normal', 'Error'],
        'temporal': {'chunk_size': 350, 'chunk_overlap': 300, 'ks': [2, 3, 4, 5]},  # 5 Hz
    },
    'triplet_recognition': {
        'dataset': 'cholect45',
        'type': 'multilabel',
        'monitor': 'ap_ivt',
        'class_names': None,  # the 100 <instrument,verb,target> triplets of <cholect_root>/dict/triplet.txt
        'temporal': {'chunk_size': 70, 'chunk_overlap': 60, 'ks': [4, 5, 6, 7]},  # 1 fps
    },
}

# Training defaults of the two steps for the arguments left unset on the command line: the frame-wise ViT
# (main.py sweeps) and the temporal module on the frozen ViT
STEP_DEFAULTS = {
    'vit': {'epochs': 15, 'batch_size': 64, 'lr': 5e-5, 'weight_decay': 0.05, 'warmup_epochs': 1.0, 'dropout': 0.0},
    'temporal': {'epochs': 30, 'lr': 1e-4, 'weight_decay': 0.01, 'warmup_epochs': 2.0, 'dropout': 0.1},
}
TEMPORAL_NUM_LAYERS = {'tcm': 2, 'mrtt': 3, 'frame': 0}  # MuST TCM on Cholec80; CurConMix+ MRTT layers per pathway
TCM_BATCH_SIZE = 24  # chunks per step, as MuST; whole videos go one per step
SPLITS = ('train', 'val', 'test')
FRAMES_DIR = 'frames'
FRAME_ASPECT = 16 / 9  # SAR-RARP50 frames are 960x540 and CholecT50 frames 854x480

# Official CholecT45 5-fold cross-validation (Nwoye & Padoy 2022, arXiv 2204.05235), in the order of the
# official CAMMA data loader (github.com/CAMMA-public/cholect45, dataloader_pth.py): --fold k is the test
# split, the other folds are concatenated in key order and their last NUM_VAL_VIDEOS videos are validation.
CHOLECT45_FOLDS = {
    1: [79, 2, 51, 6, 25, 14, 66, 23, 50],
    2: [80, 32, 5, 15, 40, 47, 26, 48, 70],
    3: [31, 57, 36, 18, 52, 68, 10, 8, 73],
    4: [42, 29, 60, 27, 65, 75, 22, 49, 12],
    5: [78, 43, 62, 35, 74, 1, 56, 4, 13],
}
NUM_VAL_VIDEOS = 5
NUM_TRIPLETS = 100
TRIPLET_COLUMNS = [f'tri{i}' for i in range(NUM_TRIPLETS)]
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
    """Parse the command line arguments of main.py.

    Training arguments left unset take the defaults of their step (STEP_DEFAULTS): the ViT, or the temporal
    module when --temporal_model is given.
    """
    parser = argparse.ArgumentParser(
        description='Frame-wise ViT (ImageNet weights) for SAR-RARP50 and CholecT45: training, inference '
                    '(--eval_only) or a temporal module on the frozen ViT (--temporal_model).')

    data = parser.add_argument_group('data')
    data.add_argument('--task', choices=list(TASKS), default=None,
                      help='Task: selects the dataset, the splits and the class names (default gesture_recognition; '
                           'with --temporal_model: the task of --checkpoint, checked if given).')
    data.add_argument('--sar_root', default=DEFAULT_SAR_ROOT,
                      help='SAR-RARP50 root containing <split>/video_xx/frames/*.jpg.')
    data.add_argument('--splits_dir', default=None,
                      help="SAR-RARP50 tasks: folder with the task's {train,val,test} CSVs "
                           "(default: preprocess_sarrarp50/<task>).")
    data.add_argument('--cholect_root', default=DEFAULT_CHOLECT_ROOT,
                      help='triplet_recognition: CholecT50 release root (data/, triplet/, dict/).')
    data.add_argument('--fold', type=int, choices=sorted(CHOLECT45_FOLDS), default=None,
                      help='triplet_recognition: CholecT45 cross-validation fold used as test split (default 1; '
                           'with --temporal_model: the fold of --checkpoint, checked if given).')
    data.add_argument('--num_workers', type=int, default=8, help='Data loader worker processes.')

    model = parser.add_argument_group('model')
    model.add_argument('--model', default='vit_base_patch16_224.augreg2_in21k_ft_in1k',
                       help='timm model name, including the pretrained tag.')
    model.add_argument('--no_pretrained', action='store_true', help='Start from random weights instead of ImageNet.')
    model.add_argument('--img_size', type=int, default=224,
                       help='Input size; full frames are resized to img_size x img_size.')
    model.add_argument('--drop_path', type=float, default=0.1, help='Stochastic depth rate.')
    model.add_argument('--dropout', type=float, default=None,
                       help='Dropout of the trained model: before the ViT head (default 0) or in the temporal '
                            'Transformers (default 0.1).')

    temporal = parser.add_argument_group('temporal module (second step, on the frozen ViT of --checkpoint)')
    temporal.add_argument('--temporal_model', choices=TEMPORAL_MODELS, default=None,
                          help='Train this temporal module on the embeddings of the frozen ViT of --checkpoint: '
                               'tcm (MuST), mrtt (CurConMix+) or frame (MRTT frame-wise branch alone).')
    temporal.add_argument('--chunk_size', type=int, default=None,
                          help='tcm: frames per chunk, 0 for whole videos (default per task, TASKS temporal).')
    temporal.add_argument('--chunk_overlap', type=int, default=None,
                          help='tcm: frames shared by consecutive chunks (default per task).')
    temporal.add_argument('--ks', type=int, nargs='+', default=None,
                          help='mrtt: pooling sizes of the pathways, in frames (default per task).')
    temporal.add_argument('--no_frame_branch', action='store_true', help='mrtt: drop the frame-wise branch.')
    temporal.add_argument('--num_layers', type=int, default=None,
                          help='Transformer layers (default 2 for tcm, 3 per pathway for mrtt).')
    temporal.add_argument('--num_heads', type=int, default=4)
    temporal.add_argument('--ffn_dim', type=int, default=2048,
                          help='Hidden size of the Transformer feed-forward blocks.')
    temporal.add_argument('--tcm_dim', type=int, default=768, help='tcm: Transformer width after the input projection.')

    train = parser.add_argument_group('training')
    train.add_argument('--epochs', type=int, default=None, help='Default 15 for the ViT, 30 for the temporal module.')
    train.add_argument('--batch_size', type=int, default=None,
                       help=f'Default 64 frames for the ViT; {TCM_BATCH_SIZE} chunks for the TCM, 1 video for '
                            'whole-video temporal modules.')
    train.add_argument('--eval_batch_size', type=int, default=128,
                       help='Frames per ViT forward pass at evaluation and when embedding frames.')
    train.add_argument('--lr', type=float, default=None,
                       help='Peak learning rate (AdamW), default 5e-5 for the ViT, 1e-4 for the temporal module.')
    train.add_argument('--min_lr', type=float, default=1e-6, help='Final learning rate of the cosine schedule.')
    train.add_argument('--weight_decay', type=float, default=None,
                       help='Default 0.05 for the ViT, 0.01 for the temporal module.')
    train.add_argument('--warmup_epochs', type=float, default=None,
                       help='Linear warmup length in epochs, default 1 for the ViT, 2 for the temporal module.')
    train.add_argument('--grad_clip', type=float, default=1.0, help='Max gradient norm (0 disables clipping).')
    train.add_argument('--label_smoothing', type=float, default=0.0)
    train.add_argument('--class_weighting', choices=['none', 'inverse', 'inverse_sqrt'], default='none',
                       help='Cross-entropy class weights from the training frame counts.')
    train.add_argument('--crop_scale_min', type=float, default=0.8,
                       help='Min area fraction of the training random crop of the ViT (1 disables cropping).')
    train.add_argument('--color_jitter', type=float, default=0.2,
                       help='Brightness/contrast/saturation jitter for the ViT training (0 disables it).')
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
    run.add_argument('--checkpoint', default=None,
                     help='Trained ViT (best.pt of a training run): evaluated by --eval_only, frozen backbone of '
                          '--temporal_model.')

    logging_group = parser.add_argument_group('wandb')
    logging_group.add_argument('--wandb_project', default='ViT_Baselines')
    logging_group.add_argument('--wandb_entity', default='dl-manrique',
                               help='wandb user or team that owns the project (default: personal account).')
    logging_group.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online',
                               help='"disabled" runs without wandb, e.g. for a quick code test.')
    logging_group.add_argument('--wandb_group', default=None,
                               help='wandb group of the run, e.g. one name shared by the 5 folds of a cross-validation.')

    args = parser.parse_args(argv)
    if not args.temporal_model:  # a temporal run takes them from its ViT checkpoint (run_temporal)
        args.task = args.task or 'gesture_recognition'
        args.fold = args.fold or 1
    if (args.eval_only or args.temporal_model) and not args.checkpoint:
        parser.error('--eval_only and --temporal_model require --checkpoint')
    if args.eval_only and args.temporal_model:
        parser.error('--eval_only evaluates a ViT checkpoint; it cannot be combined with --temporal_model')
    if not 0 < args.crop_scale_min <= 1:
        parser.error('--crop_scale_min must be in (0, 1]')
    if args.class_weighting != 'none' or args.label_smoothing > 0:
        if args.temporal_model:
            parser.error('--class_weighting and --label_smoothing only apply to the ViT training')
        if TASKS[args.task]['type'] == 'multilabel':
            parser.error('--class_weighting and --label_smoothing only apply to the cross-entropy tasks')
    for name, value in STEP_DEFAULTS['temporal' if args.temporal_model else 'vit'].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def resolve_temporal_args(args):
    """Fill the temporal-module settings of --task left unset and clear the ones the module does not use.

    Sequence sizes come from TASKS[task]['temporal']; whole-video modules get
    chunk_size 0, so the wandb config shows what each run actually used.
    """
    defaults = TASKS[args.task]['temporal']
    if args.temporal_model == 'tcm':
        args.chunk_size = defaults['chunk_size'] if args.chunk_size is None else args.chunk_size
        if args.chunk_size == 0:  # whole videos: no overlap
            args.chunk_overlap = 0
        elif args.chunk_overlap is None:
            args.chunk_overlap = defaults['chunk_overlap']
        if args.chunk_size > 0 and not 0 <= args.chunk_overlap < args.chunk_size:
            raise ValueError(f'--chunk_overlap must be in [0, {args.chunk_size}), got {args.chunk_overlap}')
    else:
        args.chunk_size, args.chunk_overlap = 0, 0
    args.ks = (defaults['ks'] if args.ks is None else args.ks) if args.temporal_model == 'mrtt' else None
    if args.num_layers is None:
        args.num_layers = TEMPORAL_NUM_LAYERS[args.temporal_model]
    if args.batch_size is None:
        args.batch_size = TCM_BATCH_SIZE if args.chunk_size > 0 else 1


def resolve_task(args):
    """Return a copy of TASKS[args.task]; triplet recognition gets its class names and component map from the dataset."""
    task = dict(TASKS[args.task])
    if task['dataset'] == 'cholect45':
        task['class_names'], task['component_map'] = load_triplet_dict(args.cholect_root)
    return task


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


def load_split(splits_dir, task, split, frame_dirs):
    """Read {split}_sarrarp50_{csv_suffix}.csv of a SAR-RARP50 task (video_id, frame_id, annotation).

    task is an entry of TASKS. Adds the path of every frame JPEG and returns the
    rows sorted by video and frame.
    """
    path = os.path.join(splits_dir, f"{split}_sarrarp50_{task['csv_suffix']}.csv")
    num_classes = len(task['class_names'])
    df = pd.read_csv(path)
    invalid = ~df['annotation'].between(0, num_classes - 1)
    if invalid.any():
        raise ValueError(f'{path}: {int(invalid.sum())} rows with a label outside [0, {num_classes - 1}]')
    missing_videos = sorted(set(df['video_id']) - set(frame_dirs))
    if missing_videos:
        raise FileNotFoundError(f'No frames folder for videos {missing_videos}')
    df['path'] = [os.path.join(frame_dirs[video_id], f'{frame_id:09d}.jpg')
                  for video_id, frame_id in zip(df['video_id'], df['frame_id'])]
    return df.sort_values(['video_id', 'frame_id']).reset_index(drop=True)


def cholect45_videos(fold, split):
    """Return the video ids (VIDxx) of a split of the CholecT45 cross-validation whose test split is fold."""
    if split == 'test':
        videos = CHOLECT45_FOLDS[fold]
    else:
        trainval = sum([videos for k, videos in CHOLECT45_FOLDS.items() if k != fold], [])
        videos = trainval[-NUM_VAL_VIDEOS:] if split == 'val' else trainval[:-NUM_VAL_VIDEOS]
    return [f'VID{video:02d}' for video in videos]


def load_cholect45_split(cholect_root, fold, split):
    """Read a split of the CholecT45 cross-validation from the official triplet labels.

    Every row of <cholect_root>/triplet/VIDxx.txt (frame id, then 100 binary
    triplet labels) is a sample, as in the official loader. Returns video_id,
    frame_id, path (data/VIDxx/<frame_id:06d>.png) and the multi-hot columns
    tri0..tri99, sorted by video and frame.
    """
    videos = []
    for video_id in cholect45_videos(fold, split):
        path = os.path.join(cholect_root, 'triplet', f'{video_id}.txt')
        rows = np.loadtxt(path, dtype=np.int64, delimiter=',', ndmin=2)
        if rows.shape[1] != NUM_TRIPLETS + 1 or not np.isin(rows[:, 1:], (0, 1)).all():
            raise ValueError(f'{path}: expected a frame id and {NUM_TRIPLETS} binary labels per row')
        video = pd.DataFrame(rows[:, 1:], columns=TRIPLET_COLUMNS)
        video.insert(0, 'video_id', video_id)
        video.insert(1, 'frame_id', rows[:, 0])
        video.insert(2, 'path', [os.path.join(cholect_root, 'data', video_id, f'{frame_id:06d}.png')
                                 for frame_id in rows[:, 0]])
        videos.append(video)
    return pd.concat(videos, ignore_index=True).sort_values(['video_id', 'frame_id']).reset_index(drop=True)


def load_triplet_dict(cholect_root):
    """Return the triplet names and the (100 x 6) triplet component map of <cholect_root>/dict.

    Names are 'instrument,verb,target' (dict/triplet.txt). The map rows are
    triplet ids 0..99 and its columns the triplet, instrument, verb, target,
    instrument-verb and instrument-target ids (dict/maps.txt, the mapping that
    ivtmetrics uses to disentangle the components); the null row (-1) is dropped.
    """
    names = {}
    with open(os.path.join(cholect_root, 'dict', 'triplet.txt')) as f:
        for line in f:
            if line.strip():
                triplet_id, name = line.strip().split(':', 1)
                names[int(triplet_id)] = name
    component_map = np.loadtxt(os.path.join(cholect_root, 'dict', 'maps.txt'), dtype=np.int64, delimiter=',',
                               comments='#', ndmin=2)
    component_map = component_map[component_map[:, 0] >= 0]
    component_map = component_map[np.argsort(component_map[:, 0])]
    if not np.array_equal(component_map[:, 0], np.arange(NUM_TRIPLETS)):
        raise ValueError(f'{cholect_root}/dict/maps.txt does not map the triplet ids 0..{NUM_TRIPLETS - 1}')
    return [names[i] for i in range(NUM_TRIPLETS)], component_map


def split_labels(df, task):
    """Return the labels of a split: class ids, or the (num_frames x 100) multi-hot matrix of a multilabel task.

    np.array copies: with pandas copy-on-write, to_numpy returns a read-only
    view and the data loader would build non-writable label tensors from it.
    """
    if task['type'] == 'multilabel':
        return np.array(df[TRIPLET_COLUMNS], dtype=np.float32)
    return np.array(df['annotation'], dtype=np.int64)


def check_frames(df):
    """Raise if the image file of a row of df is missing (one folder listing per video)."""
    missing = []
    for folder, paths in df.groupby(df['path'].map(os.path.dirname))['path']:
        available = set(os.listdir(folder)) if os.path.isdir(folder) else set()
        missing += [path for path in paths if os.path.basename(path) not in available]
    if missing:
        raise FileNotFoundError(f'{len(missing)} frames missing, e.g. {missing[:5]}')


class FrameDataset(Dataset):
    """One sample per row of a split: (transformed frame, label), labels being the output of split_labels."""

    def __init__(self, df, labels, transform):
        self.paths = df['path'].tolist()
        self.labels = labels
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
# Frame sequences (temporal module)
# ---------------------------------------------------------------------------

def chunk_starts(length, size, overlap):
    """Return the first frames of the chunks of `size` frames, overlapping by `overlap`, that cover a video.

    As in MuST, the last chunk is shifted back to end at the last frame. A
    video of at most `size` frames, or size 0, gives one sequence: the video.
    """
    if size <= 0 or length <= size:
        return [0]
    starts = list(range(0, length - size + 1, size - overlap))
    if starts[-1] + size < length:
        starts.append(length - size)
    return starts


def split_sequences(df, chunk_size, chunk_overlap):
    """Return the (first row, length) of the sequences of a split: the chunks of every video, or whole videos.

    The rows of df are sorted by video and frame, so every video is a contiguous range of rows.
    """
    video_ids = df['video_id'].to_numpy()
    starts = np.flatnonzero(np.r_[True, video_ids[1:] != video_ids[:-1]])
    lengths = np.diff(np.r_[starts, len(df)])
    return [(int(start) + s, min(chunk_size, int(length)) if chunk_size > 0 else int(length))
            for start, length in zip(starts, lengths) for s in chunk_starts(int(length), chunk_size, chunk_overlap)]


def gather_sequences(values, sequences):
    """Stack the rows of the sequences (first row, length) of values into a zero-padded (B x T x ...) batch.

    Returns (batch, padding mask (B x T), True on padded positions).
    """
    max_length = max(length for _, length in sequences)
    batch = values.new_zeros((len(sequences), max_length) + values.shape[1:])
    padding = torch.ones(len(sequences), max_length, dtype=torch.bool, device=values.device)
    for i, (start, length) in enumerate(sequences):
        batch[i, :length] = values[start:start + length]
        padding[i, :length] = False
    return batch, padding


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
