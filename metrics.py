"""
Evaluation metrics for frame-wise surgical gesture recognition on SAR-RARP50.

Frame-wise metrics pool all frames of a split: accuracy, balanced accuracy,
macro precision / recall / F1 (averaged over the classes present in the ground
truth), per-class scores and the confusion matrix.

Segmental metrics are computed per video on the 10 Hz label sequence ordered by
frame_id and then averaged over videos: F1@{10,25,50} and Edit score. F1@k is
the official SAR-RARP50 implementation (sarrarp50/metrics/action_recognition.py
in github.com/surgical-vision/SAR_RARP50-evaluation), and so is the challenge
action score: sqrt(mean video accuracy * mean video F1@10).

All metrics are fractions in [0, 1].
"""

import numpy as np
import pandas as pd

F1_OVERLAPS = (0.10, 0.25, 0.50)


def confusion_matrix(labels, preds, num_classes):
    """Return the (num_classes x num_classes) confusion matrix; rows are ground truth, columns predictions."""
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    counts = np.bincount(labels * num_classes + preds, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def framewise_metrics(cm):
    """Return (scalar metrics, per-class precision/recall/F1/support) from a confusion matrix.

    Macro averages use the classes with at least one ground-truth frame. A class
    that is never predicted gets precision 0.
    """
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    present = support > 0
    scalars = {
        'accuracy': float(tp.sum() / cm.sum()),
        'balanced_accuracy': float(recall[present].mean()),
        'precision_macro': float(precision[present].mean()),
        'recall_macro': float(recall[present].mean()),
        'f1_macro': float(f1[present].mean()),
    }
    per_class = {'precision': precision, 'recall': recall, 'f1': f1, 'support': support}
    return scalars, per_class


def segment_intervals(labels):
    """Return (segment labels, starts, ends) of the runs of equal consecutive labels; ends are exclusive."""
    labels = np.asarray(labels)
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(labels)]])
    return labels[starts], starts, ends


def segmental_f1(preds, labels, overlap):
    """Segmental F1@k of one video, as in the official SAR-RARP50 toolkit (no background class).

    Each predicted segment is matched to the ground-truth segment of the same
    class with the highest IoU; it is a true positive if that IoU >= overlap and
    the ground-truth segment was not matched before.
    """
    true_labels, true_starts, true_ends = segment_intervals(labels)
    pred_labels, pred_starts, pred_ends = segment_intervals(preds)
    true_used = np.zeros(len(true_labels), dtype=bool)
    tp = 0
    for label, start, end in zip(pred_labels, pred_starts, pred_ends):
        intersection = np.minimum(end, true_ends) - np.maximum(start, true_starts)
        union = np.maximum(end, true_ends) - np.minimum(start, true_starts)
        iou = intersection / union * (label == true_labels)
        idx = int(iou.argmax())
        if iou[idx] >= overlap and not true_used[idx]:
            tp += 1
            true_used[idx] = True
    precision = tp / len(pred_labels)
    recall = tp / len(true_labels)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def edit_score(preds, labels):
    """Edit score of one video: 1 - normalized Levenshtein distance between the segment label sequences.

    Same definition as Lea et al. (2016) and MS-TCN, returned as a fraction
    instead of a percentage. Each row of the dynamic program is vectorized: the
    insertion chain cur[j] = min(cur[j], cur[j-1] + 1) is a running minimum.
    """
    pred_segments = segment_intervals(preds)[0]
    true_segments = segment_intervals(labels)[0]
    num_true = len(true_segments)
    offsets = np.arange(num_true + 1)
    previous = offsets.astype(float)
    for i, label in enumerate(pred_segments, start=1):
        current = np.empty(num_true + 1)
        current[0] = i
        current[1:] = np.minimum(previous[1:] + 1, previous[:-1] + (true_segments != label))
        previous = np.minimum.accumulate(current - offsets) + offsets
    return float(1 - previous[-1] / max(len(pred_segments), num_true))


def per_video_metrics(video_ids, frame_ids, labels, preds):
    """Return one row per video with its frame accuracy, Edit score and F1@k, frames ordered by frame_id."""
    df = pd.DataFrame({'video_id': video_ids, 'frame_id': frame_ids, 'label': labels, 'pred': preds})
    rows = []
    for video_id, video in df.sort_values(['video_id', 'frame_id']).groupby('video_id', sort=True):
        y = video['label'].to_numpy()
        p = video['pred'].to_numpy()
        row = {'video_id': video_id, 'num_frames': len(video), 'accuracy': float((p == y).mean()),
               'edit': edit_score(p, y)}
        for overlap in F1_OVERLAPS:
            row[f'f1@{round(overlap * 100)}'] = segmental_f1(p, y, overlap)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_metrics(video_ids, frame_ids, labels, preds, num_classes, class_names=None):
    """Compute every metric of one split.

    Returns a dict with:
        scalars:          {metric name: float}, frame-wise and video-averaged metrics
                          plus the SAR-RARP50 action score.
        per_class:        DataFrame with precision, recall, F1 and support per class.
        per_video:        DataFrame with accuracy, Edit and F1@k per video.
        confusion_matrix: (num_classes x num_classes) array.
    """
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    cm = confusion_matrix(labels, preds, num_classes)
    scalars, per_class = framewise_metrics(cm)

    per_video = per_video_metrics(video_ids, frame_ids, labels, preds)
    scalars['video_accuracy'] = float(per_video['accuracy'].mean())
    for overlap in F1_OVERLAPS:
        key = f'f1@{round(overlap * 100)}'
        scalars[key] = float(per_video[key].mean())
    scalars['edit'] = float(per_video['edit'].mean())
    scalars['action_score'] = float(np.sqrt(scalars['video_accuracy'] * scalars['f1@10']))

    names = class_names if class_names is not None else [str(c) for c in range(num_classes)]
    per_class_df = pd.DataFrame({'class_id': range(num_classes), 'class_name': names, **per_class})
    return {'scalars': scalars, 'per_class': per_class_df, 'per_video': per_video, 'confusion_matrix': cm}
