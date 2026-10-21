"""
Temporal modules over per-frame ViT embeddings (second step of main.py,
--temporal_model), for every video task: they map the (B x T x C) embeddings of consecutive frames to
(B x T x num_classes) logits, one row per frame, so the number of classes only
sets the size of the output heads. padding_mask (B x T) is True on padded
frames of sequences shorter than T.

TCM        Temporal Consistency Module of MuST (Perez et al., MICCAI 2024,
           github.com/BCV-Uniandes/MuST, must/models/video_model_builder.py):
           linear projection, sinusoidal positions and a post-norm Transformer
           encoder over a chunk of frames.
MRTT       Multi-Resolution Temporal Transformer of CurConMix+ (Jeon et al.,
           IEEE TMI, github.com/yongyong98/CurConMix-TMI,
           temporal_model2.MultiResTemporalModel2 with fusion='learned'): one
           pre-norm Transformer pathway per pooling size k plus a frame-wise
           branch, fused with learned weights.
FrameHead  the frame-wise branch of the MRTT alone (no temporal context).

Architecture defaults follow MuST (Cholec80) and CurConMix+ (CholecT45); the
training recipe is the one of main.py, not the papers'.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TEMPORAL_MODELS = ('tcm', 'mrtt', 'frame')


class SinePositionalEncoding(nn.Module):
    """Add the sinusoidal position encoding of Vaswani et al. (2017) to (B x T x C) inputs, for any T."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer('div_term', torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                                                   * (-math.log(10000.0) / dim)), persistent=False)

    def forward(self, x):
        dim = x.size(2)
        angles = torch.arange(x.size(1), device=x.device, dtype=torch.float32).unsqueeze(1) * self.div_term
        pe = torch.zeros(x.size(1), dim, device=x.device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles[:, :dim // 2])
        return x + pe.to(x.dtype)


class FrameHead(nn.Module):
    """LayerNorm, dropout and a linear head on each frame embedding (the MRTT frame-wise branch)."""

    def __init__(self, dim, num_classes, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, padding_mask=None):
        return self.head(self.drop(self.norm(x)))


class TCM(nn.Module):
    """Temporal Consistency Module of MuST: per-frame logits of a chunk of frame embeddings.

    Linear projection to dim, sinusoidal positions with dropout, a post-norm
    Transformer encoder in which every frame of the chunk attends to every
    other frame, and a linear head per frame. MuST runs it on chunks of
    chunk_size frames overlapping by chunk_overlap frames and averages the
    predictions of a frame over the chunks that contain it.
    """

    def __init__(self, in_dim, num_classes, dim=768, num_layers=2, num_heads=4, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(in_dim, dim)
        self.positional_encoding = SinePositionalEncoding(dim)
        self.pos_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=ffn_dim, dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, padding_mask=None):
        x = self.pos_drop(self.positional_encoding(self.embedding(x)))
        return self.head(self.encoder(x, src_key_padding_mask=padding_mask))


class TemporalPathway(nn.Module):
    """One MRTT resolution: masked mean over windows of k frames, a pre-norm Transformer at T/k tokens,
    a linear head and nearest upsampling of the logits back to T frames."""

    def __init__(self, dim, num_classes, k, num_layers=3, num_heads=4, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.k = k
        self.positional_encoding = SinePositionalEncoding(dim)
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=ffn_dim, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, valid):
        """x: (B x T x C) embeddings; valid: (B x T) bool, False on padded frames."""
        batch, length, dim = x.shape
        pad = -length % self.k
        x = F.pad(x, (0, 0, 0, pad))
        valid = F.pad(valid, (0, pad), value=False)
        weights = valid.view(batch, -1, self.k, 1).to(x.dtype)
        counts = weights.sum(dim=2)                                              # (B x T/k x 1)
        pooled = (x.view(batch, -1, self.k, dim) * weights).sum(dim=2) / counts.clamp_min(1)
        h = self.encoder(self.positional_encoding(pooled), src_key_padding_mask=counts.squeeze(-1) == 0)
        return self.head(h).repeat_interleave(self.k, dim=1)[:, :length]       # nearest upsampling


class MRTT(nn.Module):
    """Multi-Resolution Temporal Transformer of CurConMix+.

    One TemporalPathway per pooling size in ks, mixed with learned softmax
    weights, and a FrameHead on each embedding fused with them by
    alpha = sigmoid(a), starting at 0.5. Without the frame branch the output
    is the pathway mix alone. The Transformers work on the embedding
    dimension itself (no input projection), so num_heads must divide it.
    """

    def __init__(self, dim, num_classes, ks=(4, 5, 6, 7), num_layers=3, num_heads=4, ffn_dim=2048, dropout=0.1,
                 frame_branch=True, frame_dropout=0.1):
        super().__init__()
        self.frame_branch = FrameHead(dim, num_classes, frame_dropout) if frame_branch else None
        self.pathways = nn.ModuleList(TemporalPathway(dim, num_classes, k, num_layers, num_heads, ffn_dim, dropout)
                                      for k in ks)
        self.pathway_logits = nn.Parameter(torch.zeros(len(ks)))  # softmax mixing weights of the pathways
        self.fusion_logit = nn.Parameter(torch.tensor(0.0))       # alpha = sigmoid(0) = 0.5 at the start

    def forward(self, x, padding_mask=None):
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if padding_mask is None else ~padding_mask
        weights = self.pathway_logits.softmax(dim=0)
        temporal_logits = sum(w * pathway(x, valid) for w, pathway in zip(weights, self.pathways))
        if self.frame_branch is None:
            return temporal_logits
        alpha = torch.sigmoid(self.fusion_logit)
        return alpha * self.frame_branch(x) + (1 - alpha) * temporal_logits

    @torch.no_grad()
    def fusion_weights(self):
        """Return {name: value} of the learned fusion alpha and pathway weights, for logging."""
        weights = {f'pathway_k{p.k}': float(w) for p, w in zip(self.pathways, self.pathway_logits.softmax(dim=0))}
        if self.frame_branch is not None:
            weights['alpha_frame'] = float(torch.sigmoid(self.fusion_logit))
        return weights


def build_temporal_model(name, in_dim, num_classes, num_layers, num_heads, ffn_dim, dropout, tcm_dim, ks,
                         frame_branch):
    """Create the temporal module `name` (one of TEMPORAL_MODELS) with num_classes outputs per frame."""
    if name == 'tcm':
        return TCM(in_dim, num_classes, tcm_dim, num_layers, num_heads, ffn_dim, dropout)
    if name == 'mrtt':
        return MRTT(in_dim, num_classes, tuple(ks), num_layers, num_heads, ffn_dim, dropout, frame_branch)
    if name == 'frame':
        return FrameHead(in_dim, num_classes)
    raise ValueError(f'Unknown temporal model {name}')
