"""
Frame-wise ViT classifier from timm with ImageNet-pretrained weights.

The pretrained classification head is replaced by a new linear layer with
num_classes outputs (timm re-initializes it when num_classes differs from the
checkpoint). The default model, vit_base_patch16_224.augreg2_in21k_ft_in1k, is
pretrained on ImageNet-21k and fine-tuned on ImageNet-1k; any other timm ViT
tag can be passed instead (e.g. vit_base_patch16_224.augreg_in1k for
ImageNet-1k only).
"""

import timm


def build_vit(model_name, num_classes, pretrained=True, img_size=224, drop_rate=0.0, drop_path_rate=0.1):
    """Create a timm ViT with a new num_classes head.

    If img_size differs from the pretraining resolution, timm resamples the
    position embeddings when the weights are loaded.
    """
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )


def vit_data_config(model):
    """Return the preprocessing config of the pretrained weights (mean, std, interpolation, ...).

    The normalization depends on the weights: AugReg ViTs use mean = std = 0.5,
    not the usual ImageNet statistics.
    """
    return timm.data.resolve_model_data_config(model)
