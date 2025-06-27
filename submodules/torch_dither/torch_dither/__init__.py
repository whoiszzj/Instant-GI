import torch
from . import _C


def torch_image_dither(image, random_propagation=False):
    # image: gary image -> torch.Tensor with shape [H, W], range is [0, 1]
    sampled_xy = _C.image_dither(image, random_propagation)
    return sampled_xy
