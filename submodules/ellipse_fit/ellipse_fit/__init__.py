import torch
from . import _C

def fit_ellipses(points: torch.Tensor) -> torch.Tensor:
    """
    Batch fit ellipses for [N, 6, 2] points using CUDA.
    Returns [N, 5]: (cx, cy, a, b, angle)
    """
    assert points.is_cuda and points.shape[1:] == (6, 2)
    return _C.fit_ellipses_cuda(points)