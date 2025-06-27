# ellipse_fit

Fast batch ellipse fitting for [N, 6, 2] point clouds using CUDA.

## Install

```bash
pip install .
```

## Usage

```python
import torch
from ellipse_fit import fit_ellipses

points = torch.randn(100, 6, 2, device='cuda')
results = fit_ellipses(points)
print(results.shape)  # [100, 5]
```