# gsplat Bug Fix Version

This version fix some bugs in the original [gsplat](https://github.com/XingtongGe/gsplat.git):

1. **Fix GLM matrix setting error when assigning rotation in "forward2d.py:project_gaussians_2d_scale_rot_forward_kernel" function.** This bug occurs because GLM uses column-major order when initializing matrices, while the original author assumed row-major order. This leads to rotation offsets when initializing with specific angles. Although this doesn't affect the original GaussianImage, it significantly impacts methods that use rotation angles for initialization.

2. **Fix backward bug in "backward.py:rasterize_backward_sum_kernel" function.** [Refer to here](https://github.com/XingtongGe/gsplat/pull/1). Note that after fixing this, in the GaussianImage_RS model, the activation function needs to be set to "torch.abs(self._scaling + self.bound)" to reproduce GaussianImage scores (when using random initialization). Interestingly, this activation function also has some logical bugs, as it only ensures values are greater than 0 rather than greater than self.bound. Changing it to "torch.abs(self._scaling) + self.bound" would lead to decreased scores. However, these issues don't affect our paper's network initialization approach, as the scores remain similar regardless of whether this bug is fixed.
