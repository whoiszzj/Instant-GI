#pragma once

#include <torch/extension.h>

// points: [N, 6, 2], float32, CUDA tensor
// 返回: [N, 5] (cx, cy, a, b, angle)
torch::Tensor fit_ellipses_cuda(torch::Tensor points);
