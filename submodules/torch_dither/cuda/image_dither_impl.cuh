#ifndef IMAGE_DITHER_IMPL_CUH
#define IMAGE_DITHER_IMPL_CUH

#include <torch/extension.h>
#include "auxiliary.h"

torch::Tensor image_dither_cuda(
        const int H,
        const int W,
        const bool random_propagation,
        const float *image
);

#endif //IMAGE_DITHER_IMPL_CUH