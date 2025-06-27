#ifndef IMAGE_DITHER_H
#define IMAGE_DITHER_H

#include <torch/extension.h>
#include "cuda/image_dither_impl.cuh"

torch::Tensor image_dither_wrapper(
        const torch::Tensor &image,  // [H, W]
        const bool random_propagation
);

#endif //IMAGE_DITHER_H