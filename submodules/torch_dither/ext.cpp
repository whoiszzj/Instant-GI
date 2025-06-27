#include <torch/extension.h>
#include "image_dither.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("image_dither", &image_dither_wrapper, "Image Dither");
}