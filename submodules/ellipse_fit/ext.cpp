#include <torch/extension.h>
#include "cuda/ellipse_fit.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fit_ellipses_cuda", &fit_ellipses_cuda, "Batch Ellipse Fit (CUDA)");
}
