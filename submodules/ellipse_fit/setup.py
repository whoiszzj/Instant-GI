from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="ellipse_fit",
    packages=['ellipse_fit'],
    ext_modules=[
        CUDAExtension(
            name="ellipse_fit._C",
            sources=[
            "cuda/ellipse_fit.cu",
            "ext.cpp"],
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3'],
            verbose=True
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)