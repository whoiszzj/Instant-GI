from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

os.path.dirname(os.path.abspath(__file__))

setup(
    name="torch_dither",
    packages=['torch_dither'],
    ext_modules=[
        CUDAExtension(
            name="torch_dither._C",
            sources=[
                "cuda/image_dither_impl.cu",
                "image_dither.cpp",
                "ext.cpp"
            ],
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3'],
            verbose=True
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
