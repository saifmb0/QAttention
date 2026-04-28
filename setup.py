from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='ragged_cuda',
    ext_modules=[
        CUDAExtension(
            name='ragged_cuda',
            sources=['src/ragged_cuda.cpp', 'src/ragged_cuda_kernel.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math', '-gencode', 'arch=compute_89,code=sm_89']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
