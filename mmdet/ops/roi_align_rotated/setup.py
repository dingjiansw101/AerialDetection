from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='roi_align_rotated_cuda',
    ext_modules=[
        CUDAExtension('roi_align_rotated_cuda', [
            'src/roi_align_rotated_cuda.cpp',
            'src/roi_align_rotated_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
