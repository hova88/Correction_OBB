import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext

if __name__ == '__main__':

    setup(
        name='ops',
        packages=find_packages(exclude=['tools','src']),
        cmdclass={'build_ext': BuildExtension},
        ext_modules=[
            make_cuda_ext(
                name='iou3d_cuda_ext',
                module='ops.iou3d',
                sources=[
                    'src/iou3d_api.cpp',
                    'src/iou3d.cpp',
                    'src/iou3d_kernel.cu',
                ]
            ),
        ],
    )
