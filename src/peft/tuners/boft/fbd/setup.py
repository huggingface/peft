# To install, run 
# python setup.py build_ext -i
# Ref: https://github.com/pytorch/pytorch/blob/11a40410e755b1fe74efe9eaa635e7ba5712846b/test/cpp_extensions/setup.py#L62

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# USE_NINJA = os.getenv('USE_NINJA') == '1'
os.environ["CC"] = "gcc-7"
os.environ["CXX"] = "gcc-7"

USE_NINJA = os.getenv('USE_NINJA') == '1'

setup(
    name='fbd_cuda',
    ext_modules=[
	CUDAExtension('fbd_cuda', [
        'fbd_cuda.cpp',
        'fbd_cuda_kernel.cu',
        ])
	],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=USE_NINJA)}
)
