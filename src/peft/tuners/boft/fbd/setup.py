# To install, run
# python setup.py build_ext -i

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.environ["CC"] = "gcc-7"
os.environ["CXX"] = "gcc-7"

USE_NINJA = os.getenv("USE_NINJA") == "1"

setup(
    name="fbd_cuda",
    ext_modules=[
        CUDAExtension(
            "fbd_cuda",
            [
                "fbd_cuda.cpp",
                "fbd_cuda_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
