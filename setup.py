# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os

# conda 安装
# import shutil
# from pathlib import Path
# import nvidia

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

# conda 安装
# def change_cpath():
#     NVIDIA_BASE = Path(nvidia.__file__).parent

#     include_dirs = []
#     for include_dir in NVIDIA_BASE.rglob("include"):
#         if include_dir.is_dir():
#             include_dirs.append(str(include_dir.resolve()))

#     # 拼接成 CPATH 格式（冒号分隔）
#     if include_dirs:
#         new_cpath = ":".join(include_dirs)
#         current_cpath = os.environ.get("CPATH", "")
#         # 避免重复添加
#         if new_cpath not in current_cpath:
#             os.environ["CPATH"] = new_cpath + (":" + current_cpath if current_cpath else "")
#             print(f"✅ Updated CPATH with nvidia includes: {new_cpath}")
#         else:
#             print("ℹ️ CPATH already contains nvidia include paths.")
#     else:
#         print("⚠️ No 'include' directories found under nvidia package.")

# conda 安装
# def set_gxx_gcc():
#     # 查找可执行文件路径（相当于 which）
#     cc_path = shutil.which("x86_64-conda-linux-gnu-gcc")
#     cxx_path = shutil.which("x86_64-conda-linux-gnu-g++")

#     # 设置环境变量（如果找到了路径）
#     if cc_path:
#         os.environ["CC"] = cc_path
#     else:
#         raise FileNotFoundError("x86_64-conda-linux-gnu-gcc not found in PATH")

#     if cxx_path:
#         os.environ["CXX"] = cxx_path
#     else:
#         raise FileNotFoundError("x86_64-conda-linux-gnu-g++ not found in PATH")


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "maskrcnn_benchmark", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": ["-std=c++17"]}
    define_macros = []

    # conda 安装
    # change_cpath()
    # set_gxx_gcc()

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    # conda 安装
    # sources = [os.path.join(extensions_dir, s) for s in sources]

    # uv 安装
    sources = [os.path.relpath(src, this_dir) for src in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "maskrcnn_benchmark._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="maskrcnn_benchmark",
    version="0.1",
    author="fmassa",
    url="https://github.com/facebookresearch/maskrcnn-benchmark",
    description="object detection in pytorch",
    packages=find_packages(
        exclude=(
            "configs",
            "tests",
        )
    ),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
