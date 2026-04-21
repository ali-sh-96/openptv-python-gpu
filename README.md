# OpenPTV-Python-GPU
GPU-accelerated algorithms for particle tracking velocimetry (PTV) image processing using a relaxation-based approach.

This repository provides Python modules for performing PTV analysis on image sequences using NVIDIA GPUs, with CuPy for acceleration.

## Warning
OpenPTV-Python-GPU is currently under active development, which means it might contain some bugs, and its API is subject to change.

## Installation
First, install CuPy according to your CUDA Toolkit version. For CUDA Toolkit versions 11.2 ~ 11.8 use:

    pip install cupy-cuda11x

For CUDA Toolkit versions 12.x use:

    pip install cupy-cuda12x

Then, use the following command to clone the repository:

    git clone https://github.com/ali-sh-96/openptv-python-gpu
Finally, add the directory of the cloned repository to your PYTHONPATH.

## Contributors
1. [Ali Shirinzad](https://github.com/ali-sh-96)
2. [Khodr Mohamed Jaber](https://github.com/KhodrJ)
3. Pierre E. Sullivan

## CUDA Toolkit installation on Windows
Installing CuPy requires CUDA Toolkit. First, make sure to get the latest supported NVIDIA drivers (https://www.nvidia.com/Download/index.aspx) and install Visual Studio (https://visualstudio.microsoft.com/). After Visual Studio is installed, you may install CUDA Toolkit (https://developer.nvidia.com/cuda-downloads).