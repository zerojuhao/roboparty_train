# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages, setup


setup(
    name="robo-newton",
    version="1.0.0",
    license="BSD-3-Clause",
    packages=find_packages(),
    install_requires=[
        "mujoco==3.12.0",
        "mujoco-python-viewer",
        "psutil",
        "joblib>=1.2.0",
        "pynput",
        "opencv-python",
        "onnxruntime-gpu",
        "pyvista",
        "scikit-learn",
    ],
)
