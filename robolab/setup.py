# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages, setup


setup(
    name="robolab",
    version="1.0.0",
    license="BSD-3-Clause",
    packages=find_packages(),
    package_data={"": ["*.pyi"]},
    entry_points={"isaaclab.tasks": ["robolab = robolab.tasks"]},
    python_requires=">=3.12,<3.13",
    install_requires=[
        "psutil",
        "joblib>=1.2.0",
        "opencv-python",
        "scikit-learn",
    ],
    # LIVE sim2sim/export extras beyond official pyvista-only tools.
    extras_require={
        "tools": [
            "pyvista",
            "mujoco-python-viewer",
            "onnxruntime-gpu",
            "pynput",
        ],
    },
)
