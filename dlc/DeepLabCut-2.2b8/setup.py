#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeplabcut",
    version="2.2b8",
    author="A. & M. Mathis Labs",
    author_email="alexander.mathis@bethgelab.org",
    description="Markerless pose-estimation of user-defined features with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexEMG/DeepLabCut",
    install_requires=[
        # "bayesian-optimization",
        # "certifi",
        # "chardet",
        # "click",
        # "cython",
        # "easydict",
        # "filterpy",
        # "h5py",
        # "intel-openmp",
        # "imgaug",
        # "ipython",
        # "ipython-genutils", # don't think we need this
        # "numba==0.51.1",
        # "matplotlib==3.1.3",
        # "moviepy<=1.0.1",
        # "numpy==1.16.4",
        # "opencv-python-headless",
        # "pandas>=1.0.1",
        # "patsy",
        # "python-dateutil",
        # "pyyaml",
        # "requests",
        # "ruamel.yaml>=0.15.0",
        # "setuptools",
        # "scikit-image",
        # "scikit-learn",
        # "scipy>=1.4",
        # "six",
        # "statsmodels>=0.11",
        "tables",
        "tensorpack==0.9.8", # not available on conda
        # "tqdm",
        # "wheel",
    ],
    scripts=["deeplabcut/pose_estimation_tensorflow/models/pretrained/download.sh"],
    packages=setuptools.find_packages(),
    data_files=[
        (
            "deeplabcut",
            [
                "deeplabcut/pose_cfg.yaml",
                "deeplabcut/inference_cfg.yaml",
                "deeplabcut/pose_estimation_tensorflow/models/pretrained/pretrained_model_urls.yaml",
                "deeplabcut/gui/media/logo.png",
                "deeplabcut/gui/media/dlc_1-01.png",
                "deeplabcut/pose_estimation_tensorflow/lib/nms_cython/nms_grid.pyx",
                "deeplabcut/pose_estimation_tensorflow/lib/nms_cython/nms_grid.cpp",
                "deeplabcut/pose_estimation_tensorflow/lib/nms_cython/include/nms_scoremap.hxx",
                "deeplabcut/pose_estimation_tensorflow/lib/nms_cython/include/andres/marray.hxx",
            ],
        )
    ],
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ),
    entry_points="""[console_scripts]
            dlc=dlc:main""",
)

# https://www.python.org/dev/peps/pep-0440/#compatible-release
