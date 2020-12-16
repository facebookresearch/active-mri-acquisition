#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


def parse_requirements_file(path):
    with open(path) as f:
        reqs = []
        for line in f:
            line = line.strip()
            reqs.append(line.split("==")[0])
    return reqs


reqs_main = parse_requirements_file("requirements/main.txt")
reqs_dev = parse_requirements_file("requirements/dev.txt")

setuptools.setup(
    name="active-mri-acquisition",
    version="0.1.0",
    author="Facebook AI Research",
    description="A reinforcement learning environment for active MRI acquisition.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/active-mri-acquisition/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence :: Medical Imaging",
    ],
    python_requires=">=3.7",
    install_requires=reqs_main,
    extras_require={"dev": reqs_main + reqs_dev},
)
