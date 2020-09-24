import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    reqs = []
    for line in f:
        line = line.strip()
        reqs.append(line.split("==")[0])

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
    install_requires=reqs,
)
