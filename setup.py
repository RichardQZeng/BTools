#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = [
    "label_centerlines",
    "laserchicken",
    "dask",
    "distributed",
    "geopandas",
    "huggingface_hub",
    "matplotlib",
    "pip",
    "pyogrio",
    "psutil",
    "pyqt",
    "ray-default",
    "rioxarray",
    "rpy2",
    "r-essentials",
    "r-lidr",
    "r-rcsf",
    "r-rlas",
    "r-sf",
    "r-sp",
    "r-terra",
    "scikit-image",
    "sphinx-tabs",
    "sphinx-rtd-theme",
    "xarray-spatial"
]

setup_requirements = []

test_requirements = []

setup(
    author="AppliedGRG",
    author_email="appliedgrg@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    description="An advanced forest line feature analysis platform",
    entry_points={
        "console_scripts": [
            "BERATools=beratools.gui:gui_main",
        ],
        "gui_scripts": [
            "BERATools=beratools.gui:gui_main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="BERATools",
    name="beratools",
    packages=find_packages(include=["beratools"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/RichardQZeng/BTools.git",
    version="0.9.4",
    zip_safe=False,
)
