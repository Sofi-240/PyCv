from distutils.command.install_data import install_data
import os
import numpy
import pathlib
import pkg_resources
import platform
import re
from setuptools import find_packages, setup, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from setuptools.command.install_scripts import install_scripts
import shutil
import struct
import sys
from typing import List, Set
from wheel.bdist_wheel import bdist_wheel


########################################################################################################################

if sys.version_info < (3, 5):

    def home_path() -> pathlib.Path:

        return pathlib.Path(os.path.expanduser("~"))

    pathlib.Path.home = home_path

PYTHON_EXE_DIR = os.path.dirname(sys.executable)

SYSTEM_OS_NAME = platform.system()

VERSION = "3.20"
VERSION_TUPLE = pkg_resources.parse_version(VERSION)


_setup = dict(
    name='pycv',
    version="1.0",
    description="image processing package",
    long_description=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"), 'r').read(),
    long_description_content_type="text/markdown",
    author="Sofi.T",
    url="https://github.com/Sofi-240/PyCv",
    packages=["pycv." + p for p in find_packages(where='pycv')],
    package_dir=dict((p, "pycv") for p in find_packages(where='pycv')),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Win32 (MS Windows)",
        "GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: C",
        "Programming Language :: C++",
        "Programming Language :: Python :: Implementation :: ",
    ],
    ext_modules=[
            Extension(
                "pycv._lib._src.c_pycv",
                [
                    "pycv/_lib/_src/c_pycv.c",
                    "pycv/_lib/_src/c_pycv_base.c",
                    "pycv/_lib/_src/c_pycv_filters.c",
                    "pycv/_lib/_src/c_pycv_morphology.c",
                    "pycv/_lib/_src/c_pycv_transform.c",
                    "pycv/_lib/_src/c_pycv_canny.c",
                    "pycv/_lib/_src/c_pycv_draw.c",
                    "pycv/_lib/_src/c_pycv_kd_tree.c",
                    "pycv/_lib/_src/c_pycv_convex_hull.c",
                    "pycv/_lib/_src/c_pycv_cluster.c",
                    "pycv/_lib/_src/c_pycv_minmax_tree.c",
                    "pycv/_lib/_src/c_pycv_haar_like.c",
                    "pycv/_lib/_src/c_pycv_peaks.c",
                    "pycv/_lib/_src/c_pycv_features.c",
                    "pycv/_lib/_src/c_pycv_interpolation.c",
                ],
                include_dirs=[numpy.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
            ),
        ],
    python_requires=">=3.7",
    install_requires=[
        'docutils',
        'BazSpam ==1.1',
        "numpy==1.19.3"
    ],
)

