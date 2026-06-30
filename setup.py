"""
Setup script for HABIT package
TEST

"""

import os

import numpy as np
from setuptools import Extension, find_packages, setup

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CEXT_SRC = os.path.join(
    _ROOT,
    "habit",
    "core",
    "habitat_analysis",
    "clustering_features",
    "supervoxel_cext",
    "src",
)
_SV_CMATRICES_MODULE = (
    "habit.core.habitat_analysis.clustering_features.supervoxel_cext._sv_cmatrices"
)

setup(
    name='HABIT',
    version='0.1.0',
    description='Habitat Analysis: Biomedical Imaging Toolkit',
    author='lichao19870617@163.com',
    license='HABIT Software License',
    packages=find_packages(),
    ext_modules=[
        Extension(
            _SV_CMATRICES_MODULE,
            [
                os.path.join(_CEXT_SRC, "_sv_cmatrices.c"),
                os.path.join(_CEXT_SRC, "sv_cmatrices.c"),
            ],
            include_dirs=[_CEXT_SRC, np.get_include()],
        ),
    ],
    install_requires=[
        'click>=8.0',
        'SimpleITK==2.2.1',
        'antspyx==0.4.2',
        'opencv-python',
        'numpy',
        'matplotlib',
        'trimesh',
        'scipy',
        'openpyxl',
        'pandas',
        'pyarrow',
        'tqdm',
        'pyradiomics',
        'scikit-learn',
        'scikit-image',
        'mrmr_selection',
        'pingouin',
        'statsmodels',
        'xgboost',
        'seaborn',
        'shap',
        'pyyaml',
        'lifelines',
    ],
    entry_points={
        'console_scripts': [
            'habit=habit.cli:cli',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)

