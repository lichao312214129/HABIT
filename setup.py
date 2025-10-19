"""
Setup script for HABIT package
"""

from setuptools import setup, find_packages

setup(
    name='HABIT',
    version='0.1.0',
    description='Habitat Analysis: Biomedical Imaging Toolkit',
    author='lichao19870617@163.com',
    packages=find_packages(),
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
        'tqdm',
        'pyradiomics',
        'scikit-learn',
        'mrmr_selection',
        'pingouin',
        'statsmodels',
        'xgboost',
        'seaborn',
        'shap',
        'pyyaml',
        'torch',
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
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)

