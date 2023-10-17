#!/usr/bin/env zsh

set -eu

source  ~/miniconda3/etc/profile.d/conda.sh

conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# conda env remove -n robustpointclouds
conda env create -f environment.yml
conda activate robustpointclouds
mim install "mmcv>=2.0.0rc4,<2.1.0" "mmdet>=3.0.0,<3.1.0" "mmengine>=0.7.1,<1.0.0"
if [ ! -d mmdetection3d ]; then
    git clone https://github.com/open-mmlab/mmdetection3d.git
fi
cd mmdetection3d
git checkout v1.2.0
pip install -e .
cd ..
pip install --upgrade numba
python fixNvPe.py