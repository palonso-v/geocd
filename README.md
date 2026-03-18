# GeoCD

Official implementation of the ICCV 2025 E2E3D Workshop paper "GeoCD: A Differential Local Approximation for Geodesic Chamfer Distance".

https://arxiv.org/abs/2506.23478

## Overview

This repository provides code for:

- pretraining point cloud autoencoders with standard Chamfer Distance (CD)
- finetuning pretrained models with **GeoCD**
- evaluating reconstruction models using CD and GeoCD

The current implementation supports the following backbones:

- `AE`
- `PTv3`

## Installation

Clone the repository:

git clone https://github.com/palonso-v/geocd.git
cd geocd

Create and activate a conda environment:

conda create -n geocd python=3.10
conda activate geocd

## Data Preparation

Make sure your datasets are prepared in the format expected by load_data(...).

Currently supported datasets include:

modelnet40

If you use custom dataset paths or preprocessing, update the data-loading utilities accordingly.

## Training

1. Pretrain with Chamfer Distance

python train.py \
    --backbone AE \
    --dataset modelnet40 \
    --batch_size 12 \
    --nepochs 100

Example with PTv3:

python train.py \
    --backbone PTv3 \
    --dataset modelnet40 \
    --batch_size 4 \
    --nepochs 100

2. Finetune with GeoCD

python finetune.py \
    --backbone AE \
    --dataset modelnet40 \
    --batch_size 1 \
    --nepochs 1 \
    --checkpoint_path ./outputs/geocd_AE_modelnet40_cd_best.pth \
    --k_value 5 \
    --nhops 2

3. Evaluation

python evaluate.py \
    --backbone AE \
    --dataset modelnet40 \
    --batch_size 1 \
    --checkpoint_path ./outputs/geocd_AE_modelnet40_geocd_best.pth \
    --split val \
    --metrics cd geocd \
    --k_value 5 \
    --nhops 2

## Checkpoints

During training, checkpoints are saved to --save_dir.

Typical filenames are:

geocd_AE_modelnet40_cd_best.pth

geocd_AE_modelnet40_cd_last.pth

geocd_AE_modelnet40_geocd_best.pth

geocd_AE_modelnet40_geocd_last.pth

## Citation

If you find this repository useful, please cite:

@ARTICLE{2025arXiv250623478A,
  author        = {{Alonso}, Pedro and {Li}, Tianrui and {Li}, Chongshou},
  
  title         = "{GeoCD: A Differential Local Approximation for Geodesic Chamfer Distance}",
  journal       = {arXiv e-prints},
  keywords      = {Computer Vision and Pattern Recognition},
  year          = 2025,
  month         = jun,
  eid           = {arXiv:2506.23478},
  pages         = {arXiv:2506.23478},
  doi           = {10.48550/arXiv.2506.23478},
  archivePrefix = {arXiv},
  eprint        = {2506.23478},
  primaryClass  = {cs.CV}
}

## License

MIT License
