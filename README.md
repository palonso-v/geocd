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

```bash
conda create -n geocd python=3.10
conda activate geocd
```

## Data Preparation

Make sure your datasets are prepared in the format expected by load_data(...).

Currently supported datasets include:

modelnet40

If you use custom dataset paths or preprocessing, update the data-loading utilities accordingly.

## Training

1. Pretrain with Chamfer Distance

```bash
python train.py \
    --backbone AE \
    --dataset modelnet40 \
    --batch_size 12 \
    --nepochs 100
```

2. Finetune with GeoCD

```bash
python finetune.py \
    --backbone AE \
    --dataset modelnet40 \
    --batch_size 1 \
    --nepochs 1 \
    --checkpoint_path ./outputs/geocd_AE_modelnet40_cd_best.pth \
    --k_value 5 \
    --nhops 2
```

3. Evaluation

```bash
python evaluate.py \
    --backbone AE \
    --dataset modelnet40 \
    --batch_size 1 \
    --checkpoint_path ./outputs/geocd_AE_modelnet40_geocd_best.pth \
    --split val \
    --metrics cd geocd \
    --k_value 5 \
    --nhops 2
```

## Checkpoints

During training, checkpoints are saved to --save_dir.

Typical filenames are:

geocd_AE_modelnet40_cd_best.pth

geocd_AE_modelnet40_cd_last.pth

geocd_AE_modelnet40_geocd_best.pth

geocd_AE_modelnet40_geocd_last.pth

## Citation

If you find this repository useful, please cite:

```bibtex
@ARTICLE{2025arXiv250623478A,
  author        = {{Alonso}, Pedro and {Li}, Tianrui and {Li}, Chongshou},
  title         = "{GeoCD: A Differential Local Approximation for Geodesic Chamfer Distance}",
  journal       = {arXiv e-prints},
  year          = 2025,
  eprint        = {2506.23478},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2506.23478}
}
```

## License

MIT License
