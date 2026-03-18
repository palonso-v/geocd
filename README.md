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

## Repository Structure
.
├── train.py
├── finetune.py
├── evaluate.py
├── engine.py
├── models.py
├── losses.py
├── utils.py
├── repo_components.py
└── README.md

## Method

GeoCD replaces the standard Euclidean nearest-neighbor matching of Chamfer Distance with a differentiable approximation of geodesic-aware matching.

At a high level:

The predicted and target point clouds are merged.

A k-nearest-neighbor graph is built over the merged point set.

Multi-hop propagation is used to approximate geodesic structure.

Cross-cloud distances are matched using a differentiable softmin.

The final loss averages both directions, similarly to Chamfer Distance.

This allows the reconstruction objective to account for local surface structure beyond purely Euclidean nearest-neighbor proximity.

## Installation

Clone the repository:

git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

Create and activate a conda environment:

conda create -n geocd python=3.10
conda activate geocd

Install PyTorch following the official instructions for your CUDA version, then install the remaining dependencies:

pip install numpy matplotlib h5py

If your project depends on custom CUDA Chamfer implementations, PointTransformerV3 / Pointcept code, or other local modules, install or compile them as required by your setup.

## Data Preparation

Make sure your datasets are prepared in the format expected by load_data(...) in your codebase.

Currently supported datasets include:

modelnet40

If you use custom dataset paths or preprocessing, update the data-loading utilities accordingly.

## Training

1. Pretrain with Chamfer Distance

Train a reconstruction model from scratch using standard Chamfer Distance:

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

Finetune a pretrained checkpoint using GeoCD:

python finetune.py \
    --backbone AE \
    --dataset modelnet40 \
    --batch_size 1 \
    --nepochs 1 \
    --checkpoint_path ./outputs/geocd_AE_modelnet40_cd_best.pth \
    --k_value 5 \
    --nhops 2

Example with PTv3:

python finetune.py \
    --backbone PTv3 \
    --dataset modelnet40 \
    --batch_size 1 \
    --nepochs 1 \
    --checkpoint_path ./outputs/geocd_PTv3_modelnet40_cd_best.pth \
    --k_value 5 \
    --nhops 2

3. Evaluation

Evaluate a checkpoint on a dataset split using one or more metrics:

python evaluate.py \
    --backbone AE \
    --dataset modelnet40 \
    --batch_size 1 \
    --checkpoint_path ./outputs/geocd_AE_modelnet40_geocd_best.pth \
    --split val \
    --metrics cd geocd \
    --k_value 5 \
    --nhops 2

Available evaluation metrics:

cd

geocd

Available splits:

train

val

## Main Arguments

Shared arguments

--backbone: model backbone (AE, PTv3)

--dataset: dataset name

--batch_size: batch size

--latent_dim: latent dimension for applicable models

--device: device to use, typically cuda

--seed: random seed

Training arguments

--nepochs: number of epochs

--lr: learning rate; if omitted, a backbone-specific default is used

--save_dir: directory for checkpoints

--load_checkpoint: resume CD training from a checkpoint

--checkpoint_path: path to checkpoint

GeoCD finetuning / evaluation arguments

--k_value: number of nearest neighbors in the graph

--nhops: number of graph propagation hops


## Checkpoints

During training, checkpoints are saved to --save_dir.

Typical filenames are:

geocd_AE_modelnet40_cd_best.pth

geocd_AE_modelnet40_cd_last.pth

geocd_AE_modelnet40_geocd_best.pth

geocd_AE_modelnet40_geocd_last.pth

The same naming convention applies to other backbones and datasets.

## Example Workflow

A typical workflow is:

Pretrain with CD

Finetune the pretrained model with GeoCD

Evaluate both checkpoints

Example:

python train.py \
    --backbone AE \
    --dataset modelnet40 \
    --batch_size 12 \
    --nepochs 100

python finetune.py \
    --backbone AE \
    --dataset modelnet40 \
    --batch_size 1 \
    --nepochs 1 \
    --checkpoint_path ./outputs/geocd_AE_modelnet40_cd_best.pth \
    --k_value 5 \
    --nhops 2

python evaluate.py \
    --backbone AE \
    --dataset modelnet40 \
    --batch_size 1 \
    --checkpoint_path ./outputs/geocd_AE_modelnet40_geocd_best.pth \
    --split val \
    --metrics cd geocd \
    --k_value 5 \
    --nhops 2

## Notes

GeoCD finetuning is intended to start from a pretrained CD checkpoint.

For very small batch sizes, especially batch_size=1, BatchNorm layers may be frozen during finetuning.

The GeoCD implementation uses a differentiable multi-hop kNN-based approximation of geodesic structure.

The exact behavior of the method depends on graph hyperparameters such as k_value and nhops.

## Citation

If you find this repository useful, please cite:

@ARTICLE{2025arXiv250623478A,
       author = {{Alonso}, Pedro and {Li}, Tianrui and {Li}, Chongshou},
        title = "{GeoCD: A Differential Local Approximation for Geodesic Chamfer Distance}",
      journal = {arXiv e-prints},
     keywords = {Computer Vision and Pattern Recognition},
         year = 2025,
        month = jun,
          eid = {arXiv:2506.23478},
        pages = {arXiv:2506.23478},
          doi = {10.48550/arXiv.2506.23478},
archivePrefix = {arXiv},
       eprint = {2506.23478},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250623478A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

## License

MIT License