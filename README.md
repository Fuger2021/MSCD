<div align=center><h1>
MSCD: Multi-Solution Collaborative Diffusion Model for Image Compressed Sensing
</h1>

Read in other languages: [[中文](README_zh-CN.md)]

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-MSCD-ff9000?logo=huggingface)](https://huggingface.co/Miyamura-Isumi/MSCD)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/Fuger2021/MSCD)
![GitHub](https://img.shields.io/github/license/Fuger2021/MSCD)
![GitHub last commit](https://img.shields.io/github/last-commit/Fuger2021/MSCD)
</div>

## Abstract

A multi-solution collaborative diffusion model that reframes CS reconstruction as a conditional generation task and concurrently constructs a set of candidate solutions to enrich diversity, achieving superior reconstruction quality compared to existing approaches.

## Overview

![illustration](assets/figs/unet.png)

## Setup

### Requirements

- Python ≥ 3.11
- CUDA ≥ 12.1

### Install

```shell
pip install -r requirements.txt
```

### Data & Weights

Download from [HuggingFace](https://huggingface.co/Miyamura-Isumi/MSCD). Expected directory structure:

```
.
├── data/
├── weight/
├── MSCD/
└── README.md
```

## Run MSCD

### Train

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_port=23333 train.py --cs_ratio=0.1
```

### Test

```shell
CUDA_VISIBLE_DEVICES=0 python -u test.py --cs_ratio=0.1 --data_dir="data" --testset_name="Set11"
```
