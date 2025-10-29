# Robot Learning: A Tutorial

## Introduction

1. 论文: [Robot Learning: A Tutorial](https://arxiv.org/abs/2510.12403)

## Environment

1. 环境配置

```bash 
conda create -y -n robot_learning_tutorial python=3.10

conda activate robot_learning_tutorial

pip install lerobot
```

## Issue

1. 无法访问huggingface

```
export HF_ENDPOINT=https://hf-mirror.com
```

2. Lerobot 数据集版本和 lerobot 库的版本冲突

```bash 
pip install "lerobot<0.3.0"
```