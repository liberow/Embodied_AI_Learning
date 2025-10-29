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

2. 参考 [lerobot](https://github.com/huggingface/lerobot) 安装


## parquet 文件

1. 安装

```
wget https://github.com/duckdb/duckdb/releases/download/v1.4.1/duckdb_cli-linux-amd64.zip -O duckdb.zip
unzip duckdb.zip
sudo mv duckdb /usr/local/bin/
```

2. 验证安装
```
duckdb --version
```

3. 看前 20 行
```
duckdb -
```

4. 看表结构/类型信息
```
duckdb -c "DESCRIBE SELECT * FROM '/zhouzhili/liber/lerobot/lerobot/svla_so101_pickplace/meta/tasks.parquet';"
```

5. 目录里多文件也行（通配）
```
duckdb -c "SELECT * FROM '/zhouzhili/liber/lerobot/lerobot/svla_so101_pickplace/data/chunk-000/*.parquet'
```

## Issue

1. 无法访问huggingface

```
export HF_ENDPOINT=https://hf-mirror.com
```
