# lerobot sim

这是一个使用 mujoco 实现的仿真环境，数据集和模型策略均集成了 lerobot 的标准。

## 0. 配置环境

```
# 下载依赖
uv sync

# 激活
source .venv/bin/activate
```

## 1. 数据采集

1. 在包含语言指令的场景中采集演示数据。右上/右下叠加图像来自数据集；侧视图显示在左上。

2. 数据集特征
```
fps = 20,
features={
    "observation.image": {"dtype": "image", "shape": (256, 256, 3)},
    "observation.wrist_image": {"dtype": "image", "shape": (256, 256, 3)},
    "observation.state": {"dtype": "float32", "shape": (6,)},
    "action": {"dtype": "float32", "shape": (7,)},
    "obj_init": {"dtype": "float32", "shape": (9,)},
}
```

3. 键盘控制说明

XY 平面: W 后退 / S 前进 / A 左移 / D 右移
Z 轴:    R 上升 / F 下降
旋转:    Q 左倾 / E 右倾 / 方向键控制俯仰与偏航
空格:    切换夹爪
Z:      重置（丢弃当前回合）

## 2. 可视化

## 3. 训练

### 3.1. install 
```
pip install transformers==4.50.3
pip install num2words
pip install accelerate
pip install 'safetensors>=0.4.3'
```

### 3.2. download dataset
1. git 
```
git clone https://huggingface.co/datasets/Liberow/omy_pickplace
``` 

2. cli
```
# install package
pip install huggingface_hub

# login
huggingface-cli login 

# download
huggingface-cli download Liberow/omy_pickplace \
    --repo-type dataset \
    --local-dir ./data/datasets
```

### 3.3. train 
```
python scripts/training.py --config_path configs/train_smolvla.yaml
```

## 4. 部署

### 4.1. download models
```
# download
huggingface-cli download Liberow/omy_pickplace \
    --repo-type model \
    --local-dir ./models
```

## 常用命令

| 任务           | 命令                          |
| ------------ | --------------------------- |
| 创建项目         | `uv init myapp`             |
| 创建虚拟环境       | `uv venv`                   |
| 激活环境         | `source .venv/bin/activate` |
| 安装依赖         | `uv add <package>`          |
| 安装开发依赖       | `uv add --dev <package>`    |
| 更新依赖         | `uv lock --upgrade`         |
| 运行脚本         | `uv run python main.py`     |
| 安装项目依赖（恢复环境） | `uv sync`                   |


## 常见问题

1. lerobot 仓库使用了 Git LFS 存储大文件，但远程缺少某些 LFS 对象

* 错误输出
```
× Failed to download and build `lerobot @ git+https://github.com/huggingface/lerobot.git@10b7b3532543b4adfb65760f02a49b4c537afde7`
├─▶ Git operation failed
╰─▶ process didn't exit successfully: `/usr/bin/git reset --hard 10b7b3532543b4adfb65760f02a49b4c537afde7` (exit status: 128)
    --- stderr
    Downloading tests/artifacts/cameras/image_128x128.png (38 KB)
    Error downloading object: tests/artifacts/cameras/image_128x128.png (9dc9df0): Smudge error: Error downloading tests/artifacts/cameras/image_128x128.png (9dc9df05797dc0e7b92edc845caab2e4c37c3cfcabb4ee6339c67212b5baba3b): error transferring "9dc9df05797dc0e7b92edc845caab2e4c37c3cfcabb4ee6339c67212b5baba3b": [0] remote missing object 9dc9df05797dc0e7b92edc845caab2e4c37c3cfcabb4ee6339c67212b5baba3b

    Errors logged to /root/.cache/uv/git-v0/checkouts/b2400a7a62d6a7cf/10b7b35/.git/lfs/logs/20251110T095024.56350926.log
    Use `git lfs logs last` to view the log.
    error: external filter 'git-lfs filter-process' failed
    fatal: tests/artifacts/cameras/image_128x128.png: smudge filter lfs failed
```

* 解决方法(跳过LFS文件下载)

```
# 临时跳过
GIT_LFS_SKIP_SMUDGE=1 uv run data_collection

# 或永久配置（全局）
git config --global filter.lfs.smudge "git-lfs smudge --skip"
git config --global filter.lfs.required false

# 或永久配置（仅当前仓库）
git config filter.lfs.smudge "git-lfs smudge --skip"
git config filter.lfs.required false
```

2. 鼠标问题

* 报错
```
root@c81ab3b21da2:/workspace/liber/embodied_ai/Embodied_AI_Learning/lerobot_sim# sudo apt-get install python3-tk python3-dev
```

* 解决办法
```
sudo apt-get install python3-tk python3-dev
```

## 参考项目

1. [lerobot](https://github.com/huggingface/lerobot)

2. [mujoco](https://github.com/google-deepmind/mujoco)

3. [lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial)