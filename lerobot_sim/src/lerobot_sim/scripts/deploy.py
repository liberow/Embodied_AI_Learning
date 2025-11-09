#!/usr/bin/env python3
"""
部署已训练的 SmolVLA 策略

功能：
- 加载本地或 HF 数据集元信息
- 初始化 SmolVLA 策略
- 在 MuJoCo 环境中回放策略
"""
from typing import Any
import torch
from PIL import Image
import numpy as np
import os
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot_sim.environments.pickplace_env import PickPlaceEnv


def load_dataset_metadata(dataset_name: str, roots: list):
    """加载数据集元信息"""
    for root in roots:
        try:
            return LeRobotDatasetMetadata(dataset_name, root=root)
        except Exception:
            continue
    raise FileNotFoundError(f"Dataset {dataset_name} not found in any of {roots}")


def build_policy(metadata: LeRobotDatasetMetadata, ckpt_path: str, device: str):
    """初始化 SmolVLA 策略"""
    features = dataset_to_policy_features(metadata.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}
    cfg = SmolVLAConfig(input_features=input_features, output_features=output_features, chunk_size=5, n_action_steps=5)
    _ = resolve_delta_timestamps(cfg, metadata)
    policy = SmolVLAPolicy.from_pretrained(ckpt_path, config=cfg, dataset_stats=metadata.stats)
    policy.to(device)
    policy.eval()
    return policy


def get_image_transform():
    """返回图像预处理 transform"""
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def run_policy(env: PickPlaceEnv, policy: SmolVLAPolicy, device: str, img_transform: Any, fps: int, img_size: int):
    """在环境中回放策略"""
    step = 0
    env.reset(seed=0)
    policy.reset()

    while env.env.is_viewer_alive():
        env.step_env()
        if not env.env.loop_every(fps):
            continue

        # 成功判定
        if env.check_success():
            print('Success')
            policy.reset()
            env.reset()
            step = 0

        # 获取状态与图像
        state = env.get_joint_state()[:6]
        image, wrist_image = env.grab_image()
        image = img_transform(Image.fromarray(image).resize((img_size, img_size)))
        wrist_image = img_transform(Image.fromarray(wrist_image).resize((img_size, img_size)))

        # 构造输入
        data = {
            'observation.state': torch.tensor([state]).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
            'task': [env.instruction],
        }

        # 推理动作
        action = policy.select_action(data)[0, :7].cpu().detach().numpy()

        # 执行
        env.step(action)
        env.render()
        step += 1

        if env.check_success():
            print('Success')
            break


# ==== 主程序入口 ====
def main():
    # ==== 配置 ====
    device = "cuda"
    dataset_name = "omy_pickplace"
    local_roots = ['./data/datasets/pickplace_omy_01']
    ckpt_path = './data/models/smolvla_omy/checkpoints/020000/pretrained_model'
    xml_path = './assets/mujoco/pickplace_scene.xml'
    fps = 20
    img_size = 256
    metadata = load_dataset_metadata(dataset_name, local_roots)
    policy = build_policy(metadata, ckpt_path, device)
    env = PickPlaceEnv(xml_path, action_type='joint_angle')
    img_transform = get_image_transform()

    try:
        run_policy(env, policy, device, img_transform, fps, img_size)
    finally:
        env.env.close_viewer()
        print("Viewer closed.")


if __name__ == "__main__":
    main()
