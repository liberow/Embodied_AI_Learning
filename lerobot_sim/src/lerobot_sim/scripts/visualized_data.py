#!/usr/bin/env python3
"""
可视化语言数据集 (Visualize Language-Conditioned Dataset)

功能：
- 在 MuJoCo 仿真环境中回放带语言指令的数据集
- 主窗口实时回放动作
- 右上/右下叠加数据集中对应的图像帧
"""

import torch
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from lerobot_sim.environments import PickPlaceEnv


# ==== 采样器 ====
class EpisodeSampler(torch.utils.data.Sampler):
    """单回合采样器（用于顺序回放一个 episode）"""

    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self):
        return len(self.frame_ids)


def load_dataset(root: str, repo_id: str) -> LeRobotDataset:
    """加载数据集"""
    print(f"Loading dataset from {root} ...")
    return LeRobotDataset(repo_id, root=root)


def create_dataloader(dataset: LeRobotDataset, episode_index: int):
    """为单个回合创建 DataLoader"""
    sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=1,
        sampler=sampler,
    )
    return dataloader, sampler


def setup_environment(xml_path: str) -> PickPlaceEnv:
    """创建仿真环境"""
    print(f"Initializing environment from {xml_path}")
    return PickPlaceEnv(xml_path, action_type="joint_angle")


def visualize_episode(env: PickPlaceEnv, dataloader, sampler: EpisodeSampler, fps: int):
    """主回放循环"""
    step = 0
    iter_dataloader = iter(dataloader)
    env.reset()

    while env.env.is_viewer_alive():
        env.step_env()

        if not env.env.loop_every(fps):
            continue

        try:
            data = next(iter_dataloader)
        except StopIteration:
            # 数据用完，重新开始
            iter_dataloader = iter(dataloader)
            data = next(iter_dataloader)
            env.reset()
            step = 0

        # 初始化任务
        if step == 0:
            instruction = data["task"][0]
            env.set_instruction(instruction)
            env.set_obj_pose(
                data["obj_init"][0, :3],
                data["obj_init"][0, 3:6],
                data["obj_init"][0, 6:9],
            )

        # 执行动作
        action = data["action"].numpy()[0]
        env.step(action)

        # 从数据集中提取图像 (C,H,W) -> (H,W,C)
        agent_img = (data["observation.image"][0].numpy() * 255).astype(np.uint8)
        wrist_img = (data["observation.wrist_image"][0].numpy() * 255).astype(np.uint8)
        env.rgb_agent = np.transpose(agent_img, (1, 2, 0))
        env.rgb_ego = np.transpose(wrist_img, (1, 2, 0))
        env.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)

        # 渲染当前帧
        env.render()
        step += 1

        # 若播放完一整个 episode，循环播放
        if step == len(sampler):
            iter_dataloader = iter(dataloader)
            env.reset()
            step = 0


def main():
    fps = 20 # 帧率
    episode_index = 0
    root = "./data/datasets" # 数据集保存根目录
    repo_id = "frank_pickplace"
    xml_path = "./assets/mujoco/pickplace_scene.xml"

    dataset = load_dataset(root, repo_id)
    dataloader, sampler = create_dataloader(dataset, episode_index)
    env = setup_environment(xml_path)

    try:
        visualize_episode(env, dataloader, sampler, fps)
    finally:
        env.env.close_viewer()
        print("Viewer closed.")


if __name__ == "__main__":
    main()
