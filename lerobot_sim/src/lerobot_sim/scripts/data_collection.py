#!/usr/bin/env python3
"""
使用键盘采集数据 (Keyboard Teleoperation Data Collection)
"""

import os
import shutil
import numpy as np
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot_sim.environments import PickPlaceEnv


def create_environment(xml_path: str, seed: int, state_type: str = "joint_angle"):
    """创建 PickPlace 环境"""
    return PickPlaceEnv(xml_path, seed=seed, state_type=state_type)


def init_dataset(root: str, repo_id: str, fps: int) -> LeRobotDataset:
    """初始化或加载数据集"""
    if os.path.exists(root):
        print(f"Directory {root} already exists.")
        if input("Do you want to delete it? (y/n) ").lower() == "y":
            shutil.rmtree(root)
            create_new = True
        else:
            create_new = False
    else:
        create_new = True

    if create_new:
        return LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot_type="frank",
            features={
                "observation.image": { # 主相机图像
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.wrist_image": { # 腕部相机图像
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.state": { # 6个关节角
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["state"], 
                },
                "action": { # 6 个关节角度和 1 个夹爪状态
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action"],
                },
                "obj_init": { # 物体初始位置
                    "dtype": "float32",
                    "shape": (9,),
                    "names": ["obj_init"],
                },
            },
            image_writer_threads=10, # 图像写入线程数
            image_writer_processes=5, # 图像写入进程数
        )
    else:
        print("Load from previous dataset")
        return LeRobotDataset(repo_id, root=root) 


def preprocess_image(image_array: np.ndarray) -> np.ndarray:
    """将图像缩放为 256x256 并返回 numpy 数组"""
    return np.array(Image.fromarray(image_array).resize((256, 256))) # numpy 数组 -> 转换为 PIL 图像 —> 缩放为 256x256 —> 转换为 numpy 数组


def collect_data(env: PickPlaceEnv, dataset: LeRobotDataset, episodes: int, fps: int):
    """主采集循环"""
    episode_id = 0
    record_flag = False
    action = np.zeros(7, dtype=np.float32)

    while env.env.is_viewer_alive() and episode_id < episodes:
        env.step_env()
        
        if not env.env.loop_every(fps):
            continue

        # 检查是否完成一个回合
        if env.check_success():
            dataset.save_episode()
            env.reset()
            episode_id += 1
            record_flag = False
            continue

        # 键盘遥操作
        action, reset = env.teleop_robot()
        if not record_flag and np.any(action != 0):
            record_flag = True
            print("Start recording")
            
        # 重置环境
        if reset:
            env.reset()
            dataset.clear_episode_buffer()
            record_flag = False
            continue

        # 读取相机图像
        agent_img, wrist_img = env.grab_image()
        agent_img = preprocess_image(agent_img)
        wrist_img = preprocess_image(wrist_img)

        # 执行动作
        joint_q = env.step(action)
        action = env.q[:7].astype(np.float32)

        # 记录数据
        if record_flag:
            dataset.add_frame(
                {
                    "observation.image": agent_img,
                    "observation.wrist_image": wrist_img,
                    "observation.state": joint_q[:6],
                    "action": action,
                    "obj_init": env.obj_init_pose,
                },
                task=env.instruction,
            )

        # 渲染环境
        env.render(teleop=True, idx=episode_id)


def main():
    seed = 0  # 固定随机种子，设为 None 则随机化
    fps = 20 # 帧率
    episodes = 10 # 回合数量

    repo_id = "frank_pickplace_01"
    root = "data/datasets/frank_pickplace_01"
    xml_path = "assets/mujoco/pickplace_scene.xml"

    
    env = create_environment(xml_path, seed)
    dataset = init_dataset(root, repo_id, fps)

    try:
        collect_data(env, dataset, episodes, fps)
    finally:
        env.env.close_viewer()
        print("Viewer closed.")


# ==== 程序入口 ====
if __name__ == "__main__":
    main()
