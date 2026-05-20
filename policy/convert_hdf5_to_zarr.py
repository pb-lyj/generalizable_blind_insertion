"""
将 HDF5 触觉数据转换为 Diffusion Policy 标准的 zarr 格式
"""

import os
import h5py
import zarr
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer


def load_hdf5_episode(hdf5_path):
    """从单个 HDF5 文件加载一个 episode 的数据

    注意：当前时刻 t 的 action 是下一时刻 t+1 的 end_position
    最后一帧的 action 使用当前帧的 end_position（因为没有下一帧）
    """
    with h5py.File(hdf5_path, "r") as f:
        obs = f["observation"]

        # 提取观测数据
        forces_l = obs["forces_l"][:]  # (T, 3, 20, 20)
        forces_r = obs["forces_r"][:]  # (T, 3, 20, 20)
        end_position = obs["end_position"][:]  # (T, 3)

        # 构建动作：当前时刻的 action 是下一时刻的 end_position
        T = forces_l.shape[0]
        action = np.zeros((T, 3), dtype=np.float32)

        # 对于 t=0 到 t=T-2，action[t] = end_position[t+1]
        action[:-1] = end_position[1:]

        # 对于最后一帧（t=T-1），action[T-1] = end_position[T-1]
        # （保持在当前位置，因为没有下一帧）
        action[-1] = end_position[-1]

    return {"forces_l": forces_l, "forces_r": forces_r, "action": action}


def convert_hdf5_to_zarr(
    hdf5_root_dir="hdf5_10hz",
    output_zarr_path="data/tactile_data.zarr",
    categories=None,
):
    """
    将所有 HDF5 数据转换为单个 zarr ReplayBuffer

    Args:
        hdf5_root_dir: HDF5 数据根目录
        output_zarr_path: 输出 zarr 文件路径
        categories: 要处理的类别列表，None 表示处理所有类别
    """

    # 读取 dataset_info.json
    info_path = Path(hdf5_root_dir) / "dataset_info.json"
    with open(info_path, "r") as f:
        dataset_info = json.load(f)

    if categories is None:
        categories = list(dataset_info["categories"].keys())

    print(f"处理类别: {categories}")

    # 收集所有 episode 路径
    all_episode_paths = []
    for category in categories:
        category_dir = Path(hdf5_root_dir) / category
        if not category_dir.exists():
            print(f"警告: 类别目录不存在: {category_dir}")
            continue

        episode_files = sorted(category_dir.glob("*.hdf5"))
        all_episode_paths.extend(episode_files)

    print(f"总共找到 {len(all_episode_paths)} 个 episode 文件")

    # 第一步: 扫描所有数据以确定总长度
    print("\n第1步: 扫描数据以确定形状...")
    total_frames = 0
    episode_lengths = []

    for ep_path in tqdm(all_episode_paths, desc="扫描"):
        with h5py.File(ep_path, "r") as f:
            ep_len = f["observation"]["forces_l"].shape[0]
            episode_lengths.append(ep_len)
            total_frames += ep_len

    print(f"总帧数: {total_frames}")
    print(f"Episode 数量: {len(episode_lengths)}")

    # 第二步: 创建 zarr ReplayBuffer
    print("\n第2步: 创建 zarr 结构...")
    output_path = Path(output_zarr_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 删除已存在的 zarr
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)

    # 创建 zarr 结构（手动创建以避免 numpy 版本兼容性问题）
    root = zarr.open(str(output_path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")

    # 创建 episode_ends 数组（先创建一个临时值以满足 ReplayBuffer 的检查）
    meta.create_dataset(
        "episode_ends",
        shape=(1,),
        chunks=(1024,),
        dtype=np.int64,
        data=np.array([total_frames], dtype=np.int64),
    )

    # 创建数据集
    # forces_l: (T, 3, 20, 20) - 左手触觉力阵
    data.create_dataset(
        "forces_l",
        shape=(total_frames, 3, 20, 20),
        chunks=(1, 3, 20, 20),
        dtype=np.float32,
    )

    # forces_r: (T, 3, 20, 20) - 右手触觉力阵
    data.create_dataset(
        "forces_r",
        shape=(total_frames, 3, 20, 20),
        chunks=(1, 3, 20, 20),
        dtype=np.float32,
    )

    # action: (T, 3) - 下一时刻的位置
    data.create_dataset(
        "action", shape=(total_frames, 3), chunks=(1, 3), dtype=np.float32
    )

    # 创建 ReplayBuffer 对象
    replay_buffer = ReplayBuffer(root=root)

    # 第三步: 填充数据
    print("\n第3步: 填充数据...")
    current_idx = 0
    episode_ends = []

    for ep_idx, ep_path in enumerate(tqdm(all_episode_paths, desc="转换")):
        # 加载数据
        ep_data = load_hdf5_episode(ep_path)
        ep_len = ep_data["forces_l"].shape[0]

        # 写入数据
        replay_buffer.data["forces_l"][current_idx : current_idx + ep_len] = ep_data[
            "forces_l"
        ].astype(np.float32)
        replay_buffer.data["forces_r"][current_idx : current_idx + ep_len] = ep_data[
            "forces_r"
        ].astype(np.float32)
        replay_buffer.data["action"][current_idx : current_idx + ep_len] = ep_data[
            "action"
        ].astype(np.float32)

        current_idx += ep_len
        episode_ends.append(current_idx)

    # 更新 episode_ends
    episode_ends_array = np.array(episode_ends, dtype=np.int64)
    replay_buffer.root["meta"]["episode_ends"].resize(len(episode_ends))
    replay_buffer.root["meta"]["episode_ends"][:] = episode_ends_array

    print("\n转换完成!")
    print(f"输出路径: {output_path}")
    print(f"总帧数: {total_frames}")
    print(f"Episode 数量: {len(episode_ends)}")
    print(f"\n数据集结构:")
    print(f"  forces_l: {replay_buffer.data['forces_l'].shape}")
    print(f"  forces_r: {replay_buffer.data['forces_r'].shape}")
    print(f"  action: {replay_buffer.data['action'].shape}")
    print(f"  episode_ends: {replay_buffer.root['meta']['episode_ends'].shape}")

    return replay_buffer


def verify_zarr_data(zarr_path="data/tactile_data.zarr"):
    """验证转换后的 zarr 数据"""
    print(f"\n验证 zarr 数据: {zarr_path}")
    replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")

    n_episodes = len(replay_buffer.root["meta"]["episode_ends"])
    n_frames = replay_buffer.root["meta"]["episode_ends"][-1]

    print(f"Episodes: {n_episodes}")
    print(f"总帧数: {n_frames}")

    # 检查数据形状
    print("\n数据形状:")
    for key in replay_buffer.data.keys():
        print(f"  {key}: {replay_buffer.data[key].shape}")

    # 采样检查
    print("\n采样检查 (第一个 episode 的第一帧):")
    print(
        f"  forces_l 范围: [{replay_buffer.data['forces_l'][0].min():.4f}, "
        f"{replay_buffer.data['forces_l'][0].max():.4f}]"
    )
    print(
        f"  forces_r 范围: [{replay_buffer.data['forces_r'][0].min():.4f}, "
        f"{replay_buffer.data['forces_r'][0].max():.4f}]"
    )
    print(f"  action: {replay_buffer.data['action'][0]}")

    # Episode 长度统计
    episode_ends = replay_buffer.root["meta"]["episode_ends"][:]
    episode_lengths = np.diff(np.concatenate([[0], episode_ends]))
    print(f"\nEpisode 长度统计:")
    print(f"  最小: {episode_lengths.min()}")
    print(f"  最大: {episode_lengths.max()}")
    print(f"  平均: {episode_lengths.mean():.2f}")
    print(f"  中位数: {np.median(episode_lengths):.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="将 HDF5 数据转换为 zarr 格式")
    parser.add_argument(
        "--input", type=str, default="hdf5_10hz", help="HDF5 数据根目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tactile_data.zarr",
        help="输出 zarr 文件路径",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="要处理的类别，不指定则处理所有类别",
    )
    parser.add_argument("--verify", action="store_true", help="转换后验证数据")

    args = parser.parse_args()

    # 转换数据
    replay_buffer = convert_hdf5_to_zarr(
        hdf5_root_dir=args.input,
        output_zarr_path=args.output,
        categories=args.categories,
    )

    # 验证数据
    if args.verify:
        verify_zarr_data(args.output)
