"""
可视化 zarr 触觉数据集
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diffusion_policy.common.replay_buffer import ReplayBuffer


def visualize_zarr_data(zarr_path="data/tactile_data.zarr"):
    """可视化 zarr 数据集的内容"""

    print(f"加载数据: {zarr_path}")
    replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")

    # 基本信息
    episode_ends = replay_buffer.root["meta"]["episode_ends"][:]
    n_episodes = len(episode_ends)
    n_frames = episode_ends[-1]

    print(f"\n{'='*60}")
    print(f"数据集基本信息")
    print(f"{'='*60}")
    print(f"总 Episode 数: {n_episodes}")
    print(f"总帧数: {n_frames}")

    # Episode 长度统计
    episode_lengths = np.diff(np.concatenate([[0], episode_ends]))
    print(f"\nEpisode 长度统计:")
    print(f"  最小: {episode_lengths.min()}")
    print(f"  最大: {episode_lengths.max()}")
    print(f"  平均: {episode_lengths.mean():.2f}")
    print(f"  中位数: {np.median(episode_lengths):.0f}")
    print(f"  标准差: {episode_lengths.std():.2f}")

    # 打印并保存所有 episode_ends 的具体数值
    print("\n所有 episode_ends 值:")
    print(episode_ends)
    csv_path = Path(zarr_path).parent / "episode_ends.csv"
    np.savetxt(csv_path, episode_ends, fmt="%d", delimiter=",")
    print(f"已保存 episode_ends 到: {csv_path}")

    # 数据形状
    print(f"\n{'='*60}")
    print(f"数据形状")
    print(f"{'='*60}")
    for key in replay_buffer.data.keys():
        shape = replay_buffer.data[key].shape
        dtype = replay_buffer.data[key].dtype
        print(f"  {key:12s}: {str(shape):20s} {dtype}")

    # 数据统计
    print(f"\n{'='*60}")
    print(f"数据统计")
    print(f"{'='*60}")

    forces_l = replay_buffer.data["forces_l"]
    forces_r = replay_buffer.data["forces_r"]
    action = replay_buffer.data["action"]

    print(f"forces_l:")
    print(f"  范围: [{forces_l[:100].min():.4f}, {forces_l[:100].max():.4f}]")
    print(f"  均值: {forces_l[:100].mean():.4f}")
    print(f"  标准差: {forces_l[:100].std():.4f}")

    print(f"forces_r:")
    print(f"  范围: [{forces_r[:100].min():.4f}, {forces_r[:100].max():.4f}]")
    print(f"  均值: {forces_r[:100].mean():.4f}")
    print(f"  标准差: {forces_r[:100].std():.4f}")

    print(f"action (end_position):")
    print(f"  X 范围: [{action[:, 0].min():.4f}, {action[:, 0].max():.4f}]")
    print(f"  Y 范围: [{action[:, 1].min():.4f}, {action[:, 1].max():.4f}]")
    print(f"  Z 范围: [{action[:, 2].min():.4f}, {action[:, 2].max():.4f}]")

    # 可视化
    print(f"\n{'='*60}")
    print(f"生成可视化...")
    print(f"{'='*60}")

    # 创建图形
    fig = plt.figure(figsize=(20, 12))

    # 1. Episode 长度分布
    ax1 = plt.subplot(3, 4, 1)
    ax1.hist(episode_lengths, bins=50, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Episode Length")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Episode Length Distribution")
    ax1.grid(True, alpha=0.3)

    # 2. Episode ends 累积
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(episode_ends, "b-", linewidth=1, label="cumulative")
    # scatter 每个 episode 的具体 end 值
    ax2.scatter(np.arange(n_episodes), episode_ends, c="red", s=6, label="episode_ends")
    ax2.set_xlabel("Episode Index")
    ax2.set_ylabel("Cumulative Frames")
    ax2.set_title("Episode Ends (Cumulative & Points)")
    ax2.legend(fontsize="small")
    ax2.grid(True, alpha=0.3)

    # 3. 选择第一个 episode 进行详细可视化
    first_ep_start = 0
    first_ep_end = episode_ends[0]
    first_ep_len = first_ep_end - first_ep_start

    print(f"\n可视化第一个 Episode:")
    print(f"  起始帧: {first_ep_start}")
    print(f"  结束帧: {first_ep_end}")
    print(f"  长度: {first_ep_len}")

    # 提取第一个 episode 的数据
    ep_forces_l = forces_l[first_ep_start:first_ep_end]
    ep_forces_r = forces_r[first_ep_start:first_ep_end]
    ep_action = action[first_ep_start:first_ep_end]

    # 4. 左手触觉力的第一个通道热力图（第一帧）
    ax3 = plt.subplot(3, 4, 3)
    im3 = ax3.imshow(ep_forces_l[0, 0], cmap="viridis", aspect="auto")
    ax3.set_title("Forces L - Channel 0 (Frame 0)")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    plt.colorbar(im3, ax=ax3)

    # 5. 右手触觉力的第一个通道热力图（第一帧）
    ax4 = plt.subplot(3, 4, 4)
    im4 = ax4.imshow(ep_forces_r[0, 0], cmap="viridis", aspect="auto")
    ax4.set_title("Forces R - Channel 0 (Frame 0)")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    plt.colorbar(im4, ax=ax4)

    # 6. 触觉力阵列随时间变化（左手，所有通道的均值）
    ax5 = plt.subplot(3, 4, 5)
    forces_l_mean = ep_forces_l.mean(axis=(2, 3))  # (T, 3)
    for i in range(3):
        ax5.plot(forces_l_mean[:, i], label=f"Channel {i}")
    ax5.set_xlabel("Frame")
    ax5.set_ylabel("Mean Force")
    ax5.set_title("Forces L - Mean over Spatial Dims")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 7. 触觉力阵列随时间变化（右手，所有通道的均值）
    ax6 = plt.subplot(3, 4, 6)
    forces_r_mean = ep_forces_r.mean(axis=(2, 3))  # (T, 3)
    for i in range(3):
        ax6.plot(forces_r_mean[:, i], label=f"Channel {i}")
    ax6.set_xlabel("Frame")
    ax6.set_ylabel("Mean Force")
    ax6.set_title("Forces R - Mean over Spatial Dims")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 8. Action 轨迹 (X, Y, Z 随时间)
    ax7 = plt.subplot(3, 4, 7)
    ax7.plot(ep_action[:, 0], label="X", alpha=0.7)
    ax7.plot(ep_action[:, 1], label="Y", alpha=0.7)
    ax7.plot(ep_action[:, 2], label="Z", alpha=0.7)
    ax7.set_xlabel("Frame")
    ax7.set_ylabel("Position (m)")
    ax7.set_title("Action Trajectory")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 9. Action 在 XY 平面的轨迹
    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(ep_action[:, 0], ep_action[:, 1], "b-", linewidth=1, alpha=0.7)
    ax8.scatter(
        ep_action[0, 0],
        ep_action[0, 1],
        c="green",
        s=100,
        marker="o",
        label="Start",
        zorder=5,
    )
    ax8.scatter(
        ep_action[-1, 0],
        ep_action[-1, 1],
        c="red",
        s=100,
        marker="x",
        label="End",
        zorder=5,
    )
    ax8.set_xlabel("X (m)")
    ax8.set_ylabel("Y (m)")
    ax8.set_title("Action Trajectory (XY Plane)")
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.axis("equal")

    # 10. 左手触觉力的所有 3 个通道（中间某一帧）
    mid_frame = first_ep_len // 2
    for i in range(3):
        ax = plt.subplot(3, 4, 9 + i)
        im = ax.imshow(ep_forces_l[mid_frame, i], cmap="viridis", aspect="auto")
        ax.set_title(f"Forces L Ch{i} (Frame {mid_frame})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # 11. Action 速度（位置变化）
    ax12 = plt.subplot(3, 4, 12)
    action_velocity = np.diff(ep_action, axis=0)
    action_speed = np.linalg.norm(action_velocity, axis=1)
    ax12.plot(action_speed, "b-", linewidth=1)
    ax12.set_xlabel("Frame")
    ax12.set_ylabel("Speed (m/frame)")
    ax12.set_title("Action Speed")
    ax12.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    output_path = Path(zarr_path).parent / "zarr_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n可视化已保存到: {output_path}")

    plt.show()

    return replay_buffer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="可视化 zarr 触觉数据集")
    parser.add_argument(
        "--zarr", type=str, default="data/tactile_data.zarr", help="zarr 文件路径"
    )

    args = parser.parse_args()

    visualize_zarr_data(args.zarr)
