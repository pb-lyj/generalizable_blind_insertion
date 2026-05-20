# -*- coding: utf-8 -*-
"""
遍历 ROOT_DIR 下每个子文件夹（每个文件夹一条示教轨迹）：
- 从四个TXT生成 state: float32 [T_total, 12]
- 从 _end_position.txt 生成 action: float32 [T_total, 3]（同级 data/action）
- 写入 meta/episode_ends: int64 [N_traj] —— 每条轨迹结束的全局索引（连续整数）
不写任何真实时间戳或 data/t_index。
"""

import os
import re
import json
import numpy as np
import zarr
from numcodecs import Blosc

# =========================
# 路径配置：修改为你的数据根目录
# =========================
ROOT_DIR = "data25.7_aligned/rect_med"  #
OUT_ZARR_DIR = "data/rect_med.zarr"

# 时间对齐的小数位（仅用于对齐，最终不写时间）
TIME_DECIMALS = 6

# 必须存在的文件名（state来源）
REQ_FILES = [
    "_resultant_force_l.txt",
    "_resultant_force_r.txt",
    "_resultant_moment_l.txt",
    "_resultant_moment_r.txt",
]
# 可选/建议存在的文件名（action来源）
ACTION_FILE = "_end_position.txt"  # 作为 action（默认3维）


# =========================
# 解析 & 对齐工具
# =========================
def _safe_float(line: str) -> float:
    try:
        return float(line)
    except ValueError:
        tokens = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
        return float(tokens[0]) if tokens else np.nan


def parse_txt_4perblock(path: str):
    """
    解析四行一组：time, v1, v2, v3；允许出现'===='分隔。
    返回 times[T], vals[T,3]（过滤NaN）。
    """
    times, vals = [], []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i, n = 0, len(lines)
    while i < n:
        if set(lines[i]) == {"="}:
            i += 1
            continue
        t = _safe_float(lines[i])
        i += 1
        vec = []
        for _ in range(3):
            while i < n and set(lines[i]) == {"="}:
                i += 1
            if i >= n:
                break
            vec.append(_safe_float(lines[i]))
            i += 1
        if len(vec) == 3:
            times.append(t)
            vals.append(vec)
    times = np.asarray(times, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    ok = np.isfinite(times) & np.all(np.isfinite(vals), axis=1)
    return times[ok], vals[ok]


def parse_action_file(path: str):
    """
    尝试解析 _end_position.txt：
    - 情况A（常见）：只有一个3D向量（可能一行三个数，或三行各一个数），返回 ('single', vec3)
    - 情况B：四行一组序列（time, x, y, z），返回 ('sequence', times[T], vals[T,3])
    - 其他：尽量从文本中抓出前三个数作为 vec3，返回 ('single', vec3)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f if ln.strip()]
    # 判断是否可能是四行一组：尝试抓第一块的前4行都为数值或分隔线
    # 更稳妥：尝试用四行组解析
    try:
        times, vals = parse_txt_4perblock(path)
        if times.size > 0 and vals.shape[1] == 3:
            return ("sequence", times, vals.astype(np.float32))
    except Exception:
        pass

    # 尝试“单目标”格式：抓取前三个数
    tokens = []
    for ln in raw:
        if set(ln) == {"="}:
            continue
        tokens += re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", ln)
        if len(tokens) >= 3:
            break
    if len(tokens) >= 3:
        vec = np.array(tokens[:3], dtype=np.float32)
        return ("single", vec)
    # 兜底：返回零向量
    return ("single", np.zeros(3, dtype=np.float32))


def round_times(t: np.ndarray, decimals: int = 6):
    return np.round(t.astype(np.float64), decimals)


def intersect_times(*times_list, decimals: int = 6) -> np.ndarray:
    sets = [set(round_times(t, decimals)) for t in times_list]
    common = sorted(list(set.intersection(*sets)))
    return np.asarray(common, dtype=np.float64)


def build_index(times: np.ndarray, decimals: int = 6):
    m = {}
    for i, v in enumerate(times):
        key = round(float(v), decimals)
        if key not in m:
            m[key] = i
    return m


def take_by_time(
    times: np.ndarray, vals: np.ndarray, target_times: np.ndarray, decimals: int = 6
):
    idx_map = build_index(times, decimals)
    idx = [idx_map[round(float(t), decimals)] for t in target_times]
    return vals[np.asarray(idx, dtype=np.int64)]


# =========================
# 主流程（单轨迹）
# =========================
def process_one_trajectory(traj_dir: str):
    """
    处理单条轨迹目录，返回:
      state_i: float32 [Ti, 12]
      action_i: float32 [Ti, 3]
      Ti: int
    若缺文件或数据为空，返回 None, None, 0
    """
    paths = {name: os.path.join(traj_dir, name) for name in REQ_FILES}
    if not all(os.path.isfile(p) for p in paths.values()):
        missing = [n for n, p in paths.items() if not os.path.isfile(p)]
        print(f"[WARN] skip '{traj_dir}': missing files {missing}")
        return None, None, 0

    # 解析四文件（用于 state）
    t_fl, v_fl = parse_txt_4perblock(paths["_resultant_force_l.txt"])
    t_fr, v_fr = parse_txt_4perblock(paths["_resultant_force_r.txt"])
    t_ml, v_ml = parse_txt_4perblock(paths["_resultant_moment_l.txt"])
    t_mr, v_mr = parse_txt_4perblock(paths["_resultant_moment_r.txt"])

    # 轨迹内部对齐（仅用于对齐，不写实际时间）
    common_t = intersect_times(t_fl, t_fr, t_ml, t_mr, decimals=TIME_DECIMALS)
    if common_t.size == 0:
        print(f"[WARN] skip '{traj_dir}': empty intersection after time alignment")
        return None, None, 0

    fl = take_by_time(t_fl, v_fl, common_t, decimals=TIME_DECIMALS)  # [Ti,3]
    fr = take_by_time(t_fr, v_fr, common_t, decimals=TIME_DECIMALS)  # [Ti,3]
    ml = take_by_time(t_ml, v_ml, common_t, decimals=TIME_DECIMALS)  # [Ti,3]
    mr = take_by_time(t_mr, v_mr, common_t, decimals=TIME_DECIMALS)  # [Ti,3]

    # state: [Ti,12]
    state_i = np.concatenate([fl, ml, fr, mr], axis=1).astype(np.float32)
    Ti = state_i.shape[0]

    # 解析 action 文件（若缺失则广播零向量）
    action_path = os.path.join(traj_dir, ACTION_FILE)
    if not os.path.isfile(action_path):
        print(f"[WARN] '{traj_dir}' missing {ACTION_FILE}, use zeros action")
        action_i = np.zeros((Ti, 3), dtype=np.float32)
        return state_i, action_i, Ti

    kind, *payload = parse_action_file(action_path)
    if kind == "single":
        vec3 = payload[0]  # [3,]
        action_i = np.broadcast_to(vec3.reshape(1, 3), (Ti, 3)).astype(np.float32)
    else:
        a_times, a_vals = payload  # a_times[T?], a_vals[T?,3]
        # 尝试与 common_t 对齐
        try:
            a_vals_aligned = take_by_time(
                a_times, a_vals, common_t, decimals=TIME_DECIMALS
            )
            if a_vals_aligned.shape[0] == Ti:
                action_i = a_vals_aligned.astype(np.float32)
            else:
                # 步数不匹配，退化为广播首向量
                print(
                    f"[WARN] '{traj_dir}' action length mismatch; broadcasting first vector"
                )
                first = a_vals[0] if a_vals.size else np.zeros(3, dtype=np.float32)
                action_i = np.broadcast_to(first.reshape(1, 3), (Ti, 3)).astype(
                    np.float32
                )
        except Exception:
            # 对齐失败，退化为广播首向量
            first = (
                a_vals[0]
                if (isinstance(a_vals, np.ndarray) and a_vals.size)
                else np.zeros(3, dtype=np.float32)
            )
            print(f"[WARN] '{traj_dir}' action align failed; broadcasting first vector")
            action_i = np.broadcast_to(first.reshape(1, 3), (Ti, 3)).astype(np.float32)

    return state_i, action_i, Ti


# =========================
# 主流程（合并并写 Zarr）
# =========================
def main():
    # 收集子目录（每个子目录一条轨迹）
    subdirs = sorted(
        [
            os.path.join(ROOT_DIR, d)
            for d in os.listdir(ROOT_DIR)
            if os.path.isdir(os.path.join(ROOT_DIR, d))
        ]
    )
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found under ROOT_DIR='{ROOT_DIR}'")

    states, actions = [], []
    episode_ends = []
    total_T, kept = 0, 0

    for d in subdirs:
        state_i, action_i, Ti = process_one_trajectory(d)
        if state_i is None or Ti == 0:
            continue
        states.append(state_i)  # [Ti,12]
        actions.append(action_i)  # [Ti, 3]
        total_T += Ti
        # 注意：episode_ends 语义应为“到当前轨迹结束的累计步数（exclusive end）”
        # ReplayBuffer 里会校验 data[*].shape[0] == episode_ends[-1]
        # 之前用了 total_T - 1 造成 off-by-one，导致断言失败，这里修正为 total_T
        episode_ends.append(total_T)  # 累计长度（exclusive end）
        kept += 1
        print(f"[OK] '{os.path.basename(d)}': Ti={Ti}, total_T={total_T}")

    if kept == 0:
        raise RuntimeError(
            "No valid trajectories processed. Please check file names and contents."
        )

    state_all = np.concatenate(states, axis=0).astype(np.float32)  # [T_total, 12]
    action_all = np.concatenate(actions, axis=0).astype(np.float32)  # [T_total, 3]
    episode_ends = np.asarray(episode_ends, dtype=np.int64)

    print(f"[SUMMARY] Trajectories kept: {kept}")
    print(f"[SUMMARY] state_all:  {state_all.shape}")
    print(f"[SUMMARY] action_all: {action_all.shape}")
    print(
        f"[SUMMARY] episodes:   {episode_ends.shape[0]}, last end idx: {episode_ends[-1]}"
    )

    # 写 Zarr：data/state、data/action、meta/episode_ends（不写时间）
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    os.makedirs(OUT_ZARR_DIR, exist_ok=True)
    root = zarr.open(OUT_ZARR_DIR, mode="w")
    g_data = root.create_group("data")
    g_meta = root.create_group("meta")

    # state
    g_state = g_data.empty(
        name="state",
        shape=state_all.shape,
        chunks=(min(4096, state_all.shape[0]), state_all.shape[1]),
        dtype="f4",
        compressor=compressor,
        overwrite=True,
    )
    g_state[:] = state_all
    g_state.attrs.update(
        {
            "description": "[Fx_l,Fy_l,Fz_l, Mx_l,My_l,Mz_l, Fx_r,Fy_r,Fz_r, Mx_r,My_r,Mz_r]",
            "columns": json.dumps(
                [
                    "Fx_l",
                    "Fy_l",
                    "Fz_l",
                    "Mx_l",
                    "My_l",
                    "Mz_l",
                    "Fx_r",
                    "Fy_r",
                    "Fz_r",
                    "Mx_r",
                    "My_r",
                    "Mz_r",
                ]
            ),
        }
    )

    # action
    g_action = g_data.empty(
        name="action",
        shape=action_all.shape,
        chunks=(min(4096, action_all.shape[0]), action_all.shape[1]),
        dtype="f4",
        compressor=compressor,
        overwrite=True,
    )
    g_action[:] = action_all
    g_action.attrs.update(
        {
            "description": "End effector target position per step (broadcasted if single goal per trajectory)",
            "columns": json.dumps(["x", "y", "z"]),
        }
    )

    # episode_ends
    g_epi = g_meta.empty(
        name="episode_ends",
        shape=episode_ends.shape,
        chunks=(min(1024, episode_ends.shape[0]),),
        dtype="i8",
        compressor=compressor,
        overwrite=True,
    )
    g_epi[:] = episode_ends

    # 根属性
    root.attrs.update(
        {
            "dataset_type": "state",
            "schema": "diffusion_policy/state_v1",
            "note": "Merged demos; action from _end_position.txt (sequence aligned if possible, else broadcast).",
        }
    )

    print(f"[DONE] Saved Zarr to: {OUT_ZARR_DIR}")


if __name__ == "__main__":
    main()


# =========================
# 额外：旧数据集修复工具
# =========================
def repair_episode_ends(zarr_path: str):
    """修复旧版 data_process 生成的 off-by-one episode_ends。

    旧版写入的是 inclusive end 索引 (最后一个值 = 总长度-1)。
    正确语义应为 exclusive end (最后一个值 = 总长度)。
    策略：若最后一个 episode_ends 等于 state.shape[0]-1，则整体 +1 并写回。
    """
    import zarr

    zp = os.path.expanduser(zarr_path)
    root = zarr.open(zp, mode="r+")
    state_arr = root["data/state"]  # type: ignore[assignment]
    # pyright 不识别 zarr Array 的 shape 元素类型，添加忽略
    state_len = int(state_arr.shape[0])  # type: ignore[index]
    epi_arr = root["meta/episode_ends"]  # type: ignore[assignment]
    epi_np = np.asarray(epi_arr[:], dtype=np.int64)
    if epi_np.size == 0:
        print("[REPAIR] episode_ends empty, skip.")
        return
    last = int(epi_np[-1])
    if last == state_len - 1:
        print(
            f"[REPAIR] Detected off-by-one (last episode_ends={last}, state_len={state_len}). Repairing..."
        )
        new_vals = epi_np + 1
        epi_arr.resize(new_vals.shape)  # type: ignore[attr-defined]
        epi_arr[:] = new_vals  # type: ignore[index]
        print(
            f"[REPAIR] Done. New last episode_ends={int(new_vals[-1])} (should equal state_len={state_len})."
        )
    else:
        print(
            f"[REPAIR] No repair needed (last episode_ends={last}, state_len={state_len})."
        )
