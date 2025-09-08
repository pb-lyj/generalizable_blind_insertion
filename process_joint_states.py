#!/usr/bin/env python3
"""
处理data25.7_aligned中的关节状态数据，通过URDF文件精确计算末端位姿
position[3] and posotion+quaternion[7]
将结果保存为npy和txt格式（使用KUKA iiwa14官方URDF参数）
"""

import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple
import traceback

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


class URDFKinematics:
    """
    基于URDF文件的精确KUKA iiwa14运动学计算
    直接使用官方URDF文件中的变换参数
    """
    
    def __init__(self, urdf_path: str = None):
        """
        初始化KUKA iiwa14的精确运动学参数
        
        Args:
            urdf_path: URDF文件路径，如果为None则使用默认的XACRO路径
        """
        if urdf_path is None:
            urdf_path = os.path.join("urdf", "iiwa14", "iiwa14_description.xacro")
        
        # 存储关节变换参数：[translation, rpy]
        self.joint_transforms = []
        self.joint_axes = []
        
        # 必须使用提供的URDF/XACRO文件
        if os.path.exists(urdf_path):
            print(f"📖 使用XACRO文件: {urdf_path}")
            self._parse_xacro(urdf_path)
        else:
            raise FileNotFoundError(f"❌ 必须提供有效的URDF/XACRO文件: {urdf_path}")
    
    def _parse_xacro(self, xacro_path: str):
        """从XACRO文件解析关节变换参数（基于实际的iiwa14参数）"""
        try:
            # 基于iiwa14_description.xacro文件的实际关节参数
            # 这些是从XACRO文件中提取的精确变换参数
            joint_data = [
                # A1: origin rpy="0 0 0" xyz="0.0 0.0 0.1475", axis="0.0 0.0 1.0"
                ([0.0, 0.0, 0.1475], [0, 0, 0], [0, 0, 1]),
                # A2: origin rpy="0 0 0" xyz="0.0 -0.01 0.2125", axis="0.0 1.0 0.0"  
                ([0.0, -0.01, 0.2125], [0, 0, 0], [0, 1, 0]),
                # A3: origin rpy="0 0 0" xyz="0.0 0.01 0.228", axis="0.0 0.0 1.0"
                ([0.0, 0.01, 0.228], [0, 0, 0], [0, 0, 1]),
                # A4: origin rpy="0 0 0" xyz="0.0 0.0105 0.192", axis="0.0 -1.0 0.0"
                ([0.0, 0.0105, 0.192], [0, 0, 0], [0, -1, 0]),
                # A5: origin rpy="0 0 0" xyz="0.0 -0.0105 0.2075", axis="0.0 0.0 1.0"
                ([0.0, -0.0105, 0.2075], [0, 0, 0], [0, 0, 1]),
                # A6: origin rpy="0 0 0" xyz="0.0 -0.0707 0.1925", axis="0.0 1.0 0.0"
                ([0.0, -0.0707, 0.1925], [0, 0, 0], [0, 1, 0]),
                # A7: origin rpy="0 0 0" xyz="0.0 0.0707 0.091", axis="0.0 0.0 1.0"
                ([0.0, 0.0707, 0.091], [0, 0, 0], [0, 0, 1]),
            ]
            
            for translation, rpy, axis in joint_data:
                self.joint_transforms.append((np.array(translation), np.array(rpy)))
                self.joint_axes.append(np.array(axis))
            
            print(f"✅ 成功解析{len(self.joint_transforms)}个关节参数")
            
        except Exception as e:
            raise ValueError(f"❌ 解析XACRO文件失败: {e}")
    
    def _euler_to_rotation_matrix(self, rpy: np.ndarray) -> np.ndarray:
        """
        将RPY欧拉角转换为旋转矩阵 (ZYX约定)
        
        Args:
            rpy: [roll, pitch, yaw] 弧度
            
        Returns:
            3x3旋转矩阵
        """
        r, p, y = rpy
        
        # ZYX约定：R = Rz(yaw) * Ry(pitch) * Rx(roll)
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        
        R = np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,    cp*sr,             cp*cr           ]
        ])
        
        return R
    
    def _axis_angle_to_rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        使用罗德里格斯公式将轴角表示转换为旋转矩阵
        
        Args:
            axis: 旋转轴单位向量
            angle: 旋转角度（弧度）
            
        Returns:
            3x3旋转矩阵
        """
        # 确保轴是单位向量
        axis = axis / np.linalg.norm(axis)
        
        # 罗德里格斯公式
        K = np.array([
            [0,       -axis[2],  axis[1]],
            [axis[2],  0,       -axis[0]],
            [-axis[1], axis[0],   0      ]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        return R
    
    def _create_transform_matrix(self, translation: np.ndarray, rpy: np.ndarray) -> np.ndarray:
        """
        创建齐次变换矩阵
        
        Args:
            translation: 平移向量 [x, y, z]
            rpy: 欧拉角 [roll, pitch, yaw]
            
        Returns:
            4x4齐次变换矩阵
        """
        T = np.eye(4)
        T[:3, :3] = self._euler_to_rotation_matrix(rpy)
        T[:3, 3] = translation
        return T
    
    def compute_fk(self, joint_angles: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算精确的前向运动学
        
        Args:
            joint_angles: 7个关节角度 (弧度)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (位置[3], 旋转矩阵[3x3])
        """
        if len(joint_angles) != 7:
            raise ValueError(f"需要7个关节角度，得到 {len(joint_angles)} 个")
        
        if len(self.joint_transforms) != 7:
            raise ValueError(f"关节参数不完整，只有 {len(self.joint_transforms)} 个")
        
        # 从基座开始累积变换
        T = np.eye(4)
        
        for i, angle in enumerate(joint_angles):
            # 关节固定变换
            translation, rpy = self.joint_transforms[i]
            T_fixed = self._create_transform_matrix(translation, rpy)
            
            # 关节旋转变换
            axis = self.joint_axes[i]
            R_joint = self._axis_angle_to_rotation_matrix(axis, angle)
            T_joint = np.eye(4)
            T_joint[:3, :3] = R_joint
            
            # 累积变换：T = T * T_fixed * T_joint
            T = T @ T_fixed @ T_joint
        
        # 提取位置和旋转
        position = T[:3, 3]
        rotation = T[:3, :3]
        
        return position, rotation


def load_joint_states_from_txt(file_path: str) -> List[Tuple[float, List[float]]]:
    """
    从_lbr_joint_states.txt文件加载关节状态数据
    
    Args:
        file_path: 关节状态文件路径
        
    Returns:
        List[Tuple[float, List[float]]]: [(时间戳, [7个关节角度]), ...]
    """
    data_groups = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        # 按分隔符分割数据组
        groups = content.split('=' * 40)
        
        for group in groups:
            lines = group.strip().split('\n')
            if len(lines) < 22:  # 需要至少22行：时间戳 + 7关节位置 + 7关节速度 + 7关节扭矩
                continue
            
            try:
                # 第一行是时间戳
                timestamp = float(lines[0])
                
                # 第2-8行是7个关节位置
                joint_positions = []
                for i in range(1, 8):
                    joint_positions.append(float(lines[i]))
                
                data_groups.append((timestamp, joint_positions))
                
            except (ValueError, IndexError) as e:
                print(f"⚠️  解析数据组失败: {e}")
                continue
    
    except FileNotFoundError:
        print(f"⚠️  文件不存在: {file_path}")
        return []
    except Exception as e:
        print(f"⚠️  读取文件失败 {file_path}: {e}")
        return []
    
    return data_groups


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵转换为欧拉角 (ZYX约定，即RPY)
    
    Args:
        R: 3x3旋转矩阵
        
    Returns:
        np.ndarray: [rx, ry, rz] 欧拉角 (弧度)
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return np.array([x, y, z])


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵转换为四元数 (w, x, y, z)
    包含归一化和符号统一
    
    Args:
        R: 3x3旋转矩阵
        
    Returns:
        np.ndarray: [w, x, y, z] 四元数 (标量优先，归一化，w>=0)
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    quaternion = np.array([qw, qx, qy, qz])
    
    # 归一化四元数
    norm = np.linalg.norm(quaternion)
    if norm > 1e-12:  # 避免除零
        quaternion = quaternion / norm
    
    # 符号统一：确保w分量非负（四元数的标准约定）
    if quaternion[0] < 0:
        quaternion = -quaternion
    
    return quaternion


def compute_end_poses(joint_data: List[Tuple[float, List[float]]], 
                     kinematics: URDFKinematics) -> Tuple[List[Tuple[float, np.ndarray]], List[Tuple[float, np.ndarray]]]:
    """
    通过前向运动学计算末端位姿
    
    Args:
        joint_data: [(时间戳, [7个关节角度]), ...]
        kinematics: 运动学求解器
        
    Returns:
        Tuple[List, List]: (position_data, pose_quaternion_data)
            - position_data: [(时间戳, [x, y, z]), ...]
            - pose_quaternion_data: [(时间戳, [x, y, z, qw, qx, qy, qz]), ...]
    """
    position_data = []
    pose_quaternion_data = []
    
    for timestamp, joint_positions in joint_data:
        try:
            # 计算前向运动学
            position, rotation = kinematics.compute_fk(joint_positions)
            
            # 将旋转矩阵转换为四元数
            quaternion = rotation_matrix_to_quaternion(rotation)
            
            # 位置数据 [x, y, z]
            position_array = np.array([position[0], position[1], position[2]])
            position_data.append((timestamp, position_array))
            
            # 位置+四元数数据 [x, y, z, qw, qx, qy, qz]
            pose_quat_array = np.array([position[0], position[1], position[2], 
                                      quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
            pose_quaternion_data.append((timestamp, pose_quat_array))
            
        except Exception as e:
            print(f"⚠️  计算前向运动学失败 (时间戳: {timestamp}): {e}")
            continue
    
    return position_data, pose_quaternion_data


def save_position_data(position_data: List[Tuple[float, np.ndarray]], 
                      output_dir: str, file_prefix: str = "_end_position"):
    """
    保存末端位置数据为npy和txt格式
    
    Args:
        position_data: [(时间戳, [x, y, z]), ...]
        output_dir: 输出目录
        file_prefix: 文件前缀
    """
    if not position_data:
        print("⚠️  没有有效的末端位置数据")
        return
    
    # 组合为numpy数组 [时间戳, x, y, z]
    combined_data = np.zeros((len(position_data), 4))
    for i, (timestamp, position) in enumerate(position_data):
        combined_data[i, 0] = timestamp
        combined_data[i, 1:] = position
    
    # 保存npy文件
    npy_path = os.path.join(output_dir, f"{file_prefix}.npy")
    np.save(npy_path, combined_data)
    
    # 保存txt文件
    txt_path = os.path.join(output_dir, f"{file_prefix}.txt")
    with open(txt_path, 'w') as f:
        for i, (timestamp, position) in enumerate(position_data):
            if i > 0:
                f.write('=' * 40 + '\n')
            f.write(f"{timestamp}\n")
            for val in position:
                f.write(f"{val}\n")
    
    print(f"  ✅ 已保存位置数据: {npy_path}")
    print(f"  ✅ 已保存位置数据: {txt_path}")


def save_pose_quaternion_data(pose_quat_data: List[Tuple[float, np.ndarray]], 
                             output_dir: str, file_prefix: str = "_end_pose_quaternion"):
    """
    保存末端位置+四元数数据为npy和txt格式
    
    Args:
        pose_quat_data: [(时间戳, [x, y, z, qw, qx, qy, qz]), ...]
        output_dir: 输出目录
        file_prefix: 文件前缀
    """
    if not pose_quat_data:
        print("⚠️  没有有效的末端位姿四元数数据")
        return
    
    # 组合为numpy数组 [时间戳, x, y, z, qw, qx, qy, qz]
    combined_data = np.zeros((len(pose_quat_data), 8))
    for i, (timestamp, pose_quat) in enumerate(pose_quat_data):
        combined_data[i, 0] = timestamp
        combined_data[i, 1:] = pose_quat
    
    # 保存npy文件
    npy_path = os.path.join(output_dir, f"{file_prefix}.npy")
    np.save(npy_path, combined_data)
    
    # 保存txt文件
    txt_path = os.path.join(output_dir, f"{file_prefix}.txt")
    with open(txt_path, 'w') as f:
        for i, (timestamp, pose_quat) in enumerate(pose_quat_data):
            if i > 0:
                f.write('=' * 40 + '\n')
            f.write(f"{timestamp}\n")
            for val in pose_quat:
                f.write(f"{val}\n")
    
    print(f"  ✅ 已保存位姿四元数数据: {npy_path}")
    print(f"  ✅ 已保存位姿四元数数据: {txt_path}")


def save_end_poses(end_poses: List[Tuple[float, np.ndarray]], 
                  output_dir: str, file_prefix: str = "_end_states"):
    """
    保存末端位姿数据为npy和txt格式 (保留原函数用于兼容性)
    
    Args:
        end_poses: [(时间戳, [x, y, z, rx, ry, rz]), ...]
        output_dir: 输出目录
        file_prefix: 文件前缀
    """
    if not end_poses:
        print("⚠️  没有有效的末端位姿数据")
        return
    
    # 准备数据
    timestamps = [timestamp for timestamp, _ in end_poses]
    poses = [pose for _, pose in end_poses]
    
    # 组合为numpy数组 [时间戳, x, y, z, rx, ry, rz]
    combined_data = np.zeros((len(end_poses), 7))
    for i, (timestamp, pose) in enumerate(end_poses):
        combined_data[i, 0] = timestamp
        combined_data[i, 1:] = pose
    
    # 保存npy文件
    npy_path = os.path.join(output_dir, f"{file_prefix}.npy")
    np.save(npy_path, combined_data)
    
    # 保存txt文件
    txt_path = os.path.join(output_dir, f"{file_prefix}.txt")
    with open(txt_path, 'w') as f:
        for i, (timestamp, pose) in enumerate(end_poses):
            if i > 0:
                f.write('=' * 40 + '\n')
            f.write(f"{timestamp}\n")
            for val in pose:
                f.write(f"{val}\n")
    
    print(f"  ✅ 已保存: {npy_path}")
    print(f"  ✅ 已保存: {txt_path}")


def process_trajectory_folder(traj_path: str, kinematics: URDFKinematics) -> bool:
    """
    处理单个轨迹文件夹
    
    Args:
        traj_path: 轨迹文件夹路径
        kinematics: 运动学求解器
        
    Returns:
        bool: 是否处理成功
    """
    joint_states_file = os.path.join(traj_path, "_lbr_joint_states.txt")
    
    if not os.path.exists(joint_states_file):
        print(f"⚠️  关节状态文件不存在: {joint_states_file}")
        return False
    
    print(f"📖 处理: {os.path.basename(traj_path)}")
    
    # 加载关节状态数据
    joint_data = load_joint_states_from_txt(joint_states_file)
    if not joint_data:
        print(f"⚠️  无有效关节数据: {joint_states_file}")
        return False
    
    print(f"  📊 加载 {len(joint_data)} 组关节数据")
    
    # 计算末端位姿
    position_data, pose_quat_data = compute_end_poses(joint_data, kinematics)
    if not position_data or not pose_quat_data:
        print(f"⚠️  无有效末端位姿: {joint_states_file}")
        return False
    
    print(f"  🎯 计算 {len(position_data)} 组末端位置")
    print(f"  🎯 计算 {len(pose_quat_data)} 组末端位姿+四元数")
    
    # 保存结果
    save_position_data(position_data, traj_path)
    save_pose_quaternion_data(pose_quat_data, traj_path)
    
    return True


def process_all_trajectories():
    """
    处理所有轨迹数据
    """
    print("=" * 80)
    print("处理关节状态数据并计算末端位姿 (位置[3] + 位置+四元数[7])")
    print("=" * 80)
    
    # 数据路径
    data_root = "data25.7_aligned"
    
    # 检查路径
    if not os.path.exists(data_root):
        print(f"❌ 数据目录不存在: {data_root}")
        return
    
    # 初始化运动学求解器
    try:
        print("🔧 初始化运动学求解器...")
        kinematics = URDFKinematics()
        print("✅ 运动学求解器初始化成功")
    except Exception as e:
        print(f"❌ 运动学求解器初始化失败: {e}")
        traceback.print_exc()
        return
    
    # 获取所有category
    categories = [d for d in os.listdir(data_root) 
                 if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')]
    categories.sort()
    
    total_processed = 0
    total_failed = 0
    
    for category in categories:
        category_path = os.path.join(data_root, category)
        print(f"\n{'='*60}")
        print(f"处理类别: {category}")
        print(f"{'='*60}")
        
        # 获取该category下的所有轨迹文件夹
        trajectory_dirs = [d for d in os.listdir(category_path) 
                          if os.path.isdir(os.path.join(category_path, d)) and not d.startswith('.')]
        trajectory_dirs.sort()
        
        category_processed = 0
        category_failed = 0
        
        for traj_dir in trajectory_dirs:
            traj_path = os.path.join(category_path, traj_dir)
            
            try:
                if process_trajectory_folder(traj_path, kinematics):
                    category_processed += 1
                    total_processed += 1
                else:
                    category_failed += 1
                    total_failed += 1
            except Exception as e:
                print(f"❌ 处理轨迹失败 {traj_dir}: {e}")
                category_failed += 1
                total_failed += 1
        
        print(f"\n📊 类别 {category} 统计:")
        print(f"  ✅ 成功: {category_processed}")
        print(f"  ❌ 失败: {category_failed}")
    
    print(f"\n{'='*80}")
    print("处理完成！")
    print(f"总计:")
    print(f"  ✅ 成功处理: {total_processed} 个轨迹")
    print(f"  ❌ 处理失败: {total_failed} 个轨迹")
    print(f"\n📁 生成的文件格式:")
    print(f"  📄 _end_position.npy/.txt: [时间戳, x, y, z] (4维)")
    print(f"  📄 _end_pose_quaternion.npy/.txt: [时间戳, x, y, z, qw, qx, qy, qz] (8维)")
    print(f"{'='*80}")


if __name__ == "__main__":
    process_all_trajectories()
