"""
简化的策略学习数据集 - 仅支持MLP所需的基本功能
"""
import os
import torch
import numpy as np
import random
import hashlib
import json
from torch.utils.data import Dataset

class PointPairDataset(Dataset):
    """
    简化的策略学习数据集，仅保留MLP训练所需的功能
    """
    
    def __init__(self, data_root, categories=None, is_train=True, use_resultant = True, use_forces = False,
                 normalization_config=None, prediction_step=1):
        """
        Args:
            data_root: 数据根目录路径
            categories: 要包含的类别列表
            is_train: 是否加载训练集数据
            prediction_step: 预测步长，t时刻预测t+prediction_step时刻目标 (默认1)
            normalization_config: 归一化配置字典，格式为：
                {
                    'actions': {'method': 'minmax'/'zscore', 'params': {...}},
                    'forces': {'method': 'minmax'/'zscore', 'params': {...}},
                    'resultants': {'method': 'minmax'/'zscore', 'params': {...}}
                }
                如果params为None，则自动计算参数
        """
        self.data_root = data_root
        self.use_resultant = use_resultant
        self.use_forces = use_forces
        self.prediction_step = prediction_step
        self.train_ratio = 0.8
        self.random_seed = 42
        self.is_train = is_train
        
        # 设置默认归一化配置
        if normalization_config is None:
            normalization_config = {
                'actions': {'method': 'zscore', 'params': None},
                'resultants': {'method': 'zscore', 'params': None},
                'forces': {'method': 'zscore', 'params': None}
            }
        self.normalization_config = normalization_config
        
        # 检查是否使用预计算的归一化参数
        self.use_precomputed_normalization = all(
            config['params'] is not None 
            for config in normalization_config.values()
        )
            
        # 初始化轨迹数据和索引
        self._load_trajectory_metadata(categories)
        self._build_indices()
        
        # 处理归一化参数
        if not self.use_precomputed_normalization:
            if not self._load_normalization_params(categories):
                self._compute_global_normalization_params_streaming(categories)
            
        self._print_dataset_info()

    def _load_trajectory_metadata(self, categories):
        """加载轨迹元数据"""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        if categories is None:
            categories = [
                "cir_lar", "cir_med", "cir_sma",
                "rect_lar", "rect_med", "rect_sma", 
                "tri_lar", "tri_med", "tri_sma"
            ]
        
        self.trajectories = []
        
        for category in categories:
            category_path = os.path.join(self.data_root, category)
            if not os.path.exists(category_path):
                print(f"⚠️ 类别路径不存在: {category_path}")
                continue
            
            trajectory_dirs = sorted([
                d for d in os.listdir(category_path) 
                if os.path.isdir(os.path.join(category_path, d))
            ])
            
            for traj_dir in trajectory_dirs:
                traj_path = os.path.join(category_path, traj_dir)
                self.trajectories.append({
                    'path': traj_path,
                    'category': category,
                    'dir_name': traj_dir
                })
        
        # 划分训练/测试轨迹
        train_count = int(len(self.trajectories) * self.train_ratio)
        random.shuffle(self.trajectories)
        
        if self.is_train:
            self.trajectories = self.trajectories[:train_count]
        else:
            self.trajectories = self.trajectories[train_count:]


    def _build_indices(self):
        """构建数据索引"""
        self.indices = []
        
        for traj_idx, traj_info in enumerate(self.trajectories):
            # 读取末端位置数据确定长度
            position_data = np.load(os.path.join(traj_info['path'], "_end_position.npy"))
            position_length = len(position_data)
            
            # 时序模式：t时刻预测t+n时刻，确保t+n不超出轨迹边界
            # t的范围：[0, position_length - prediction_step - 1]
            max_t_idx = position_length - self.prediction_step
            for step_idx in range(max_t_idx):  # step_idx就是t的索引
                self.indices.append({
                    'traj_idx': traj_idx,
                    't_idx': step_idx,  # t时刻索引
                    'target_idx': step_idx + self.prediction_step  # t+n时刻索引
                })

        
        # 打乱索引
        random.shuffle(self.indices)
        print(f"时序模式 (预测步长={self.prediction_step}): {len(self.indices)} 样本已加载。")
        

    def _print_dataset_info(self):
        """打印数据集信息"""
        total_trajectories = len(self.trajectories)
        total_samples = len(self.indices)
        
        print(f"[FlexiblePolicyDataset] 数据统计:")
        print(f"  - 轨迹数量: {total_trajectories}")
        print(f"  - 样本数量: {total_samples}")
        print(f"  - 当前集合: {'训练集' if self.is_train else '测试集'}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """获取单个样本"""
        index_info = self.indices[idx]
        traj_info = self.trajectories[index_info['traj_idx']]
        traj_path = traj_info['path']
        t_idx = index_info['t_idx']
        target_idx = index_info['target_idx']
        
        # 时序模式：返回t时刻输入和t+n时刻目标
        return self._load_singlepoint_data(traj_path, traj_info, t_idx, target_idx)


    def _load_singlepoint_data(self, traj_path, traj_info, t_idx, target_idx):
        """
        加载时序数据：t时刻输入 -> t+n时刻目标
        Args:
            t_idx: t时刻的索引
            target_idx: t+n时刻的索引
        Returns:
            result: 包含t时刻输入数据和t+n时刻目标动作的字典
        """
        # 加载末端位置数据
        position_data = np.load(os.path.join(traj_path, "_end_position.npy"))
        
        # t时刻的动作（当前动作）
        current_action = position_data[t_idx, 1:4]  # XYZ坐标
        current_action = self._normalize_data(current_action, 'actions')
        
        # t+n时刻的动作（目标动作）
        next_action = position_data[target_idx, 1:4]  # XYZ坐标
        next_action = self._normalize_data(next_action, 'actions')
        
        result = {
            'current_action': torch.FloatTensor(current_action),  # t时刻动作，作为输入
            'next_action': torch.FloatTensor(next_action),        # t+n时刻动作，作为目标
            'category': traj_info['category'],
            'trajectory_id': traj_info['dir_name'],
            't_idx': t_idx,
            'target_idx': target_idx,
            'prediction_step': self.prediction_step  # 新增：预测步长信息
        }
        
        # 加载t时刻的触觉数据（输入特征）
        if self.use_resultant:
            # 加载resultants数据
            resultant_force_l_data = np.load(os.path.join(traj_path, "_resultant_force_l.npy"))
            resultant_force_r_data = np.load(os.path.join(traj_path, "_resultant_force_r.npy"))
            resultant_moment_l_data = np.load(os.path.join(traj_path, "_resultant_moment_l.npy"))
            resultant_moment_r_data = np.load(os.path.join(traj_path, "_resultant_moment_r.npy"))
            
            resultant_force_l = self._normalize_data(resultant_force_l_data[t_idx], 'resultants')
            resultant_force_r = self._normalize_data(resultant_force_r_data[t_idx], 'resultants')
            resultant_moment_l = self._normalize_data(resultant_moment_l_data[t_idx], 'resultants')
            resultant_moment_r = self._normalize_data(resultant_moment_r_data[t_idx], 'resultants')
            
            result['resultant_force_l'] = torch.FloatTensor(resultant_force_l)  
            result['resultant_force_r'] = torch.FloatTensor(resultant_force_r)  
            result['resultant_moment_l'] = torch.FloatTensor(resultant_moment_l)  
            result['resultant_moment_r'] = torch.FloatTensor(resultant_moment_r) 
        
        if self.use_forces:
            forces_l = np.load(os.path.join(traj_path, "_forces_l.npy"))
            forces_r = np.load(os.path.join(traj_path, "_forces_r.npy"))
            
            forces_l = self._normalize_data(forces_l[t_idx], 'forces')
            forces_r = self._normalize_data(forces_r[t_idx], 'forces')
            
            result['forces_l'] = torch.FloatTensor(forces_l)  # (3, 20, 20)
            result['forces_r'] = torch.FloatTensor(forces_r)  # (3, 20, 20)
        
        return result

    # 归一化方法
    def _normalize_data(self, data, data_type):
        """数据归一化处理"""
        ## 检查参数
        if data_type not in self.normalization_config:
            return data
        
        config = self.normalization_config[data_type]
        if config['method'] is None:
            return data
        
        params = config['params']
        if params is None:
            return data
        
        ## 应用归一化
        method = config['method']
        if method == 'zscore':
            return (data - params['mean']) / (params['std'] + 1e-8)
        elif method == 'minmax':
            return 2 * (data - params['min']) / (params['max'] - params['min'] + 1e-8) - 1
        
        return data

    def _get_normalization_cache_path(self, categories):
        """生成归一化参数缓存文件路径"""
        config_str = f"{self.data_root}_{categories}_{self.train_ratio}_{self.random_seed}_{self.normalization_config}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:16]
        cache_dir = os.path.join(self.data_root, ".normalization_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"norm_params_{config_hash}.json")

    def _compute_global_normalization_params_streaming(self, categories):
        """计算全局归一化参数"""
        print("🔄 计算全局归一化参数...")
        
        # 初始化统计累积器
        stats = {}
        for data_type in ['actions', 'resultants']:
            config = self.normalization_config.get(data_type)
            if config and config['method'] is not None:
                stats[data_type] = {
                    'count': 0,
                    'sum': 0.0,
                    'sum_sq': 0.0,
                    'min_val': float('inf'),
                    'max_val': float('-inf')
                }
        
        # 只有当使用forces时才初始化forces统计器
        if self.use_forces:
            config = self.normalization_config.get('forces')
            if config and config['method'] is not None:
                stats['forces'] = {
                    'count': 0,
                    'sum': 0.0,
                    'sum_sq': 0.0,
                    'min_val': float('inf'),
                    'max_val': float('-inf')
                }
        
        # 流式处理训练集轨迹
        for traj_info in self.trajectories:
            traj_path = traj_info['path']
            
            # 处理actions数据 - 使用末端位置数据
            if 'actions' in stats:
                position_data = np.load(os.path.join(traj_path, "_end_position.npy"))[:, 1:4]  # 只取XYZ列
                self._update_stats(stats['actions'], position_data)
            
            # 处理resultants数据
            if 'resultants' in stats:
                resultant_files = [
                    "_resultant_force_l.npy", "_resultant_force_r.npy",
                    "_resultant_moment_l.npy", "_resultant_moment_r.npy"
                ]
                for file in resultant_files:
                    data = np.load(os.path.join(traj_path, file))
                    self._update_stats(stats['resultants'], data)
            
            # 处理forces数据
            if 'forces' in stats and self.use_forces:
                force_files = ["_forces_l.npy", "_forces_r.npy"]
                for file in force_files:
                    file_path = os.path.join(traj_path, file)
                    if os.path.exists(file_path):
                        data = np.load(file_path)
                        self._update_stats(stats['forces'], data)
        
        # 计算最终的归一化参数并更新配置
        for data_type, stat in stats.items():
            config = self.normalization_config[data_type]
            method = config['method']
            
            if method is None:
                continue
                
            mean = stat['sum'] / stat['count']
            variance = stat['sum_sq'] / stat['count'] - mean ** 2
            std = np.sqrt(variance)
            
            if method == 'zscore':
                config['params'] = {
                    'mean': mean,
                    'std': std
                }
            elif method == 'minmax':
                config['params'] = {
                    'min': stat['min_val'],
                    'max': stat['max_val']
                }
        
        # 保存参数到缓存文件
        cache_path = self._get_normalization_cache_path(categories)
        with open(cache_path, 'w') as f:
            json.dump(self.normalization_config, f, indent=2)
        print(f"💾 归一化参数已保存到: {cache_path}")

    def _update_stats(self, stat_dict, data):
        """更新统计信息"""
        data = data.astype(np.float64)
        stat_dict['count'] += data.size
        stat_dict['sum'] += np.sum(data)
        stat_dict['sum_sq'] += np.sum(data ** 2)
        stat_dict['min_val'] = min(stat_dict['min_val'], np.min(data))
        stat_dict['max_val'] = max(stat_dict['max_val'], np.max(data))

    def _load_normalization_params(self, categories):
        """加载已保存的归一化参数"""
        cache_path = self._get_normalization_cache_path(categories)
        
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self.normalization_config = json.load(f)
            print(f"📁 已加载归一化参数: {cache_path}")
            return True
        return False

    def denormalize_data(self, normalized_data, data_type):
        """反归一化"""
        if data_type not in self.normalization_config:
            return normalized_data
        
        config = self.normalization_config[data_type]
        if config['method'] is None or config['params'] is None:
            return normalized_data
        
        method = config['method']
        params = config['params']
        
        if method == 'minmax':
            # 反向minmax: x = (norm + 1) / 2 * (max - min) + min
            return (normalized_data + 1) / 2 * (params['max'] - params['min']) + params['min']
        elif method == 'zscore':
            # 反向zscore: x = norm * std + mean
            return normalized_data * params['std'] + params['mean']
        
        return normalized_data

    # def get_normalization_params(self):
    #     """获取归一化参数"""
    #     return self.normalization_config.copy()


def create_classic_datasets(data_root, categories=None,
                           normalization_config=None, prediction_step=1):
    """
    创建训练集和测试集
    Args:
        prediction_step: 预测步长，t时刻预测t+prediction_step时刻目标
    """
    # 1. 先创建训练集
    train_dataset = PointPairDataset(
        data_root=data_root,
        categories=categories,
        is_train=True,
        normalization_config=normalization_config,
        prediction_step=prediction_step
    )
    
    # 2. 创建测试集，使用训练集的归一化参数
    test_dataset = PointPairDataset(
        data_root=data_root,
        categories=categories,
        is_train=False,
        normalization_config=train_dataset.normalization_config,
        prediction_step=prediction_step
    )
    
    print(f"📊 归一化信息:")
    action_params = train_dataset.normalization_config.get('actions', {})
    result_params = train_dataset.normalization_config.get('resultants', {})
    print(f"   Actions归一化: {action_params.get('method', 'None')}")
    print(f"   Resultants归一化: {result_params.get('method', 'None')}")
    print(f"   测试集使用预计算参数: {test_dataset.use_precomputed_normalization}")
    
    return train_dataset, test_dataset, train_dataset.normalization_config
