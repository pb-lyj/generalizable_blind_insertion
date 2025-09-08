"""
触觉原型发现数据集
适配现有的力数据格式 (3, 20, 20)
"""

import os
import numpy as np
import torch
import json
import hashlib
from torch.utils.data import Dataset
from glob import glob


class TactileForcesDataset(Dataset):
    """
    触觉力数据集，用于原型发现
    """
    
    def __init__(self, data_root, categories=None, start_frame=0, is_train=True, augment=False, 
                 normalize_method='zscore', normalization_config=None):
        """
        Args:
            data_root: 数据根目录路径 (data25.7_aligned)
            categories: 要包含的类别列表，如 ["cir_lar", "rect_med"] 等
            start_frame: 从第几帧开始截取数据
            is_train: 是否加载训练集数据，True=训练集，False=测试集
            augment: 是否应用数据增强
            normalize_method: 归一化方法 ['zscore', 'minmax', 'channel_wise']
                - 'zscore': Z-score标准化（默认）
                - 'minmax': Min-Max归一化到[-1,1]
                - 'channel_wise': 按通道分别进行Min-Max归一化到[-1,1]
            normalization_config: 归一化配置字典，格式为：
                {'method': 'zscore'/'minmax'/'channel_wise', 'params': {...}}
                如果params为None，则自动计算参数
        """
        self.samples = []
        self.augment = augment
        self.normalize_method = normalize_method
        self.train_ratio = 0.8
        self.random_seed = 42
        self.is_train = is_train
        self.data_root = data_root
        self.start_frame = start_frame
        
        # 设置归一化配置
        if normalization_config is None:
            normalization_config = {'method': normalize_method, 'params': None}
        self.normalization_config = normalization_config
        
        # 检查是否使用预计算的归一化参数
        self.use_precomputed_normalization = (
            self.normalization_config.get('params') is not None
        )
        
        # 设置随机种子以确保可重现的划分
        import random
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        if categories is None:
            categories = [
                "cir_lar", "cir_med", "cir_sma",
                "rect_lar", "rect_med", "rect_sma", 
                "tri_lar", "tri_med", "tri_sma"
            ]
        
        total_frames = 0
        valid_frames = 0
        total_episodes = 0
        train_episodes = 0
        test_episodes = 0
        
        for category in categories:
            category_path = os.path.join(data_root, category)
            if not os.path.exists(category_path):
                print(f"⚠️  警告: 类别目录不存在: {category_path}")
                continue
                
            # 获取该类别下的所有数据目录
            data_dirs = sorted([d for d in os.listdir(category_path) 
                              if os.path.isdir(os.path.join(category_path, d))])
            total_episodes += len(data_dirs)
            
            # 计算该类别的训练集数量
            category_train_count = int(len(data_dirs) * self.train_ratio)
            
            # 随机打乱数据目录顺序
            import random
            random.shuffle(data_dirs)
            
            # 划分训练集和测试集
            train_dirs = data_dirs[:category_train_count]
            test_dirs = data_dirs[category_train_count:]
            
            # 根据is_train参数选择使用训练集还是测试集
            selected_dirs = train_dirs if is_train else test_dirs
            
            for data_dir in selected_dirs:
                data_dir_path = os.path.join(category_path, data_dir)
                
                # 查找力数据文件 - 可能是forces.npy或_forces_l.npy/_forces_r.npy
                tactile_files = []
                forces_file = os.path.join(data_dir_path, "forces.npy")
                left_forces_file = os.path.join(data_dir_path, "_forces_l.npy")
                right_forces_file = os.path.join(data_dir_path, "_forces_r.npy")
                
                if os.path.exists(forces_file):
                    tactile_files.append(forces_file)
                elif os.path.exists(left_forces_file) and os.path.exists(right_forces_file):
                    tactile_files.extend([left_forces_file, right_forces_file])
                else:
                    continue
                    
                for tactile_file in tactile_files:
                    try:
                        data = np.load(tactile_file)  # shape (T, C, H, W) or (T, 6, 20, 20)
                        
                        # 处理不同的数据格式
                        if data.shape[1] == 6:  # 合并的左右手数据 (T, 6, 20, 20)
                            total_frames += data.shape[0]
                            
                            # 只使用从 start_frame 开始的数据
                            if data.shape[0] > start_frame:
                                for t in range(start_frame, data.shape[0]):
                                    frame = data[t]  # (6, 20, 20)
                                    
                                    # 分别提取左右手传感器的力数据
                                    left_sensor = frame[0:3]   # 左手传感器: channels 0,1,2
                                    right_sensor = frame[3:6]  # 右手传感器: channels 3,4,5
                                    
                                    self.samples.append(left_sensor)
                                    self.samples.append(right_sensor)
                                    valid_frames += 1
                        
                        elif data.shape[1] == 3:  # 单手数据 (T, 3, 20, 20)
                            total_frames += data.shape[0]
                            
                            # 只使用从 start_frame 开始的数据
                            if data.shape[0] > start_frame:
                                for t in range(start_frame, data.shape[0]):
                                    frame = data[t]  # (3, 20, 20)
                                    self.samples.append(frame)
                                    valid_frames += 1
                        
                    except Exception as e:
                        print(f"⚠️  警告: 无法加载 {tactile_file}: {e}")
                    continue
            
            train_episodes += len(train_dirs)
            test_episodes += len(test_dirs)
                    
        if len(self.samples) == 0:
            raise ValueError(f"未找到有效数据! 检查数据路径: {data_root}")
            
        self.samples = np.stack(self.samples)
        
        # 处理归一化参数
        if not self.use_precomputed_normalization:
            if not self._load_normalization_params(categories):
                self._compute_normalization_params()
        
        # 数据归一化
        self.samples = self._normalize_data(self.samples)
        
        print(f"[TactileForcesDataset] 数据统计:")
        print(f"  - 数据根目录: {data_root}")
        print(f"  - 包含类别: {categories}")
        print(f"  - 归一化方法: {self.normalize_method}")
        print(f"  - 总情节数: {total_episodes}")
        print(f"  - 训练集比例: {self.train_ratio:.1%}")
        print(f"  - 随机种子: {self.random_seed}")
        print(f"  - 训练集情节数: {train_episodes}")
        print(f"  - 测试集情节数: {test_episodes}")
        print(f"  - 当前加载: {'训练集' if is_train else '测试集'}")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - 截取起始帧: {start_frame}")
        print(f"  - 有效帧数: {valid_frames}")
        print(f"  - 总样本数: {len(self.samples)} (包含左右手传感器)")
        print(f"  - 样本形状: {self.samples.shape}")
        print(f"  - 数据范围: [{self.samples.min():.4f}, {self.samples.max():.4f}]")

    def _normalize_data(self, data):
        """
        数据归一化处理，使用预计算的归一化参数
        Args:
            data: 原始数据 (N, 3, 20, 20)
        Returns:
            标准化后的数据
        """
        if self.normalization_config['method'] is None:
            return data
        
        params = self.normalization_config['params']
        if params is None:
            return data
        
        method = self.normalization_config['method']
        normalized_data = np.zeros_like(data)
        
        if method == 'zscore':
            # Z-score标准化
            for i in range(data.shape[1]):  # 对每个通道分别处理
                channel_data = data[:, i, :, :]
                mean = params['channel_means'][i]
                std = params['channel_stds'][i]
                normalized_data[:, i, :, :] = (channel_data - mean) / (std + 1e-8)
        
        elif method == 'minmax':
            # Min-Max归一化到[-1,1]
            for i in range(data.shape[1]):  # 对每个通道分别处理
                channel_data = data[:, i, :, :]
                min_val = params['channel_mins'][i]
                max_val = params['channel_maxs'][i]
                # 先归一化到[0,1]，再映射到[-1,1]
                normalized_data[:, i, :, :] = 2 * (channel_data - min_val) / (max_val - min_val + 1e-8) - 1
        
        elif method == 'channel_wise':
            # 按通道分别进行Min-Max归一化到[-1,1]
            for i in range(data.shape[1]):  # 对每个通道分别处理
                channel_data = data[:, i, :, :]
                min_val = params['channel_mins'][i]
                max_val = params['channel_maxs'][i]
                # 先归一化到[0,1]，再映射到[-1,1]
                normalized_data[:, i, :, :] = 2 * (channel_data - min_val) / (max_val - min_val + 1e-8) - 1
        
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
        
        return normalized_data

    def _compute_normalization_params(self):
        """计算归一化参数"""
        print("🔄 计算触觉数据归一化参数...")
        
        method = self.normalization_config['method']
        if method is None:
            return
        
        data = self.samples  # (N, 3, 20, 20)
        params = {}
        
        if method in ['zscore']:
            # Z-score参数：每个通道的均值和标准差
            channel_means = []
            channel_stds = []
            
            for i in range(data.shape[1]):
                channel_data = data[:, i, :, :]
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                channel_means.append(float(mean))
                channel_stds.append(float(std))
            
            params['channel_means'] = channel_means
            params['channel_stds'] = channel_stds
        
        elif method in ['minmax', 'channel_wise']:
            # Min-Max参数：每个通道的最小值和最大值
            channel_mins = []
            channel_maxs = []
            
            for i in range(data.shape[1]):
                channel_data = data[:, i, :, :]
                min_val = np.min(channel_data)
                max_val = np.max(channel_data)
                channel_mins.append(float(min_val))
                channel_maxs.append(float(max_val))
            
            params['channel_mins'] = channel_mins
            params['channel_maxs'] = channel_maxs
        
        self.normalization_config['params'] = params
        
        # 保存参数到缓存文件
        cache_path = self._get_normalization_cache_path()
        with open(cache_path, 'w') as f:
            json.dump(self.normalization_config, f, indent=2)
        print(f"💾 归一化参数已保存到: {cache_path}")

    def _get_normalization_cache_path(self):
        """生成归一化参数缓存文件路径"""
        config_str = f"{self.data_root}_{self.normalize_method}_{self.train_ratio}_{self.random_seed}_{self.start_frame}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:16]
        cache_dir = os.path.join(self.data_root, ".tactile_normalization_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"tactile_norm_params_{config_hash}.json")

    def _load_normalization_params(self, categories):
        """加载已保存的归一化参数"""
        cache_path = self._get_normalization_cache_path()
        
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self.normalization_config = json.load(f)
            print(f"📁 已加载触觉归一化参数: {cache_path}")
            return True
        return False

    def denormalize_data(self, normalized_data):
        """
        反归一化
        Args:
            normalized_data: 归一化后的数据 (N, 3, 20, 20) 或 (3, 20, 20)
        Returns:
            反归一化后的原始数据
        """
        if self.normalization_config['method'] is None:
            return normalized_data
        
        params = self.normalization_config['params']
        if params is None:
            return normalized_data
        
        method = self.normalization_config['method']
        
        # 确保数据是numpy数组
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.cpu().numpy()
        
        original_shape = normalized_data.shape
        if len(original_shape) == 3:  # (3, 20, 20)
            normalized_data = normalized_data[np.newaxis, :]  # (1, 3, 20, 20)
        
        denormalized_data = np.zeros_like(normalized_data)
        
        if method == 'zscore':
            # 反向zscore: x = norm * std + mean
            for i in range(normalized_data.shape[1]):
                channel_data = normalized_data[:, i, :, :]
                mean = params['channel_means'][i]
                std = params['channel_stds'][i]
                denormalized_data[:, i, :, :] = channel_data * std + mean
        
        elif method in ['minmax', 'channel_wise']:
            # 反向minmax: x = (norm + 1) / 2 * (max - min) + min
            for i in range(normalized_data.shape[1]):
                channel_data = normalized_data[:, i, :, :]
                min_val = params['channel_mins'][i]
                max_val = params['channel_maxs'][i]
                denormalized_data[:, i, :, :] = (channel_data + 1) / 2 * (max_val - min_val) + min_val
        
        # 恢复原始形状
        if len(original_shape) == 3:
            denormalized_data = denormalized_data[0]
        
        return denormalized_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]  # shape: (3, 20, 20)
        
        # 数据增强
        if self.augment and np.random.rand() > 0.5:
            sample = self._apply_augmentation(sample)
        
        # 转换为torch张量
        sample = torch.FloatTensor(sample)
        
        # 重建任务，输入和输出相同
        return {
            'image': sample,
            'target': sample  # 重建目标
        }

    def _apply_augmentation(self, sample):
        """
        应用数据增强
        """
        # 添加高斯噪声
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, sample.shape)
            sample = sample + noise
        
        # 随机翻转
        if np.random.rand() > 0.5:
            sample = np.flip(sample, axis=1)  # 水平翻转
        
        if np.random.rand() > 0.5:
            sample = np.flip(sample, axis=2)  # 垂直翻转
        
        return sample.copy()  # 确保返回连续内存


def create_train_test_tactile_datasets(data_root, categories=None, start_frame=0, 
                                     augment_train=True, augment_test=False, 
                                     normalize_method='zscore'):
    """
    便捷函数：创建训练集和测试集的触觉数据集
    
    Args:
        data_root: 数据根目录路径
        categories: 要包含的类别列表
        start_frame: 起始帧
        augment_train: 训练集是否使用数据增强
        augment_test: 测试集是否使用数据增强
        normalize_method: 归一化方法 ['zscore', 'minmax', 'channel_wise']
    
    Returns:
        tuple: (train_dataset, test_dataset, normalization_config)
    """
    # 1. 先创建训练集
    train_dataset = TactileForcesDataset(
        data_root=data_root,
        categories=categories,
        start_frame=start_frame,
        is_train=True,
        augment=augment_train,
        normalize_method=normalize_method
    )
    
    # 2. 创建测试集，使用训练集的归一化参数
    test_dataset = TactileForcesDataset(
        data_root=data_root,
        categories=categories,
        start_frame=start_frame,
        is_train=False,
        augment=augment_test,
        normalize_method=normalize_method,
        normalization_config=train_dataset.normalization_config
    )
    
    print(f"📊 触觉数据归一化信息:")
    print(f"   归一化方法: {train_dataset.normalization_config.get('method', 'None')}")
    print(f"   测试集使用预计算参数: {test_dataset.use_precomputed_normalization}")
    
    return train_dataset, test_dataset, train_dataset.normalization_config


class TactileSampleDataset(Dataset):
    """
    原始的触觉样本数据集 (向后兼容)
    """
    def __init__(self, root_dirs, start_frame=20):
        """
        Args:
            root_dirs: 字符串或字符串列表，数据目录路径
            start_frame: 从第几帧开始截取数据
        """
        self.samples = []
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
            
        total_frames = 0
        valid_frames = 0
        
        for root_dir in root_dirs:
            paths = sorted(glob(os.path.join(root_dir, "episode_*", "tactile.npy")))
            for path in paths:
                data = np.load(path)  # shape (T, 6, H, W)
                total_frames += data.shape[0]
                
                # 只使用从 start_frame 开始的数据
                if data.shape[0] > start_frame:
                    for t in range(start_frame, data.shape[0]):
                        frame = data[t, -1]  # use final frame
                        self.samples.append(frame[0:3])  # sensor 1
                        self.samples.append(frame[3:6])  # sensor 2
                        valid_frames += 1
                    
        self.samples = np.stack(self.samples)
        self.samples[:, 0:2] /= 0.05  # normalize XY range
        
        total_episodes = sum(len(glob(os.path.join(root, "episode_*"))) 
                           for root in root_dirs)
        print(f"[TactileSampleDataset] 数据统计:")
        print(f"  - 环境数量: {len(root_dirs)}")
        print(f"  - 情节数量: {total_episodes}")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - 截取起始帧: {start_frame}")
        print(f"  - 有效帧数: {valid_frames}")
        print(f"  - 总样本数: {len(self.samples)} (包含左右传感器)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32)
