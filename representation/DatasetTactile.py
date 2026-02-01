"""
Tactile Force Dataset
Adapted for force data format (3, 20, 20)
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
    Tactile force dataset for representation learning.
    """
    
    def __init__(self, data_root, categories=None, start_frame=0, is_train=True, augment=False, 
                 normalize_method='zscore', normalization_config=None):
        """
        Args:
            data_root: Data root directory path (e.g., data25.7_aligned)
            categories: List of categories to include, e.g., ["cir_lar", "rect_med"]
            start_frame: Starting frame index for data slicing
            is_train: If True, load training set; if False, load test set
            augment: Whether to apply data augmentation
            normalize_method: Normalization method ['zscore', 'minmax', 'channel_wise']
            normalization_config: Normalization config dict: {'method': str, 'params': dict or None}
        """
        self.samples = []
        self.augment = augment
        self.normalize_method = normalize_method
        self.train_ratio = 0.8
        self.random_seed = 42
        self.is_train = is_train
        self.data_root = data_root
        self.start_frame = start_frame
        
        if normalization_config is None:
            normalization_config = {'method': normalize_method, 'params': None}
        self.normalization_config = normalization_config
        
        self.use_precomputed_normalization = (
            self.normalization_config.get('params') is not None
        )
        
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
                print(f"⚠️  Warning: Category directory does not exist: {category_path}")
                continue
                
            data_dirs = sorted([d for d in os.listdir(category_path) 
                              if os.path.isdir(os.path.join(category_path, d))])
            total_episodes += len(data_dirs)
            
            category_train_count = int(len(data_dirs) * self.train_ratio)
            
            import random
            random.shuffle(data_dirs)
            
            train_dirs = data_dirs[:category_train_count]
            test_dirs = data_dirs[category_train_count:]
            
            selected_dirs = train_dirs if is_train else test_dirs
            
            for data_dir in selected_dirs:
                data_dir_path = os.path.join(category_path, data_dir)
                
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
                        data = np.load(tactile_file)  # shape (T, C, H, W)
                        
                        if data.shape[1] == 6:  # Combined left-right data (T, 6, 20, 20)
                            total_frames += data.shape[0]
                            
                            if data.shape[0] > start_frame:
                                for t in range(start_frame, data.shape[0]):
                                    frame = data[t]  # (6, 20, 20)
                                    
                                    left_sensor = frame[0:3]   # Left sensor: channels 0,1,2
                                    right_sensor = frame[3:6]  # Right sensor: channels 3,4,5
                                    
                                    self.samples.append(left_sensor)
                                    self.samples.append(right_sensor)
                                    valid_frames += 1
                        
                        elif data.shape[1] == 3:  # Single hand data (T, 3, 20, 20)
                            total_frames += data.shape[0]
                            
                            if data.shape[0] > start_frame:
                                for t in range(start_frame, data.shape[0]):
                                    frame = data[t]  # (3, 20, 20)
                                    self.samples.append(frame)
                                    valid_frames += 1
                        
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to load {tactile_file}: {e}")
                    continue
            
            train_episodes += len(train_dirs)
            test_episodes += len(test_dirs)
                    
        if len(self.samples) == 0:
            raise ValueError(f"No valid data found! Check data path: {data_root}")
            
        self.samples = np.stack(self.samples)
        
        if not self.use_precomputed_normalization:
            if not self._load_normalization_params(categories):
                self._compute_normalization_params()
        
        self.samples = self._normalize_data(self.samples)
        
        print(f"[TactileForcesDataset] Dataset Statistics:")
        print(f"  - Data root: {data_root}")
        print(f"  - Categories: {categories}")
        print(f"  - Normalization: {self.normalize_method}")
        print(f"  - Total episodes: {total_episodes}")
        print(f"  - Train ratio: {self.train_ratio:.1%}")
        print(f"  - Random seed: {self.random_seed}")
        print(f"  - Train episodes: {train_episodes}")
        print(f"  - Test episodes: {test_episodes}")
        print(f"  - Current split: {'Train' if is_train else 'Test'}")
        print(f"  - Total frames: {total_frames}")
        print(f"  - Start frame: {start_frame}")
        print(f"  - Valid frames: {valid_frames}")
        print(f"  - Total samples: {len(self.samples)} (left+right sensors)")
        print(f"  - Sample shape: {self.samples.shape}")
        print(f"  - Data range: [{self.samples.min():.4f}, {self.samples.max():.4f}]")

    def _normalize_data(self, data):
        """
        Normalize data using precomputed normalization parameters.
        Args:
            data: Raw data (N, 3, 20, 20)
        Returns:
            Normalized data
        """
        if self.normalization_config['method'] is None:
            return data
        
        params = self.normalization_config['params']
        if params is None:
            return data
        
        method = self.normalization_config['method']
        normalized_data = np.zeros_like(data)
        
        if method == 'zscore':
            for i in range(data.shape[1]):
                channel_data = data[:, i, :, :]
                mean = params['channel_means'][i]
                std = params['channel_stds'][i]
                normalized_data[:, i, :, :] = (channel_data - mean) / (std + 1e-8)
        
        elif method == 'minmax':
            for i in range(data.shape[1]):
                channel_data = data[:, i, :, :]
                min_val = params['channel_mins'][i]
                max_val = params['channel_maxs'][i]
                normalized_data[:, i, :, :] = 2 * (channel_data - min_val) / (max_val - min_val + 1e-8) - 1
        
        elif method == 'channel_wise':
            for i in range(data.shape[1]):
                channel_data = data[:, i, :, :]
                min_val = params['channel_mins'][i]
                max_val = params['channel_maxs'][i]
                normalized_data[:, i, :, :] = 2 * (channel_data - min_val) / (max_val - min_val + 1e-8) - 1
        
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        return normalized_data

    def _compute_normalization_params(self):
        """Compute normalization parameters from training data."""
        print("🔄 Computing tactile normalization parameters...")
        
        method = self.normalization_config['method']
        if method is None:
            return
        
        data = self.samples  # (N, 3, 20, 20)
        params = {}
        
        if method in ['zscore']:
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
        
        cache_path = self._get_normalization_cache_path()
        with open(cache_path, 'w') as f:
            json.dump(self.normalization_config, f, indent=2)
        print(f"💾 Normalization params saved to: {cache_path}")

    def _get_normalization_cache_path(self):
        """Generate normalization parameter cache file path."""
        config_str = f"{self.data_root}_{self.normalize_method}_{self.train_ratio}_{self.random_seed}_{self.start_frame}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:16]
        cache_dir = os.path.join(self.data_root, ".tactile_normalization_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"tactile_norm_params_{config_hash}.json")

    def _load_normalization_params(self, categories):
        """Load saved normalization parameters."""
        cache_path = self._get_normalization_cache_path()
        
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self.normalization_config = json.load(f)
            print(f"📁 Loaded normalization params from: {cache_path}")
            return True
        return False

    def denormalize_data(self, normalized_data):
        """
        Denormalize data back to original scale.
        Args:
            normalized_data: Normalized data (N, 3, 20, 20) or (3, 20, 20)
        Returns:
            Denormalized original data
        """
        if self.normalization_config['method'] is None:
            return normalized_data
        
        params = self.normalization_config['params']
        if params is None:
            return normalized_data
        
        method = self.normalization_config['method']
        
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.cpu().numpy()
        
        original_shape = normalized_data.shape
        if len(original_shape) == 3:  # (3, 20, 20)
            normalized_data = normalized_data[np.newaxis, :]  # (1, 3, 20, 20)
        
        denormalized_data = np.zeros_like(normalized_data)
        
        if method == 'zscore':
            for i in range(normalized_data.shape[1]):
                channel_data = normalized_data[:, i, :, :]
                mean = params['channel_means'][i]
                std = params['channel_stds'][i]
                denormalized_data[:, i, :, :] = channel_data * std + mean
        
        elif method in ['minmax', 'channel_wise']:
            for i in range(normalized_data.shape[1]):
                channel_data = normalized_data[:, i, :, :]
                min_val = params['channel_mins'][i]
                max_val = params['channel_maxs'][i]
                denormalized_data[:, i, :, :] = (channel_data + 1) / 2 * (max_val - min_val) + min_val
        
        if len(original_shape) == 3:
            denormalized_data = denormalized_data[0]
        
        return denormalized_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]  # shape: (3, 20, 20)
        
        if self.augment and np.random.rand() > 0.5:
            sample = self._apply_augmentation(sample)
        
        sample = torch.FloatTensor(sample)
        
        return {
            'image': sample,
            'target': sample
        }

    def _apply_augmentation(self, sample):
        """Apply data augmentation."""
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, sample.shape)
            sample = sample + noise
        
        if np.random.rand() > 0.5:
            sample = np.flip(sample, axis=1)
        
        if np.random.rand() > 0.5:
            sample = np.flip(sample, axis=2)
        
        return sample.copy()


def create_train_test_tactile_datasets(data_root, categories=None, start_frame=0, 
                                     augment_train=True, augment_test=False, 
                                     normalize_method='zscore'):
    """
    Create train and test tactile datasets.
    
    Args:
        data_root: Data root directory path
        categories: List of categories to include
        start_frame: Starting frame index
        augment_train: Whether to augment training data
        augment_test: Whether to augment test data
        normalize_method: Normalization method ['zscore', 'minmax', 'channel_wise']
    
    Returns:
        tuple: (train_dataset, test_dataset, normalization_config)
    """
    train_dataset = TactileForcesDataset(
        data_root=data_root,
        categories=categories,
        start_frame=start_frame,
        is_train=True,
        augment=augment_train,
        normalize_method=normalize_method
    )
    
    test_dataset = TactileForcesDataset(
        data_root=data_root,
        categories=categories,
        start_frame=start_frame,
        is_train=False,
        augment=augment_test,
        normalize_method=normalize_method,
        normalization_config=train_dataset.normalization_config
    )
    
    print(f"📊 Tactile normalization info:")
    print(f"   Method: {train_dataset.normalization_config.get('method', 'None')}")
    print(f"   Test uses precomputed params: {test_dataset.use_precomputed_normalization}")
    
    return train_dataset, test_dataset, train_dataset.normalization_config


class TactileSampleDataset(Dataset):
    """
    Legacy tactile sample dataset (for backward compatibility).
    """
    def __init__(self, root_dirs, start_frame=20):
        """
        Args:
            root_dirs: String or list of strings for data directory paths
            start_frame: Starting frame index for data slicing
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
                
                if data.shape[0] > start_frame:
                    for t in range(start_frame, data.shape[0]):
                        frame = data[t, -1]
                        self.samples.append(frame[0:3])
                        self.samples.append(frame[3:6])
                        valid_frames += 1
                    
        self.samples = np.stack(self.samples)
        self.samples[:, 0:2] /= 0.05
        
        total_episodes = sum(len(glob(os.path.join(root, "episode_*"))) 
                           for root in root_dirs)
        print(f"[TactileSampleDataset] Dataset Statistics:")
        print(f"  - Environments: {len(root_dirs)}")
        print(f"  - Episodes: {total_episodes}")
        print(f"  - Total frames: {total_frames}")
        print(f"  - Start frame: {start_frame}")
        print(f"  - Valid frames: {valid_frames}")
        print(f"  - Total samples: {len(self.samples)} (left+right sensors)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32)
