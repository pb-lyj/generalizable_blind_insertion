from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, BaseImageDataset
from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat


class TactileLowdimDataset(BaseLowdimDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        state_key="state",
        action_key="action",
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[state_key, action_key]
        )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after


class TactileArrayDataset(BaseImageDataset):
    """
    用于触觉力阵列数据的数据集类（多模态观测版本）

    数据格式：
    - forces_l: (T, 3, 20, 20) - 左手触觉力阵列
    - forces_r: (T, 3, 20, 20) - 右手触觉力阵列
    - action: (T, 3) - 末端执行器位置

    返回格式（类似图像数据集）：
    - obs: dict with keys ['forces_l', 'forces_r']
    - action: (T, 3)
    """

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        forces_l_key="forces_l",
        forces_r_key="forces_r",
        action_key="action",
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        n_obs_steps=None,
    ):
        super().__init__()

        # 加载所有需要的键
        keys = [forces_l_key, forces_r_key, action_key]
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )

        # 如果指定了 n_obs_steps，只取前 k 个观测
        key_first_k = dict()
        if n_obs_steps is not None:
            key_first_k[forces_l_key] = n_obs_steps
            key_first_k[forces_r_key] = n_obs_steps

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.forces_l_key = forces_l_key
        self.forces_r_key = forces_r_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """创建 normalizer，为每个触觉输入和动作分别归一化"""
        normalizer = LinearNormalizer()

        # action normalizer
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer[self.action_key]
        )

        # tactile normalizers - 使用相同的统计信息
        # 触觉力的范围通常在 [-max_force, max_force]
        normalizer[self.forces_l_key] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer[self.forces_l_key]
        )
        normalizer[self.forces_r_key] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer[self.forces_r_key]
        )

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回格式:
        {
            'obs': {
                'forces_l': (T, 3, 20, 20),
                'forces_r': (T, 3, 20, 20)
            },
            'action': (T, 3)
        }
        """
        sample = self.sampler.sample_sequence(idx)

        # 只取前 n_obs_steps 的观测（如果指定）
        T_slice = slice(self.n_obs_steps)

        # 构建观测字典 - 保持原始 4D 形状用于 CNN
        obs_dict = {
            self.forces_l_key: sample[self.forces_l_key][T_slice].astype(np.float32),
            self.forces_r_key: sample[self.forces_r_key][T_slice].astype(np.float32),
        }

        action = sample[self.action_key].astype(np.float32)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }
        return torch_data
