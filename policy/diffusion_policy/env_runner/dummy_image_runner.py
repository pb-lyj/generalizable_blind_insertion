"""
Dummy Image Runner for offline training without real environment
用于离线训练的虚拟环境运行器（无真实环境）
"""

from typing import Dict
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class DummyImageRunner(BaseImageRunner):
    """
    虚拟环境运行器，用于不需要真实环境评估的离线训练场景
    例如：触觉数据集训练
    """

    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy: BaseImagePolicy) -> Dict:
        """
        返回空的运行结果，不执行真实环境评估
        """
        # 返回空字典或默认指标
        return {
            "test_mean_score": 0.0,
            "test_rollout_count": 0,
        }
