"""Tactile low-dim runner.

当前触觉任务没有在线交互环境（env），训练是纯离线的。
原配置引用了 `BaseLowdimRunner` 却传入了大量不被其 __init__ 接受的参数（如 n_train 等），
导致 Hydra 实例化时报 `unexpected keyword argument` 错误。

此 Runner 只是一个占位实现：
 - 接收并记录所有多余参数（以便后续扩展）
 - 在 `run` 时返回空的度量 dict（或可选返回占位统计）
 - 保持接口兼容，使训练主循环无需改动即可继续运行

后续如果需要：
 - 可在 run 中对验证集重新取样，计算动作 MSE/MAE 等指标
 - 或接入真实/仿真在线评估环境
"""

from typing import Dict, Any
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class TactileLowdimRunner(BaseLowdimRunner):
    def __init__(self, output_dir, **kwargs):
        """保存所有传入参数，避免 Hydra 报错。

        Parameters
        ----------
        output_dir : str
            工作目录（用于后续如需写可视化 / 结果文件）
        **kwargs : Any
            来自配置中的其余键（n_train, n_test 等），当前不使用，仅保存。
        """
        super().__init__(output_dir)
        self.extra_cfg: Dict[str, Any] = dict(kwargs)

    def run(self, policy: BaseLowdimPolicy) -> Dict:
        """占位评估函数。

        返回空字典即可；训练循环会仅依赖 train_loss 做 top-k。
        如需添加指标，可在此构造：
            metrics = {"tactile/placeholder": 0.0}
        并返回。
        """
        # 占位：未来可接入验证数据集计算动作误差 / 成功率等
        return {}
