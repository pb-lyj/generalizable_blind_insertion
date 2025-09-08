import numpy as np
import torch

class EarlyStopping:
    """提前停止训练，避免过拟合
    Args:
        patience (int): 容忍多少个 epoch 没有提升（默认 10）
        min_delta (float): 最小改善幅度，低于这个视为没有提升（默认 0.0）
        mode (str): 'min'（指标越小越好）或 'max'（指标越大越好）
        verbose (bool): 是否打印提示
    """
    def __init__(self, patience=10, min_delta=0.0, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False  # 不停止

        improvement = (
            (self.mode == 'min' and self.best_score - current_score > self.min_delta) or
            (self.mode == 'max' and current_score - self.best_score > self.min_delta)
        )

        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ EarlyStopping patience {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

# 使用示例
# # 初始化
# early_stopper = EarlyStopping(patience=10, min_delta=1e-4, mode='min')

# for epoch in range(num_epochs):
#     train_one_epoch(...)   # 训练
#     val_loss, val_l1 = evaluate(...)   # 验证

#     print(f"Epoch {epoch}: val L1 = {val_l1:.4f} mm")

#     # 用验证集 L1 作为监控指标（越小越好）
#     if early_stopper.step(val_l1):
#         print(f"⚠️ 提前停止训练，在 epoch {epoch} 提前结束。")
#         break
