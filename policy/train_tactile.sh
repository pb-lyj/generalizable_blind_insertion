#!/bin/bash

# 触觉Peg-in-Hole任务训练脚本
# 使用 Diffusion Policy 训练触觉力阵列数据

echo "================================================"
echo "Training Tactile Peg-in-Hole Policy"
echo "================================================"
echo ""
echo "配置信息："
echo "  任务: tactile_peg_in_hole"
echo "  策略: DiffusionUnetTactilePolicy"
echo "  编码器: MultiTactileObsEncoder"
echo "  数据: data/tactile_data.zarr"
echo ""
echo "================================================"
echo ""

# 检查数据是否存在
if [ ! -d "data/tactile_data.zarr" ]; then
    echo "错误: 数据文件 data/tactile_data.zarr 不存在！"
    echo "请先运行转换脚本: python convert_hdf5_to_zarr.py"
    exit 1
fi

# 运行训练
python train.py \
    --config-name=train_diffusion_unet_tactile_workspace \
    training.seed=42 \
    training.device=cuda:0 \
    training.num_epochs=1000 \
    dataloader.batch_size=64 \
    exp_name=tactile_baseline

echo ""
echo "================================================"
echo "训练完成！"
echo "结果保存在: data/outputs/"
echo "================================================"
