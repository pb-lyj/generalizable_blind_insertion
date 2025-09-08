#!/usr/bin/env python3
"""
触觉数据集归一化功能测试脚本
"""

from tactile_dataset import create_train_test_tactile_datasets
import numpy as np

def test_tactile_normalization():
    """测试触觉数据集的归一化功能"""
    
    print("🧪 触觉数据集归一化功能全面测试")
    print("=" * 60)
    
    # 测试参数
    test_categories = ['cir_lar']  # 使用一个小类别快速测试
    methods = ['zscore', 'minmax', 'channel_wise']
    
    results = {}
    
    for method in methods:
        print(f"\n📊 测试归一化方法: {method}")
        print("-" * 40)
        
        # 创建数据集
        train_dataset, test_dataset, norm_config = create_train_test_tactile_datasets(
            data_root='data25.7_aligned',
            categories=test_categories,
            normalize_method=method,
            start_frame=0
        )
        
        # 获取样本
        sample = train_dataset[0]
        image = sample['image']
        
        # 测试反归一化
        denorm_image = train_dataset.denormalize_data(image)
        
        # 记录结果
        results[method] = {
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'sample_shape': tuple(image.shape),
            'normalized_range': (float(image.min()), float(image.max())),
            'denormalized_range': (float(denorm_image.min()), float(denorm_image.max())),
            'norm_config': norm_config
        }
        
        print(f"   ✅ 训练集样本数: {results[method]['train_samples']}")
        print(f"   ✅ 测试集样本数: {results[method]['test_samples']}")
        print(f"   ✅ 样本形状: {results[method]['sample_shape']}")
        print(f"   ✅ 归一化数据范围: [{results[method]['normalized_range'][0]:.4f}, {results[method]['normalized_range'][1]:.4f}]")
        print(f"   ✅ 反归一化数据范围: [{results[method]['denormalized_range'][0]:.4f}, {results[method]['denormalized_range'][1]:.4f}]")
        
        # 验证归一化范围
        if method == 'zscore':
            print(f"   📝 Z-score归一化：数据应该大致在[-3, 3]范围内")
        elif method in ['minmax', 'channel_wise']:
            print(f"   📝 Min-Max归一化：数据应该在[-1, 1]范围内")
            
    print("\n" + "=" * 60)
    print("🎯 测试总结:")
    print("=" * 60)
    
    for method, result in results.items():
        print(f"\n{method}:")
        print(f"  - 样本数: {result['train_samples']} (训练) + {result['test_samples']} (测试)")
        print(f"  - 归一化范围: [{result['normalized_range'][0]:.4f}, {result['normalized_range'][1]:.4f}]")
        
        # 验证归一化是否正确
        min_val, max_val = result['normalized_range']
        if method == 'zscore':
            if -4 <= min_val <= 4 and -4 <= max_val <= 4:
                print("  ✅ Z-score归一化范围正常")
            else:
                print("  ⚠️ Z-score归一化范围异常")
        elif method in ['minmax', 'channel_wise']:
            if -1.1 <= min_val <= 1.1 and -1.1 <= max_val <= 1.1:
                print("  ✅ Min-Max归一化范围正常")
            else:
                print("  ⚠️ Min-Max归一化范围异常")
    
    print("\n✅ 触觉数据集归一化功能测试完成！")
    print("📁 归一化参数已缓存到 data25.7_aligned/.tactile_normalization_cache/")
    
    return results

if __name__ == "__main__":
    test_results = test_tactile_normalization()
