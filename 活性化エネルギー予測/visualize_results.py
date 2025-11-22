"""
活性化エネルギー予測の結果を可視化するスクリプト
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def visualize_activation_energy_comparison(results: dict, save_path: str = None):
    """
    ノイズ除去前後の活性化エネルギー予測精度を可視化
    
    Args:
        results: evaluate_activation_energy_predictionの結果
        save_path: 保存パス（Noneの場合は表示のみ）
    """
    errors_before = results['errors_before']
    errors_after = results['errors_after']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 誤差の分布（ヒストグラム）
    ax1 = axes[0]
    ax1.hist(errors_before, bins=50, alpha=0.7, label='ノイズあり', color='red', edgecolor='black')
    ax1.hist(errors_after, bins=50, alpha=0.7, label='ノイズ除去後', color='blue', edgecolor='black')
    ax1.set_xlabel('誤差 (meV)', fontsize=12)
    ax1.set_ylabel('サンプル数', fontsize=12)
    ax1.set_title('活性化エネルギー予測誤差の分布', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. 誤差の比較（散布図）
    ax2 = axes[1]
    ax2.scatter(errors_before, errors_after, alpha=0.5, s=10)
    
    # 対角線を描画（y = x）
    max_error = max(errors_before.max(), errors_after.max())
    ax2.plot([0, max_error], [0, max_error], 'r--', linewidth=2, label='y = x')
    
    ax2.set_xlabel('ノイズありの誤差 (meV)', fontsize=12)
    ax2.set_ylabel('ノイズ除去後の誤差 (meV)', fontsize=12)
    ax2.set_title('ノイズ除去前後の誤差比較', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"グラフを '{save_path}' に保存しました")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    from evaluate_activation_energy import evaluate_activation_energy_prediction
    
    # 評価を実行
    results = evaluate_activation_energy_prediction(
        model_path='自己教師あり学習/task4_output/best_val_model.pth',
        pickle_path='data_lowF_noise.pickle',
        device='cuda',
        num_samples=1000,  # 可視化用に1000サンプル
    )
    
    # 可視化
    visualize_activation_energy_comparison(
        results,
        save_path='活性化エネルギー予測/activation_energy_comparison.png'
    )

