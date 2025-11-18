"""
PSDデータにノイズを付与したものとマスクを適用したものを可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams

# 日本語フォントの設定
import platform
if platform.system() == 'Darwin':  # macOS
    rcParams['font.family'] = 'Hiragino Sans'
elif platform.system() == 'Windows':
    rcParams['font.family'] = 'MS Gothic'
else:
    rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 12

# データの生成（サンプルPSDデータ）
np.random.seed(42)
L = 3000  # 3000ポイント
num_intervals = 30
points_per_interval = L // num_intervals

# 元のPSDデータ（減衰する特性を模擬）
x = np.linspace(0, 1, L)
original_psd = np.exp(-x * 5) * (1 + 0.1 * np.sin(x * 20)) + 0.01 * np.random.randn(L)
original_psd = np.maximum(original_psd, 0.01)  # 負の値を防ぐ

# ノイズ区間を選択（例: 区間10）
noise_interval = 10
noise_start = noise_interval * points_per_interval
noise_end = min(noise_start + points_per_interval, L)

# ノイズを付与
noisy_psd = original_psd.copy()
noise_level = 0.3
noise = noise_level * np.random.randn(points_per_interval)
noisy_psd[noise_start:noise_end] += noise

# マスク区間を選択（ノイズ区間以外から15%を選択）
available_intervals = [i for i in range(num_intervals) if i != noise_interval]
mask_ratio = 0.15
num_masked_intervals = max(1, int(len(available_intervals) * mask_ratio))
masked_interval_indices = np.random.choice(available_intervals, size=num_masked_intervals, replace=False)

# マスク位置を計算
mask_positions = np.zeros(L, dtype=bool)
for interval_idx in masked_interval_indices:
    start_idx = interval_idx * points_per_interval
    end_idx = min(start_idx + points_per_interval, L)
    mask_positions[start_idx:end_idx] = True

# マスクを適用
masked_psd = noisy_psd.copy()
masked_psd[mask_positions] = 0.0

# 図の作成
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
fig.suptitle('PSD Data: Noisy → Masked', fontsize=16, fontweight='bold')

# 1. 主タスク: マスクを適用したPSDデータ
ax1 = axes[0]
ax1.plot(noisy_psd, 'r-', linewidth=0.5, alpha=0.3, label='Noisy PSD', linestyle='--')
ax1.plot(masked_psd, 'g-', linewidth=0.5, alpha=0.7, label='Masked PSD')
ax1.set_ylabel('パワースペクトル密度', fontsize=12)
ax1.set_title('主タスク: マスク予測', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# マスク区間をハイライト
for interval_idx in masked_interval_indices:
    start_idx = interval_idx * points_per_interval
    end_idx = min(start_idx + points_per_interval, L)
    mask_rect = patches.Rectangle(
        (start_idx, ax1.get_ylim()[0]),
        end_idx - start_idx,
        ax1.get_ylim()[1] - ax1.get_ylim()[0],
        linewidth=2,
        edgecolor='green',
        facecolor='green',
        alpha=0.2
    )
    ax1.add_patch(mask_rect)
    # マスク区間のラベル
    ax1.text(start_idx + (end_idx - start_idx) / 2, ax1.get_ylim()[1] * 0.9,
             f'Mask\n{interval_idx}', 
             ha='center', va='top', fontsize=8, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

# ノイズ区間も表示（マスクされていないことを示す）
noise_rect1 = patches.Rectangle(
    (noise_start, ax1.get_ylim()[0]),
    points_per_interval,
    ax1.get_ylim()[1] - ax1.get_ylim()[0],
    linewidth=2,
    edgecolor='red',
    facecolor='red',
    alpha=0.1,
    linestyle='--'
)
ax1.add_patch(noise_rect1)
ax1.text(noise_start + points_per_interval / 2, ax1.get_ylim()[1] * 0.7,
         f'Noise\nInterval {noise_interval}', 
         ha='center', va='top', fontsize=8, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))

# 区間の境界線を描画
for i in range(0, num_intervals + 1):
    x_pos = i * points_per_interval
    if x_pos < L:
        ax1.axvline(x=x_pos, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# 2. 副タスク: ノイズを付与したPSDデータ
ax2 = axes[1]
ax2.plot(original_psd, 'b-', linewidth=0.5, alpha=0.3, label='Original PSD', linestyle='--')
ax2.plot(noisy_psd, 'r-', linewidth=0.5, alpha=0.7, label='Noisy PSD')
ax2.set_xlabel('周波数', fontsize=12)
ax2.set_ylabel('パワースペクトル密度', fontsize=12)
ax2.set_title('副タスク: ノイズ検知', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# ノイズ区間をハイライト
noise_rect2 = patches.Rectangle(
    (noise_start, ax2.get_ylim()[0]),
    points_per_interval,
    ax2.get_ylim()[1] - ax2.get_ylim()[0],
    linewidth=2,
    edgecolor='red',
    facecolor='red',
    alpha=0.2,
    label='Noise Interval'
)
ax2.add_patch(noise_rect2)

# 区間の境界線を描画
for i in range(0, num_intervals + 1):
    x_pos = i * points_per_interval
    if x_pos < L:
        ax2.axvline(x=x_pos, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# ノイズ区間のラベル
ax2.text(noise_start + points_per_interval / 2, ax2.get_ylim()[1] * 0.9,
         f'Noise\nInterval {noise_interval}', 
         ha='center', va='top', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))


plt.tight_layout()

# デスクトップに保存
output_path = '/Users/hashimototaiga/Desktop/psd_noise_mask_visualization.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"図を保存しました: {output_path}")

# 表示
plt.show()

# 統計情報を表示
print("\n=== 統計情報 ===")
print(f"総ポイント数: {L}")
print(f"区間数: {num_intervals}")
print(f"1区間あたりのポイント数: {points_per_interval}")
print(f"\nノイズ区間: {noise_interval} ({noise_start}〜{noise_end-1}ポイント)")
print(f"マスク区間数: {num_masked_intervals}")
print(f"マスク区間: {masked_interval_indices}")
print(f"マスクされたポイント数: {mask_positions.sum()} ({mask_positions.sum()/L*100:.1f}%)")

