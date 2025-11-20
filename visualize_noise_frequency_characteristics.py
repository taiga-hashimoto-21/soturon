"""
PSDデータに3種類のノイズを付与したグラフを可視化
周波数特性版のノイズを使用
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ノイズ関数をインポート（括弧を含むフォルダ名のため、直接パスを指定）
noise_module_path = os.path.join(project_root, 'ノイズの付与(共通)')
sys.path.insert(0, noise_module_path)
from add_noise import add_noise_to_interval

# 日本語フォントの設定
import platform
if platform.system() == 'Darwin':  # Mac
    plt.rcParams['font.family'] = 'Hiragino Sans'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# パラメータ設定
NUM_INTERVALS = 30
NOISE_LEVEL = 1.0  # ノイズレベルを上げて見やすくする
NOISE_INTERVAL = 17  # 区間17（約1700-1800ポイント付近）にノイズを付与
FREQ_MIN = 0.0
FREQ_MAX = 1000.0  # Hz（仮定）

# データの読み込み
print("データを読み込み中...")
with open('data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)

x = data['x']  # PSDデータ (32000, 1, 3000)

# 1つのサンプルを選ぶ（最初のサンプル）
sample_idx = 0
original_psd = x[sample_idx, 0, :]  # (3000,)

# Tensorに変換
if isinstance(original_psd, np.ndarray):
    original_psd = torch.from_numpy(original_psd).float()

print(f"サンプル {sample_idx} を使用")
print(f"PSDデータ形状: {original_psd.shape}")
print(f"PSDデータ範囲: {original_psd.min():.2e} ～ {original_psd.max():.2e}")

# ノイズ付与領域の計算
points_per_interval = len(original_psd) // NUM_INTERVALS
noise_start_idx = NOISE_INTERVAL * points_per_interval
noise_end_idx = noise_start_idx + points_per_interval

print(f"\nノイズ付与領域: 区間 {NOISE_INTERVAL} (インデックス {noise_start_idx}-{noise_end_idx})")

# 3種類のノイズを付与
print("\nノイズを付与中...")

# パターン1: 周波数帯域集中ノイズ
psd_frequency_band, _, _ = add_noise_to_interval(
    original_psd.clone(),
    NOISE_INTERVAL,
    noise_type='frequency_band',
    noise_level=NOISE_LEVEL,
    num_intervals=NUM_INTERVALS,
    freq_min=FREQ_MIN,
    freq_max=FREQ_MAX
)

# パターン2: 局所スパイクノイズ
psd_localized_spike, _, _ = add_noise_to_interval(
    original_psd.clone(),
    NOISE_INTERVAL,
    noise_type='localized_spike',
    noise_level=NOISE_LEVEL,
    num_intervals=NUM_INTERVALS,
    freq_min=FREQ_MIN,
    freq_max=FREQ_MAX
)

# パターン3: 振幅依存ノイズ
psd_amplitude_dependent, _, _ = add_noise_to_interval(
    original_psd.clone(),
    NOISE_INTERVAL,
    noise_type='amplitude_dependent',
    noise_level=NOISE_LEVEL,
    num_intervals=NUM_INTERVALS,
    freq_min=FREQ_MIN,
    freq_max=FREQ_MAX
)

# NumPy配列に変換
original_psd_np = original_psd.numpy()
psd_frequency_band_np = psd_frequency_band.numpy()
psd_localized_spike_np = psd_localized_spike.numpy()
psd_amplitude_dependent_np = psd_amplitude_dependent.numpy()

# 周波数ポイント軸
freq_points = np.arange(len(original_psd_np))

# すべてのデータからY軸の範囲を計算
all_data = np.concatenate([
    original_psd_np,
    psd_frequency_band_np,
    psd_localized_spike_np,
    psd_amplitude_dependent_np
])
y_min = np.min(all_data[all_data > 0]) * 0.5  # 最小値の0.5倍（0以下の値を除外）
y_max = np.max(all_data) * 2.0  # 最大値の2倍

print(f"\nY軸の範囲: {y_min:.2e} ～ {y_max:.2e}")

# グラフを作成
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle('PSDデータに3種類のノイズを付与した比較（周波数特性版）', fontsize=16, fontweight='bold')

# 1. 元のPSDデータ（ノイズなし）
ax1 = axes[0]
ax1.plot(freq_points, original_psd_np, color='#2C3E50', linewidth=1.5, label='元のPSDデータ（ノイズなし）')
ax1.axvspan(noise_start_idx, noise_end_idx, alpha=0.2, color='lightgray', label='ノイズ付与領域')
ax1.set_ylabel('PSD値', fontsize=12)
ax1.set_title('元のPSDデータ（ノイズなし）', fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.set_ylim(y_min, y_max)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xlim(0, len(original_psd_np))
ax1.set_xlabel('周波数ポイント', fontsize=12)

# 2. パターン1: 周波数帯域集中ノイズ
ax2 = axes[1]
ax2.plot(freq_points, original_psd_np, color='#2C3E50', linewidth=1.0, alpha=0.5, linestyle='--', label='元のPSDデータ')
ax2.plot(freq_points, psd_frequency_band_np, color='#3498DB', linewidth=1.5, label='周波数帯域集中ノイズ付与後')
ax2.axvspan(noise_start_idx, noise_end_idx, alpha=0.2, color='red', label='ノイズ付与領域')
ax2.set_ylabel('PSD値', fontsize=12)
ax2.set_title('パターン1: 周波数帯域集中ノイズ（電源ノイズ、共振、クロストークを模擬）', fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.set_ylim(y_min, y_max)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', fontsize=10)
ax2.set_xlim(0, len(original_psd_np))
ax2.set_xlabel('周波数ポイント', fontsize=12)

# 3. パターン2: 局所スパイクノイズ
ax3 = axes[2]
ax3.plot(freq_points, original_psd_np, color='#2C3E50', linewidth=1.0, alpha=0.5, linestyle='--', label='元のPSDデータ')
ax3.plot(freq_points, psd_localized_spike_np, color='#2ECC71', linewidth=1.5, label='局所スパイクノイズ付与後')
ax3.axvspan(noise_start_idx, noise_end_idx, alpha=0.2, color='red', label='ノイズ付与領域')
ax3.set_ylabel('PSD値', fontsize=12)
ax3.set_title('パターン2: 局所スパイクノイズ（電磁干渉 (EMI)、静電気放電 (ESD)、接触不良を模擬）', fontsize=13, fontweight='bold')
ax3.set_yscale('log')
ax3.set_ylim(y_min, y_max)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='upper right', fontsize=10)
ax3.set_xlim(0, len(original_psd_np))
ax3.set_xlabel('周波数ポイント', fontsize=12)

# 4. パターン3: 振幅依存ノイズ
ax4 = axes[3]
ax4.plot(freq_points, original_psd_np, color='#2C3E50', linewidth=1.0, alpha=0.5, linestyle='--', label='元のPSDデータ')
ax4.plot(freq_points, psd_amplitude_dependent_np, color='#E74C3C', linewidth=1.5, label='振幅依存ノイズ付与後')
ax4.axvspan(noise_start_idx, noise_end_idx, alpha=0.2, color='red', label='ノイズ付与領域')
ax4.set_xlabel('周波数ポイント', fontsize=12)
ax4.set_ylabel('PSD値', fontsize=12)
ax4.set_title('パターン3: 振幅依存ノイズ（非線形増幅器の歪み、ADCの量子化ノイズ、飽和を模擬）', fontsize=13, fontweight='bold')
ax4.set_yscale('log')
ax4.set_ylim(y_min, y_max)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(loc='upper right', fontsize=10)
ax4.set_xlim(0, len(original_psd_np))

plt.tight_layout()

# 保存
output_path = 'noise_frequency_characteristics_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nグラフを保存しました: {output_path}")
print(f"保存先: {os.path.abspath(output_path)}")

# 表示（バックグラウンド実行の場合は表示されない可能性がある）
try:
    plt.show()
except:
    pass

print("\n完了！")

