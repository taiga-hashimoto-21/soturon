"""
ノイズ付与後のPSDデータを逆フーリエ変換で時系列データに変換し、
ノイズが乗った場所を可視化
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

# ノイズ関数をインポート
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
NOISE_LEVEL = 1.0
NOISE_INTERVAL = 17  # 区間17（約1700-1800ポイント付近）にノイズを付与
SAMPLING_RATE = 30000.0  # Hz（サンプリング周波数）
FREQ_MIN = 0.0
FREQ_MAX = SAMPLING_RATE / 2.0  # ナイキスト周波数（15000 Hz）

def psd_to_timeseries(psd_data, use_zero_phase=True):
    """
    PSDデータを時系列データに変換（IFFT、位相0）
    
    Args:
        psd_data: PSDデータ (3000ポイント)
        use_zero_phase: 位相0を使用するか
    
    Returns:
        時系列データ
    """
    if isinstance(psd_data, np.ndarray):
        psd_data = torch.from_numpy(psd_data).float()
    
    seq_len = len(psd_data)
    
    # PSDは実数なので、複素スペクトルを構築
    # PSD = |FFT(time_series)|² なので、|FFT(time_series)| = sqrt(PSD)
    magnitude = torch.sqrt(psd_data.clamp(min=1e-30))  # 負の値や0を防ぐ
    
    # 位相情報を仮定（位相0）
    if use_zero_phase:
        phase = torch.zeros_like(magnitude)
    else:
        # ランダム位相（再現性のためシード固定）
        rng = np.random.default_rng(seed=42)
        phase = torch.from_numpy(rng.uniform(0, 2 * np.pi, seq_len)).float()
    
    # 複素数の周波数スペクトルを構築
    complex_spectrum = magnitude * torch.exp(1j * phase)
    
    # IFFTで時系列データに変換
    time_series = torch.fft.ifft(complex_spectrum).real
    
    return time_series

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

# 時間軸の計算
dt = 1.0 / SAMPLING_RATE  # サンプリング間隔（秒）
time_axis = np.arange(len(original_psd)) * dt  # 時間軸（秒）

print(f"\nノイズ付与領域: 区間 {NOISE_INTERVAL} (インデックス {noise_start_idx}-{noise_end_idx})")
print(f"サンプリングレート: {SAMPLING_RATE:.1f} Hz")
print(f"データポイント数: {len(original_psd)}")
print(f"時間範囲: 0 ～ {time_axis[-1]:.3f} 秒")
print(f"周波数範囲: {FREQ_MIN:.1f} ～ {FREQ_MAX:.1f} Hz（ナイキスト周波数）")

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

# PSDデータを時系列データに変換
print("\nPSDデータを時系列データに変換中（IFFT、位相0）...")

original_timeseries = psd_to_timeseries(original_psd, use_zero_phase=True)
timeseries_frequency_band = psd_to_timeseries(psd_frequency_band, use_zero_phase=True)
timeseries_localized_spike = psd_to_timeseries(psd_localized_spike, use_zero_phase=True)
timeseries_amplitude_dependent = psd_to_timeseries(psd_amplitude_dependent, use_zero_phase=True)

# NumPy配列に変換
original_timeseries_np = original_timeseries.numpy()
timeseries_frequency_band_np = timeseries_frequency_band.numpy()
timeseries_localized_spike_np = timeseries_localized_spike.numpy()
timeseries_amplitude_dependent_np = timeseries_amplitude_dependent.numpy()

# ノイズの差分を計算
noise_frequency_band_diff = timeseries_frequency_band_np - original_timeseries_np
noise_localized_spike_diff = timeseries_localized_spike_np - original_timeseries_np
noise_amplitude_dependent_diff = timeseries_amplitude_dependent_np - original_timeseries_np

# ノイズが乗った時間範囲を計算
noise_start_time = time_axis[noise_start_idx]
noise_end_time = time_axis[noise_end_idx - 1]

print(f"ノイズ付与時間範囲: {noise_start_time:.3f} ～ {noise_end_time:.3f} 秒")

# グラフを作成
fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.suptitle('ノイズ付与後のPSDデータを時系列データに変換した結果', fontsize=16, fontweight='bold')

# 1. パターン1: 周波数帯域集中ノイズ
ax1 = axes[0, 0]
ax1.plot(time_axis, original_timeseries_np, color='#2C3E50', linewidth=1.0, alpha=0.5, linestyle='--', label='元の時系列データ')
ax1.plot(time_axis, timeseries_frequency_band_np, color='#3498DB', linewidth=1.5, label='周波数帯域集中ノイズ付与後')
ax1.axvspan(noise_start_time, noise_end_time, alpha=0.2, color='red', label='ノイズ付与領域')
ax1.set_ylabel('振幅', fontsize=12)
ax1.set_title('パターン1: 周波数帯域集中ノイズ（時系列データ）', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', fontsize=9)
ax1.set_xlim(time_axis[0], time_axis[-1])

ax1_diff = axes[0, 1]
ax1_diff.plot(time_axis, noise_frequency_band_diff, color='#3498DB', linewidth=1.5, label='ノイズの差分')
ax1_diff.axvspan(noise_start_time, noise_end_time, alpha=0.2, color='red', label='ノイズ付与領域')
ax1_diff.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax1_diff.set_ylabel('ノイズの差分', fontsize=12)
ax1_diff.set_title('パターン1: ノイズの差分（時系列データ）', fontsize=13, fontweight='bold')
ax1_diff.grid(True, alpha=0.3, linestyle='--')
ax1_diff.legend(loc='upper right', fontsize=9)
ax1_diff.set_xlim(time_axis[0], time_axis[-1])

# 2. パターン2: 局所スパイクノイズ
ax2 = axes[1, 0]
ax2.plot(time_axis, original_timeseries_np, color='#2C3E50', linewidth=1.0, alpha=0.5, linestyle='--', label='元の時系列データ')
ax2.plot(time_axis, timeseries_localized_spike_np, color='#2ECC71', linewidth=1.5, label='局所スパイクノイズ付与後')
ax2.axvspan(noise_start_time, noise_end_time, alpha=0.2, color='red', label='ノイズ付与領域')
ax2.set_ylabel('振幅', fontsize=12)
ax2.set_title('パターン2: 局所スパイクノイズ（時系列データ）', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', fontsize=9)
ax2.set_xlim(time_axis[0], time_axis[-1])

ax2_diff = axes[1, 1]
ax2_diff.plot(time_axis, noise_localized_spike_diff, color='#2ECC71', linewidth=1.5, label='ノイズの差分')
ax2_diff.axvspan(noise_start_time, noise_end_time, alpha=0.2, color='red', label='ノイズ付与領域')
ax2_diff.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax2_diff.set_ylabel('ノイズの差分', fontsize=12)
ax2_diff.set_title('パターン2: ノイズの差分（時系列データ）', fontsize=13, fontweight='bold')
ax2_diff.grid(True, alpha=0.3, linestyle='--')
ax2_diff.legend(loc='upper right', fontsize=9)
ax2_diff.set_xlim(time_axis[0], time_axis[-1])

# 3. パターン3: 振幅依存ノイズ
ax3 = axes[2, 0]
ax3.plot(time_axis, original_timeseries_np, color='#2C3E50', linewidth=1.0, alpha=0.5, linestyle='--', label='元の時系列データ')
ax3.plot(time_axis, timeseries_amplitude_dependent_np, color='#E74C3C', linewidth=1.5, label='振幅依存ノイズ付与後')
ax3.axvspan(noise_start_time, noise_end_time, alpha=0.2, color='red', label='ノイズ付与領域')
ax3.set_ylabel('振幅', fontsize=12)
ax3.set_title('パターン3: 振幅依存ノイズ（時系列データ）', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='upper right', fontsize=9)
ax3.set_xlim(time_axis[0], time_axis[-1])

ax3_diff = axes[2, 1]
ax3_diff.plot(time_axis, noise_amplitude_dependent_diff, color='#E74C3C', linewidth=1.5, label='ノイズの差分')
ax3_diff.axvspan(noise_start_time, noise_end_time, alpha=0.2, color='red', label='ノイズ付与領域')
ax3_diff.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax3_diff.set_ylabel('ノイズの差分', fontsize=12)
ax3_diff.set_title('パターン3: ノイズの差分（時系列データ）', fontsize=13, fontweight='bold')
ax3_diff.grid(True, alpha=0.3, linestyle='--')
ax3_diff.legend(loc='upper right', fontsize=9)
ax3_diff.set_xlim(time_axis[0], time_axis[-1])

# 4. 全体の比較（拡大表示）
# ノイズ領域を中心に拡大表示
zoom_start_idx = max(0, noise_start_idx - 200)
zoom_end_idx = min(len(time_axis), noise_end_idx + 200)
zoom_time_axis = time_axis[zoom_start_idx:zoom_end_idx]

ax4 = axes[3, 0]
ax4.plot(zoom_time_axis, original_timeseries_np[zoom_start_idx:zoom_end_idx], 
         color='#2C3E50', linewidth=1.0, alpha=0.5, linestyle='--', label='元の時系列データ')
ax4.plot(zoom_time_axis, timeseries_frequency_band_np[zoom_start_idx:zoom_end_idx], 
         color='#3498DB', linewidth=1.5, alpha=0.7, label='周波数帯域集中')
ax4.plot(zoom_time_axis, timeseries_localized_spike_np[zoom_start_idx:zoom_end_idx], 
         color='#2ECC71', linewidth=1.5, alpha=0.7, label='局所スパイク')
ax4.plot(zoom_time_axis, timeseries_amplitude_dependent_np[zoom_start_idx:zoom_end_idx], 
         color='#E74C3C', linewidth=1.5, alpha=0.7, label='振幅依存')
ax4.axvspan(noise_start_time, noise_end_time, alpha=0.2, color='red', label='ノイズ付与領域')
ax4.set_xlabel('時間 (秒)', fontsize=12)
ax4.set_ylabel('振幅', fontsize=12)
ax4.set_title('3種類のノイズの比較（拡大表示）', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(loc='upper right', fontsize=9)

ax4_diff = axes[3, 1]
ax4_diff.plot(zoom_time_axis, noise_frequency_band_diff[zoom_start_idx:zoom_end_idx], 
              color='#3498DB', linewidth=1.5, alpha=0.7, label='周波数帯域集中')
ax4_diff.plot(zoom_time_axis, noise_localized_spike_diff[zoom_start_idx:zoom_end_idx], 
              color='#2ECC71', linewidth=1.5, alpha=0.7, label='局所スパイク')
ax4_diff.plot(zoom_time_axis, noise_amplitude_dependent_diff[zoom_start_idx:zoom_end_idx], 
              color='#E74C3C', linewidth=1.5, alpha=0.7, label='振幅依存')
ax4_diff.axvspan(noise_start_time, noise_end_time, alpha=0.2, color='red', label='ノイズ付与領域')
ax4_diff.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax4_diff.set_xlabel('時間 (秒)', fontsize=12)
ax4_diff.set_ylabel('ノイズの差分', fontsize=12)
ax4_diff.set_title('3種類のノイズの差分比較（拡大表示）', fontsize=13, fontweight='bold')
ax4_diff.grid(True, alpha=0.3, linestyle='--')
ax4_diff.legend(loc='upper right', fontsize=9)

plt.tight_layout()

# 保存
output_path = 'noise_in_timeseries_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nグラフを保存しました: {output_path}")
print(f"保存先: {os.path.abspath(output_path)}")

# 表示（バックグラウンド実行の場合は表示されない可能性がある）
try:
    plt.show()
except:
    pass

print("\n完了！")

