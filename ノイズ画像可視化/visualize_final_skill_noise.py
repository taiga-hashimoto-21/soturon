"""
採用した3種類のノイズを可視化
1. 特定周波数干渉ノイズ（3000 Hz）
2. クロック漏れノイズ（5000 Hz）
3. 電源ノイズ
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import interpolate

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 日本語フォントの設定
import platform
if platform.system() == 'Darwin':  # Mac
    plt.rcParams['font.family'] = 'Hiragino Sans'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# パラメータ設定
SAMPLING_RATE = 30000.0  # Hz
FREQ_MIN = 0.0
FREQ_MAX = SAMPLING_RATE / 2.0  # ナイキスト周波数（15000 Hz）

def psd_to_timeseries(psd_data, use_zero_phase=True):
    """PSDデータを時系列データに変換（IFFT）
    GitHubリポジトリ noise3 の実装をそのまま使用
    """
    # NumPy配列に変換
    if isinstance(psd_data, torch.Tensor):
        psd = psd_data.cpu().numpy()
    else:
        psd = np.array(psd_data)
    
    # PSDから振幅スペクトルを推定（リポジトリの実装そのまま）
    amplitude_spectrum = np.sqrt(psd * len(psd))
    
    # ランダムな位相を追加（リポジトリの実装そのまま）
    np.random.seed(42)
    random_phase = np.random.uniform(0, 2*np.pi, len(amplitude_spectrum))
    
    # 複素スペクトルを構築
    complex_spectrum = amplitude_spectrum * np.exp(1j * random_phase)
    
    # エルミート対称性を保証（リポジトリの実装そのまま）
    full_spectrum = np.zeros(len(psd) * 2, dtype=complex)
    full_spectrum[0] = complex_spectrum[0].real
    full_spectrum[1:len(psd)] = complex_spectrum[1:]
    full_spectrum[len(psd)+1:] = np.conj(complex_spectrum[-1:0:-1])
    
    # 逆フーリエ変換（リポジトリの実装そのまま、6000ポイントのまま）
    time_signal = np.fft.ifft(full_spectrum).real
    
    # リポジトリでは6000ポイントのまま使っているので、そのまま返す
    return torch.from_numpy(time_signal).float()

def timeseries_to_psd(time_series_data):
    """時系列データをPSDデータに変換（FFT）"""
    if isinstance(time_series_data, np.ndarray):
        time_series_data = torch.from_numpy(time_series_data).float()
    
    # FFTを計算
    fft_result = torch.fft.fft(time_series_data)
    
    # PSD = |FFT|²
    psd_full = torch.abs(fft_result) ** 2
    
    n = len(time_series_data)
    
    # ナイキスト周波数までのPSD（N/2+1ポイント）
    psd_nyquist = psd_full[:n // 2 + 1]
    
    # 元のPSDデータの長さに合わせる（3000ポイント）
    if len(psd_nyquist) == 3000:
        return psd_nyquist
    elif len(psd_nyquist) < 3000:
        # ナイキスト周波数までのデータを3000ポイントに補間
        freq_old = np.linspace(0, SAMPLING_RATE / 2, len(psd_nyquist))
        freq_new = np.linspace(0, SAMPLING_RATE / 2, 3000)
        
        # 対数スケールで補間
        psd_nyquist_log = torch.log(psd_nyquist.clamp(min=1e-30))
        f_interp = interpolate.interp1d(freq_old, psd_nyquist_log.numpy(), kind='linear', 
                                       fill_value=(psd_nyquist_log[0].item(), psd_nyquist_log[-1].item()), 
                                       bounds_error=False)
        psd_interp_log = f_interp(freq_new)
        psd_interp = torch.exp(torch.from_numpy(psd_interp_log).float())
        
        return psd_interp
    else:
        return psd_nyquist[:3000]

# ===== 時系列データにノイズを付与する関数 =====

def add_interference_noise_to_timeseries(time_series_data, amplitude):
    """
    時系列データに特定周波数の干渉ノイズを付与
    近接機器からの漏れ（例：3000Hzのモーター動作周波数）
    全時間にわたって特定周波数の正弦波が加わる
    """
    noisy_data = time_series_data.clone()
    
    dt = 1.0 / SAMPLING_RATE
    n = len(time_series_data)
    time_axis = torch.arange(n) * dt
    
    # 特定周波数（3000Hz）の正弦波を全時間にわたって追加
    interference_freq = 3000.0  # Hz
    interference_noise = amplitude * torch.sin(2 * np.pi * interference_freq * time_axis)
    
    noisy_data += interference_noise
    return noisy_data

def add_clock_leakage_noise_to_timeseries(time_series_data, amplitude):
    """
    時系列データにクロック漏れノイズを付与
    デジタル回路のクロック信号の漏れ（例：5000Hzのクロック周波数）
    全時間にわたってクロック周波数の信号が漏れる
    """
    noisy_data = time_series_data.clone()
    
    dt = 1.0 / SAMPLING_RATE
    n = len(time_series_data)
    time_axis = torch.arange(n) * dt
    
    # クロック周波数（5000Hz）の信号を全時間にわたって追加
    clock_freq = 5000.0  # Hz
    # クロック信号は方形波に近いが、ここでは正弦波で近似
    clock_noise = amplitude * torch.sin(2 * np.pi * clock_freq * time_axis)
    
    noisy_data += clock_noise
    return noisy_data

def add_cable_crosstalk_noise_to_timeseries(time_series_data, amplitude):
    """
    時系列データにケーブル配置ノイズ（クロストーク）を付与
    特定の周波数帯域に集中する干渉ノイズ
    """
    noisy_data = time_series_data.clone()
    
    dt = 1.0 / SAMPLING_RATE
    n = len(time_series_data)
    time_axis = torch.arange(n) * dt
    
    # 特定の周波数帯域（3-4kHz）に集中する干渉ノイズ
    # 近接するケーブルからの漏れを模擬
    center_freq = 3500.0  # Hz
    bandwidth = 1000.0  # Hz
    
    # 帯域内の複数の周波数成分
    np.random.seed(44)
    crosstalk_noise = torch.zeros(n)
    for i in range(5):
        freq = center_freq - bandwidth/2 + (bandwidth / 4) * i
        phase = np.random.uniform(0, 2*np.pi)
        amp = np.random.uniform(0.4, 0.8)
        crosstalk_noise += amp * amplitude * torch.sin(2 * np.pi * freq * time_axis + phase)
    
    noisy_data += crosstalk_noise
    return noisy_data

def add_power_supply_noise_to_timeseries(time_series_data, amplitude):
    """
    時系列データに電源ノイズを付与
    電源周波数とスイッチングノイズを含む
    """
    noisy_data = time_series_data.clone()
    
    dt = 1.0 / SAMPLING_RATE
    n = len(time_series_data)
    time_axis = torch.arange(n) * dt
    
    # 電源周波数（50Hz）とスイッチングノイズ（高周波）
    power_freq = 50.0  # Hz
    
    # 電源周波数とその高調波
    power_noise = (
        amplitude * 0.8 * torch.sin(2 * np.pi * power_freq * time_axis) +
        amplitude * 0.4 * torch.sin(2 * np.pi * power_freq * 2 * time_axis) +  # 100 Hz
        amplitude * 0.2 * torch.sin(2 * np.pi * power_freq * 3 * time_axis)   # 150 Hz
    )
    
    # スイッチングノイズ（高周波成分）
    switching_noise = amplitude * 0.6 * torch.sin(2 * np.pi * 2000.0 * time_axis)  # 2kHzのスイッチングノイズ
    
    noisy_data += power_noise + switching_noise
    return noisy_data

# ===== PSDデータに周波数特性としてノイズを付与する関数 =====

def add_interference_noise_to_psd(psd_data, amplitude_ratio=0.1):
    """
    PSDデータに特定周波数の干渉ノイズを周波数特性として付与
    特定周波数（3000Hz）にのみピーク
    """
    noisy_psd = psd_data.clone()
    
    frequencies = torch.linspace(FREQ_MIN, FREQ_MAX, len(psd_data))
    
    # 特定周波数（3000Hz）にのみピーク
    interference_freq = 3000.0  # Hz
    base_amplitude = psd_data.mean() * amplitude_ratio
    
    # 非常に狭い帯域幅でピークを追加
    sigma_freq = 10.0  # 10 Hzの帯域幅（狭いピーク）
    peak_response = base_amplitude * torch.exp(-0.5 * ((frequencies - interference_freq) / sigma_freq) ** 2)
    noisy_psd += peak_response
    
    return noisy_psd

def add_clock_leakage_noise_to_psd(psd_data, amplitude_ratio=0.1):
    """
    PSDデータにクロック漏れノイズを周波数特性として付与
    クロック周波数（5000Hz）にのみピーク
    """
    noisy_psd = psd_data.clone()
    
    frequencies = torch.linspace(FREQ_MIN, FREQ_MAX, len(psd_data))
    
    # クロック周波数（5000Hz）にのみピーク
    clock_freq = 5000.0  # Hz
    base_amplitude = psd_data.mean() * amplitude_ratio
    
    # 非常に狭い帯域幅でピークを追加
    sigma_freq = 10.0  # 10 Hzの帯域幅（狭いピーク）
    peak_response = base_amplitude * torch.exp(-0.5 * ((frequencies - clock_freq) / sigma_freq) ** 2)
    noisy_psd += peak_response
    
    return noisy_psd

def add_cable_crosstalk_noise_to_psd(psd_data, amplitude_ratio=0.1):
    """
    PSDデータにケーブル配置ノイズ（クロストーク）を周波数特性として付与
    特定の周波数帯域（3-4kHz）に集中
    """
    noisy_psd = psd_data.clone()
    
    frequencies = torch.linspace(FREQ_MIN, FREQ_MAX, len(psd_data))
    
    # 特定の周波数帯域（3-4kHz）に集中
    center_freq = 3500.0  # Hz
    bandwidth = 1000.0  # Hz
    
    base_amplitude = psd_data.mean() * amplitude_ratio
    
    # 帯域内の複数の周波数にピーク
    np.random.seed(44)
    for i in range(5):
        freq = center_freq - bandwidth/2 + (bandwidth / 4) * i
        amp = np.random.uniform(0.4, 0.8)
        sigma_freq = 150.0  # 150 Hzの帯域幅
        peak_response = amp * base_amplitude * torch.exp(-0.5 * ((frequencies - freq) / sigma_freq) ** 2)
        noisy_psd += peak_response
    
    return noisy_psd

def add_power_supply_noise_to_psd(psd_data, amplitude_ratio=0.1):
    """
    PSDデータに電源ノイズを周波数特性として付与
    電源周波数とスイッチングノイズにピーク
    """
    noisy_psd = psd_data.clone()
    
    frequencies = torch.linspace(FREQ_MIN, FREQ_MAX, len(psd_data))
    
    base_amplitude = psd_data.mean() * amplitude_ratio
    
    # 電源周波数（50Hz）とその高調波
    power_freq = 50.0
    harmonics = [1, 2, 3]  # 50Hz, 100Hz, 150Hz
    amplitudes = [0.8, 0.4, 0.2]
    
    for harmonic, amp in zip(harmonics, amplitudes):
        freq = power_freq * harmonic
        sigma_freq = 10.0  # 10 Hzの帯域幅
        peak_response = amp * base_amplitude * torch.exp(-0.5 * ((frequencies - freq) / sigma_freq) ** 2)
        noisy_psd += peak_response
    
    # スイッチングノイズ（2kHz）
    switching_freq = 2000.0
    sigma_freq = 100.0  # 100 Hzの帯域幅
    peak_response = 0.6 * base_amplitude * torch.exp(-0.5 * ((frequencies - switching_freq) / sigma_freq) ** 2)
    noisy_psd += peak_response
    
    return noisy_psd

# データの読み込み
print("PSDデータを読み込み中...")
# プロジェクトルートからデータファイルを読み込む
data_path = os.path.join(project_root, '..', 'data_lowF_noise.pickle')
if not os.path.exists(data_path):
    data_path = os.path.join(os.path.dirname(project_root), 'data_lowF_noise.pickle')
with open(data_path, 'rb') as f:
    data_psd = pickle.load(f)

sample_idx = 0
original_psd = data_psd['x'][sample_idx, 0, :]  # PSDデータ

if isinstance(original_psd, np.ndarray):
    original_psd = torch.from_numpy(original_psd).float()
else:
    original_psd = original_psd.float()

print(f"PSDデータ形状: {original_psd.shape}")

# PSDから時系列データに変換
print("\nPSDから時系列データに変換中...")
original_timeseries = psd_to_timeseries(original_psd, use_zero_phase=False)

print(f"時系列データ形状: {original_timeseries.shape}")

# 時間軸と周波数軸
dt = 1.0 / SAMPLING_RATE
time_axis = np.arange(len(original_timeseries)) * dt
frequencies = torch.linspace(FREQ_MIN, FREQ_MAX, len(original_psd))

print(f"\n時系列データの長さ: {len(original_timeseries)}")
print(f"時間範囲: 0 ～ {time_axis[-1]:.3f} 秒")
print(f"周波数範囲: {FREQ_MIN:.1f} ～ {FREQ_MAX:.1f} Hz")

# ===== ケース1: 時系列データにノイズを付与 =====
print("\n" + "=" * 60)
print("ケース1: 時系列データにノイズを付与 → PSDに変換")
print("=" * 60)

noise_amplitude_ts = original_timeseries.std() * 1.5  # ノイズレベルを上げる

# パターン1: 特定周波数の干渉ノイズ
timeseries_interference = add_interference_noise_to_timeseries(original_timeseries, noise_amplitude_ts)
psd_from_interference = timeseries_to_psd(timeseries_interference)

# パターン2: クロック漏れノイズ
timeseries_clock = add_clock_leakage_noise_to_timeseries(original_timeseries, noise_amplitude_ts)
psd_from_clock = timeseries_to_psd(timeseries_clock)

# パターン3: 電源ノイズ
timeseries_power = add_power_supply_noise_to_timeseries(original_timeseries, noise_amplitude_ts)
psd_from_power = timeseries_to_psd(timeseries_power)

# ===== ケース2: PSDデータに周波数特性としてノイズを付与 =====
print("\n" + "=" * 60)
print("ケース2: PSDデータに周波数特性としてノイズを付与 → 時系列に変換")
print("=" * 60)

# パターン1: 特定周波数の干渉ノイズ
psd_interference = add_interference_noise_to_psd(original_psd, amplitude_ratio=2.0)  # ノイズレベルを上げる
timeseries_from_interference = psd_to_timeseries(psd_interference, use_zero_phase=False)

# パターン2: クロック漏れノイズ
psd_clock = add_clock_leakage_noise_to_psd(original_psd, amplitude_ratio=2.0)  # ノイズレベルを上げる
timeseries_from_clock = psd_to_timeseries(psd_clock, use_zero_phase=False)

# パターン3: 電源ノイズ
psd_power = add_power_supply_noise_to_psd(original_psd, amplitude_ratio=2.0)  # ノイズレベルを上げる
timeseries_from_power = psd_to_timeseries(psd_power, use_zero_phase=False)

# NumPy配列に変換
original_psd_np = original_psd.numpy()
original_timeseries_np = original_timeseries.numpy()

print(f"\n時系列データ範囲: {original_timeseries_np.min():.2e} ～ {original_timeseries_np.max():.2e}")
print(f"PSDデータ範囲: {original_psd_np.min():.2e} ～ {original_psd_np.max():.2e}")

# グラフを作成（1列4行：1行目は元のデータ、2-4行目は各ノイズ）
fig, axes = plt.subplots(4, 1, figsize=(18, 16))

noise_types = [
    ("電源ノイズ", "#E67E22"),
    ("特定周波数干渉ノイズ", "#E74C3C"),
    ("クロック漏れノイズ", "#3498DB")
]

# ケース2のデータ（PSDデータに周波数特性としてノイズを付与）
case2_psd_data = [
    psd_power.numpy(),
    psd_interference.numpy(), 
    psd_clock.numpy()
]
case2_timeseries_data = [
    timeseries_from_power.numpy(),
    timeseries_from_interference.numpy(), 
    timeseries_from_clock.numpy()
]

# 元のデータの色（灰色）
original_color = '#808080'

# 1行目: 元のPSDデータ
ax_original_psd = axes[0]
ax_original_psd.plot(frequencies.numpy(), original_psd_np, color=original_color, marker='.', markersize=1.5, linestyle='None')
ax_original_psd.set_xlabel('周波数 (Hz)', fontsize=11)
ax_original_psd.set_ylabel('PSD値', fontsize=11)
ax_original_psd.set_title('元のPSDデータ', fontsize=12, fontweight='bold')
ax_original_psd.set_yscale('log')
ax_original_psd.grid(True, alpha=0.3)
ax_original_psd.set_xlim(frequencies[0].item(), frequencies[-1].item())

# 2-4行目: 各ノイズのPSD
for i, (noise_name, color) in enumerate(noise_types):
    row_idx = i + 1  # 1行目は元のデータなので、2行目から開始
    
    # PSDデータに周波数特性としてノイズを付与した場合
    ax_psd = axes[row_idx]
    ax_psd.plot(frequencies.numpy(), original_psd_np, color=original_color, marker='.', markersize=1.0, alpha=0.5, linestyle='None', label='元のPSDデータ')
    ax_psd.plot(frequencies.numpy(), case2_psd_data[i], color=color, marker='.', markersize=1.5, linestyle='None', label='ノイズ付与後')
    ax_psd.set_xlabel('周波数 (Hz)', fontsize=11)
    ax_psd.set_ylabel('PSD値', fontsize=11)
    ax_psd.set_title(f'{noise_name}', fontsize=12, fontweight='bold')
    ax_psd.set_yscale('log')
    ax_psd.grid(True, alpha=0.3)
    ax_psd.legend(fontsize=9)
    ax_psd.set_xlim(frequencies[0].item(), frequencies[-1].item())

plt.tight_layout()

# 保存
output_path = 'final_skill_noise_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nグラフを保存しました: {output_path}")
print(f"保存先: {os.path.abspath(output_path)}")

try:
    plt.show()
except:
    pass

print("\n完了！")

