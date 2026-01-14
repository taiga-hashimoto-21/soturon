"""
各ノイズを個別のファイルに分けて可視化
それぞれのグラフの縦の長さを1.5倍にする
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

# 時間軸と周波数軸
frequencies = torch.linspace(FREQ_MIN, FREQ_MAX, len(original_psd))

# NumPy配列に変換
original_psd_np = original_psd.numpy()

# ノイズの定義
noise_configs = [
    {
        'name': '電源ノイズ',
        'color': '#E67E22',
        'filename': 'power_noise.png',
        'function': add_power_supply_noise_to_psd
    },
    {
        'name': '特定周波数干渉ノイズ',
        'color': '#E74C3C',
        'filename': 'interference_noise.png',
        'function': add_interference_noise_to_psd
    },
    {
        'name': 'クロック漏れノイズ',
        'color': '#3498DB',
        'filename': 'clock_leakage_noise.png',
        'function': add_clock_leakage_noise_to_psd
    }
]

# 元のデータの色（灰色）
original_color = '#808080'

# 各ノイズごとに個別のグラフを作成
for noise_config in noise_configs:
    print(f"\n{noise_config['name']}のグラフを作成中...")
    
    # PSDデータに周波数特性としてノイズを付与
    psd_noisy = noise_config['function'](original_psd, amplitude_ratio=2.0)
    psd_noisy_np = psd_noisy.numpy()
    
    # グラフを作成（2列1行：左に元のデータ、右にノイズ付与後のデータ）
    # 縦の長さを1.5倍にする（元々4行で16だったので、1行で4、1.5倍で6）
    fig, axes = plt.subplots(1, 2, figsize=(30, 6))
    
    # 左列: 元のPSDデータ
    ax_original = axes[0]
    ax_original.plot(frequencies.numpy(), original_psd_np, color=original_color, marker='.', markersize=1.5, linestyle='None')
    ax_original.set_xlabel('周波数 (Hz)', fontsize=11)
    ax_original.set_ylabel('PSD値', fontsize=11)
    ax_original.set_title('元のPSDデータ', fontsize=12, fontweight='bold')
    ax_original.set_yscale('log')
    ax_original.grid(True, alpha=0.3)
    ax_original.set_xlim(frequencies[0].item(), frequencies[-1].item())
    
    # 右列: ノイズ付与後のデータ
    ax_noisy = axes[1]
    ax_noisy.plot(frequencies.numpy(), psd_noisy_np, color=noise_config['color'], marker='.', markersize=1.5, linestyle='None')
    ax_noisy.set_xlabel('周波数 (Hz)', fontsize=11)
    ax_noisy.set_ylabel('PSD値', fontsize=11)
    ax_noisy.set_title(f'{noise_config["name"]}', fontsize=12, fontweight='bold')
    ax_noisy.set_yscale('log')
    ax_noisy.grid(True, alpha=0.3)
    ax_noisy.set_xlim(frequencies[0].item(), frequencies[-1].item())
    
    plt.tight_layout()
    
    # 保存
    output_path = noise_config['filename']
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"グラフを保存しました: {output_path}")
    print(f"保存先: {os.path.abspath(output_path)}")
    
    plt.close()

print("\n全てのグラフの作成が完了しました！")

