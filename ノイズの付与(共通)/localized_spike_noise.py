"""
局所スパイクノイズの実装（周波数特性版）
特定の周波数に集中的に発生するスパイク状のノイズを周波数特性として理論的に定義
電磁干渉（EMI）、静電気放電（ESD）、接触不良などを模擬
"""

import torch
import numpy as np


def add_localized_spike_noise(psd_data, interval_idx, noise_level=0.1, num_intervals=30, spike_ratio=0.15,
                              freq_min=0.0, freq_max=1000.0):
    """
    パターン2: 局所スパイクノイズ（周波数特性版）
    特定の周波数に集中的に発生するスパイク状のノイズを周波数特性として理論的に定義
    電磁干渉（EMI）、静電気放電（ESD）、接触不良などを模擬
    
    Args:
        psd_data: PSDデータ (3000ポイント)
        interval_idx: ノイズを加える区間のインデックス
        noise_level: ノイズの強度（元の値に対する倍率）
        num_intervals: 区間数
        spike_ratio: スパイクが発生する周波数の割合（0.15 = 15%の周波数に大きなスパイク）
        freq_min: PSDデータの最小周波数（Hz）
        freq_max: PSDデータの最大周波数（Hz）
    
    Returns:
        ノイズを加えたPSDデータ, 開始インデックス, 終了インデックス
    """
    noisy_data = psd_data.clone()
    num_points = len(psd_data)
    points_per_interval = num_points // num_intervals
    
    start_idx = interval_idx * points_per_interval
    end_idx = start_idx + points_per_interval
    
    # その区間の平均値を取得
    interval_mean = psd_data[start_idx:end_idx].mean()
    
    # 周波数軸を生成（実際の周波数：Hz）
    frequencies = torch.linspace(freq_min, freq_max, num_points)
    f_interval = frequencies[start_idx:end_idx]  # 区間内の周波数軸
    
    # 区間の周波数範囲を計算
    f_start = f_interval[0].item()
    f_end = f_interval[-1].item()
    
    # スパイクが発生する周波数の数を計算
    num_spike_freqs = int(points_per_interval * spike_ratio)
    num_spike_freqs = max(1, num_spike_freqs)  # 最低1つの周波数はスパイク
    
    # スパイクが発生する周波数をランダムに選択（Hz単位）
    spike_freq_indices = torch.randperm(points_per_interval)[:num_spike_freqs]
    spike_frequencies = f_interval[spike_freq_indices]  # スパイクの周波数（Hz）
    
    # ノイズを初期化
    noise = torch.zeros(points_per_interval)
    
    # スパイクの周波数特性を定義: N(f) = A * exp(-(f - f_spike)² / (2σ²))
    # 非常に狭いガウシアン（デルタ関数の近似）
    # スパイクの帯域幅（Hz単位）：区間幅の1%程度
    spike_bandwidth = (f_end - f_start) * 0.01
    if spike_bandwidth < 1e-10:
        spike_bandwidth = (f_end - f_start) / 100.0
    sigma_spike = spike_bandwidth / (2 * np.sqrt(2 * np.log(2)))  # 3dB帯域幅からσを計算
    
    # 各スパイク周波数に対して周波数特性を生成
    for i, f_spike in enumerate(spike_frequencies):
        # スパイクの周波数特性: 狭いガウシアン
        spike_response = torch.exp(-0.5 * ((f_interval - f_spike.item()) / sigma_spike) ** 2)
        
        # スパイク強度（3-5倍の強度）
        spike_strength_multiplier = 3.0 + torch.rand(1).item() * 2.0  # 3.0-5.0の範囲
        spike_noise = torch.randn(1).item() * interval_mean * noise_level * spike_strength_multiplier
        
        # スパイクノイズを加える
        noise += spike_response * spike_noise
    
    # その他の周波数には小さなバックグラウンドノイズ（0.1倍の強度）
    background_noise = torch.randn(points_per_interval) * interval_mean * noise_level * 0.1
    noise += background_noise
    
    # ノイズを加える
    noisy_data[start_idx:end_idx] += noise
    
    return noisy_data, start_idx, end_idx

