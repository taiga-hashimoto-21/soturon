"""
電源ノイズの実装（周波数特性版）
電源周波数とスイッチングノイズを周波数特性として理論的に定義
"""

import torch
import numpy as np


def add_power_supply_noise(psd_data, interval_idx, noise_level=0.1, num_intervals=30,
                           freq_min=0.0, freq_max=15000.0):
    """
    電源ノイズ（周波数特性版）
    電源周波数（50 Hz）とその高調波、スイッチングノイズ（2 kHz）を周波数特性として理論的に定義
    
    Args:
        psd_data: PSDデータ (3000ポイント)
        interval_idx: ノイズを加える区間のインデックス（このノイズは全体に適用されるため無視される）
        noise_level: ノイズの強度（元の値に対する倍率）
        num_intervals: 区間数（使用しない）
        freq_min: PSDデータの最小周波数（Hz）
        freq_max: PSDデータの最大周波数（Hz）
    
    Returns:
        ノイズを加えたPSDデータ, 開始インデックス, 終了インデックス
    """
    noisy_data = psd_data.clone()
    num_points = len(psd_data)
    
    # 周波数軸を生成（実際の周波数：Hz）
    frequencies = torch.linspace(freq_min, freq_max, num_points)
    
    # ベース振幅を計算
    base_amplitude = psd_data.mean() * noise_level
    
    # 電源周波数（50Hz）とその高調波
    power_freq = 50.0
    harmonics = [1, 2, 3]  # 50Hz, 100Hz, 150Hz
    amplitudes = [0.8, 0.4, 0.2]
    
    for harmonic, amp in zip(harmonics, amplitudes):
        freq = power_freq * harmonic
        sigma_freq = 10.0  # 10 Hzの帯域幅
        peak_response = amp * base_amplitude * torch.exp(-0.5 * ((frequencies - freq) / sigma_freq) ** 2)
        noisy_data += peak_response
    
    # スイッチングノイズ（2kHz）
    switching_freq = 2000.0
    sigma_freq = 100.0  # 100 Hzの帯域幅
    peak_response = 0.6 * base_amplitude * torch.exp(-0.5 * ((frequencies - switching_freq) / sigma_freq) ** 2)
    noisy_data += peak_response
    
    # このノイズは全体に適用されるため、開始と終了インデックスは全体を返す
    return noisy_data, 0, num_points

