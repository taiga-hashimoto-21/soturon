"""
クロック漏れノイズの実装（周波数特性版）
クロック周波数（5000 Hz）にピークを持つノイズを周波数特性として理論的に定義
"""

import torch
import numpy as np


def add_clock_leakage_noise(psd_data, interval_idx, noise_level=0.1, num_intervals=30,
                            freq_min=0.0, freq_max=15000.0):
    """
    クロック漏れノイズ（周波数特性版）
    クロック周波数（5000 Hz）にのみピークを持つノイズを周波数特性として理論的に定義
    
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
    
    # クロック周波数（5000Hz）にのみピーク
    clock_freq = 5000.0  # Hz
    base_amplitude = psd_data.mean() * noise_level
    
    # 非常に狭い帯域幅でピークを追加
    sigma_freq = 10.0  # 10 Hzの帯域幅（狭いピーク）
    peak_response = base_amplitude * torch.exp(-0.5 * ((frequencies - clock_freq) / sigma_freq) ** 2)
    noisy_data += peak_response
    
    # このノイズは全体に適用されるため、開始と終了インデックスは全体を返す
    return noisy_data, 0, num_points

