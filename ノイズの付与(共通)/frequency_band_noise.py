"""
周波数帯域集中ノイズの実装（周波数特性版）
特定の周波数帯域に集中的に発生するノイズを周波数特性として理論的に定義
電源ノイズ、共振、クロストークなどを模擬
"""

import torch
import numpy as np


def add_frequency_band_noise(psd_data, interval_idx, noise_level=0.1, num_intervals=30, band_ratio=0.3,
                             freq_min=0.0, freq_max=1000.0):
    """
    パターン1: 周波数帯域集中ノイズ（周波数特性版）
    特定の周波数帯域に集中的に発生するノイズを周波数特性として理論的に定義
    電源ノイズ、共振、クロストークなどを模擬
    
    Args:
        psd_data: PSDデータ (3000ポイント)
        interval_idx: ノイズを加える区間のインデックス
        noise_level: ノイズの強度（元の値に対する倍率）
        num_intervals: 区間数
        band_ratio: ノイズが集中する帯域の割合（0.3 = 30%の周波数範囲に集中的にノイズ）
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
    f_center = (f_start + f_end) / 2.0  # 区間の中心周波数（Hz）
    
    # ノイズが集中する帯域の周波数幅を計算（Hz単位）
    freq_band_width = (f_end - f_start) * band_ratio  # Hz
    
    # 帯域の中心周波数をランダムに決定（帯域が区間内に収まるように）
    min_f_center = f_start + freq_band_width / 2.0
    max_f_center = f_end - freq_band_width / 2.0
    if max_f_center <= min_f_center:
        # 区間が狭すぎる場合は区間の中心を使用
        band_center_freq = f_center
    else:
        # ランダムに中心周波数を選択
        band_center_freq = min_f_center + torch.rand(1).item() * (max_f_center - min_f_center)
    
    # 周波数特性を理論的に定義: N(f) = A * exp(-(f - f_center)² / (2σ²))
    # ガウシアン分布の標準偏差（Hz単位）
    sigma_freq = freq_band_width / (2 * np.sqrt(2 * np.log(2)))  # 3dB帯域幅からσを計算
    if sigma_freq < 1e-10:  # 0割りを防ぐ
        sigma_freq = (f_end - f_start) / 10.0
    
    # 周波数特性: ガウシアン型
    # N(f) = exp(-(f - f_center)² / (2σ²))
    frequency_response = torch.exp(-0.5 * ((f_interval - band_center_freq) / sigma_freq) ** 2)
    
    # ベースノイズを生成（ランダム成分）
    base_noise = torch.randn(points_per_interval) * interval_mean * noise_level
    
    # 周波数特性を適用してノイズを生成
    noise = base_noise * frequency_response
    
    # ノイズを加える
    noisy_data[start_idx:end_idx] += noise
    
    return noisy_data, start_idx, end_idx

