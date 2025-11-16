"""
振幅依存ノイズの実装
元の信号が大きい領域に集中的にノイズが発生
非線形増幅器の歪み、ADCの量子化ノイズ、飽和などを模擬
"""

import torch


def add_amplitude_dependent_noise(psd_data, interval_idx, noise_level=0.1, num_intervals=30, top_ratio=0.3):
    """
    パターン3: 振幅依存ノイズ
    元の信号が大きい領域に集中的にノイズが発生
    非線形増幅器の歪み、ADCの量子化ノイズ、飽和などを模擬
    
    Args:
        psd_data: PSDデータ (3000ポイント)
        interval_idx: ノイズを加える区間のインデックス
        noise_level: ノイズの強度（元の値に対する倍率）
        num_intervals: 区間数
        top_ratio: ノイズが集中する上位の割合（0.3 = 上位30%のポイントに集中的にノイズ）
    
    Returns:
        ノイズを加えたPSDデータ, 開始インデックス, 終了インデックス
    """
    noisy_data = psd_data.clone()
    points_per_interval = len(psd_data) // num_intervals
    
    start_idx = interval_idx * points_per_interval
    end_idx = start_idx + points_per_interval
    
    # その区間のデータを取得
    interval_data = psd_data[start_idx:end_idx]
    
    # 振幅の大きい順にソートしてインデックスを取得
    sorted_indices = torch.argsort(interval_data, descending=True)
    
    # 上位のポイント数を計算
    num_top_points = int(points_per_interval * top_ratio)
    num_top_points = max(1, num_top_points)  # 最低1ポイント
    
    # 上位のポイントのインデックス
    top_indices = sorted_indices[:num_top_points]
    
    # ノイズを初期化
    noise = torch.zeros(points_per_interval)
    
    # 上位ポイントには大きなノイズ（振幅に比例）
    # 各ポイントの振幅に応じてノイズの強度を変える
    top_values = interval_data[top_indices]
    top_mean = interval_data.mean()
    
    # 振幅が大きいほどノイズが大きくなる（最大2倍まで）
    amplitude_factors = 1.0 + (top_values / (top_mean + 1e-10))
    amplitude_factors = torch.clamp(amplitude_factors, 1.0, 2.0)
    
    noise[top_indices] = torch.randn(num_top_points) * top_mean * noise_level * amplitude_factors
    
    # その他のポイントには小さなノイズ（0.2倍の強度）
    other_indices = torch.ones(points_per_interval, dtype=torch.bool)
    other_indices[top_indices] = False
    noise[other_indices] = torch.randn(points_per_interval - num_top_points) * top_mean * noise_level * 0.2
    
    # ノイズを加える
    noisy_data[start_idx:end_idx] += noise
    
    return noisy_data, start_idx, end_idx

