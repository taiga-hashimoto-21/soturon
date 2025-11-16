"""
局所スパイクノイズの実装
一部のポイントに集中的に発生するスパイク状のノイズ
電磁干渉（EMI）、静電気放電（ESD）、接触不良などを模擬
"""

import torch


def add_localized_spike_noise(psd_data, interval_idx, noise_level=0.1, num_intervals=30, spike_ratio=0.15):
    """
    パターン2: 局所スパイクノイズ
    一部のポイントに集中的に発生するスパイク状のノイズ
    電磁干渉（EMI）、静電気放電（ESD）、接触不良などを模擬
    
    Args:
        psd_data: PSDデータ (3000ポイント)
        interval_idx: ノイズを加える区間のインデックス
        noise_level: ノイズの強度（元の値に対する倍率）
        num_intervals: 区間数
        spike_ratio: スパイクが発生するポイントの割合（0.15 = 15%のポイントに大きなスパイク）
    
    Returns:
        ノイズを加えたPSDデータ, 開始インデックス, 終了インデックス
    """
    noisy_data = psd_data.clone()
    points_per_interval = len(psd_data) // num_intervals
    
    start_idx = interval_idx * points_per_interval
    end_idx = start_idx + points_per_interval
    
    # その区間の平均値を取得
    interval_mean = psd_data[start_idx:end_idx].mean()
    
    # スパイクが発生するポイント数を計算
    num_spike_points = int(points_per_interval * spike_ratio)
    num_spike_points = max(1, num_spike_points)  # 最低1ポイントはスパイク
    
    # スパイクが発生するポイントをランダムに選択
    spike_indices = torch.randperm(points_per_interval)[:num_spike_points]
    
    # ノイズを初期化
    noise = torch.zeros(points_per_interval)
    
    # スパイクポイントには大きなノイズ（3-5倍の強度）
    spike_strength_multiplier = 3.0 + torch.rand(num_spike_points) * 2.0  # 3.0-5.0の範囲
    noise[spike_indices] = torch.randn(num_spike_points) * interval_mean * noise_level * spike_strength_multiplier
    
    # その他のポイントには小さなバックグラウンドノイズ（0.1倍の強度）
    other_indices = torch.ones(points_per_interval, dtype=torch.bool)
    other_indices[spike_indices] = False
    noise[other_indices] = torch.randn(points_per_interval - num_spike_points) * interval_mean * noise_level * 0.1
    
    # ノイズを加える
    noisy_data[start_idx:end_idx] += noise
    
    return noisy_data, start_idx, end_idx

