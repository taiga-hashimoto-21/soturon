"""
振幅依存ノイズの実装（周波数特性版）
元の信号の振幅に依存したノイズを周波数特性として理論的に定義
非線形増幅器の歪み、ADCの量子化ノイズ、飽和などを模擬
"""

import torch


def add_amplitude_dependent_noise(psd_data, interval_idx, noise_level=0.1, num_intervals=30, top_ratio=0.3,
                                  freq_min=0.0, freq_max=1000.0):
    """
    パターン3: 振幅依存ノイズ（周波数特性版）
    元の信号の振幅に依存したノイズを周波数特性として理論的に定義
    非線形増幅器の歪み、ADCの量子化ノイズ、飽和などを模擬
    
    周波数特性: N(f) = A * sqrt(PSD(f)) * noise_level
    PSDの振幅が大きい周波数ほどノイズが大きくなる
    
    Args:
        psd_data: PSDデータ (3000ポイント)
        interval_idx: ノイズを加える区間のインデックス
        noise_level: ノイズの強度（元の値に対する倍率）
        num_intervals: 区間数
        top_ratio: ノイズが集中する上位の割合（0.3 = 上位30%の周波数に集中的にノイズ）
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
    
    # その区間のデータを取得
    interval_data = psd_data[start_idx:end_idx]
    
    # 周波数軸を生成（実際の周波数：Hz）
    frequencies = torch.linspace(freq_min, freq_max, num_points)
    f_interval = frequencies[start_idx:end_idx]  # 区間内の周波数軸
    
    # 振幅の大きい順にソートしてインデックスを取得
    sorted_indices = torch.argsort(interval_data, descending=True)
    
    # 上位のポイント数を計算
    num_top_points = int(points_per_interval * top_ratio)
    num_top_points = max(1, num_top_points)  # 最低1ポイント
    
    # 上位のポイントのインデックス
    top_indices = sorted_indices[:num_top_points]
    
    # ノイズを初期化
    noise = torch.zeros(points_per_interval)
    
    # 周波数特性を理論的に定義: N(f) = A * sqrt(PSD(f))
    # PSDの振幅が大きい周波数ほどノイズが大きくなる
    top_values = interval_data[top_indices]
    top_mean = interval_data.mean()
    
    # 振幅が大きいほどノイズが大きくなる（最大2倍まで）
    amplitude_factors = 1.0 + (top_values / (top_mean + 1e-10))
    amplitude_factors = torch.clamp(amplitude_factors, 1.0, 2.0)
    
    # 上位周波数には大きなノイズ（振幅に比例）
    noise[top_indices] = torch.randn(num_top_points) * top_mean * noise_level * amplitude_factors
    
    # その他の周波数には小さなノイズ（0.2倍の強度）
    other_indices = torch.ones(points_per_interval, dtype=torch.bool)
    other_indices[top_indices] = False
    noise[other_indices] = torch.randn(points_per_interval - num_top_points) * top_mean * noise_level * 0.2
    
    # ノイズを加える
    noisy_data[start_idx:end_idx] += noise
    
    return noisy_data, start_idx, end_idx

