"""
周波数帯域集中ノイズの実装
特定の周波数帯域に集中的に発生するノイズ
電源ノイズ、共振、クロストークなどを模擬
"""

import torch


def add_frequency_band_noise(psd_data, interval_idx, noise_level=0.1, num_intervals=30, band_ratio=0.3):
    """
    パターン1: 周波数帯域集中ノイズ
    特定の周波数帯域に集中的に発生するノイズ
    電源ノイズ、共振、クロストークなどを模擬
    
    Args:
        psd_data: PSDデータ (3000ポイント)
        interval_idx: ノイズを加える区間のインデックス
        noise_level: ノイズの強度（元の値に対する倍率）
        num_intervals: 区間数
        band_ratio: ノイズが集中する帯域の割合（0.3 = 30%のポイントに集中的にノイズ）
    
    Returns:
        ノイズを加えたPSDデータ, 開始インデックス, 終了インデックス
    """
    noisy_data = psd_data.clone()
    points_per_interval = len(psd_data) // num_intervals
    
    start_idx = interval_idx * points_per_interval
    end_idx = start_idx + points_per_interval
    
    # その区間の平均値を取得
    interval_mean = psd_data[start_idx:end_idx].mean()
    
    # ノイズが集中する帯域の幅を計算
    band_width = int(points_per_interval * band_ratio)
    
    # 帯域の中心位置をランダムに決定（帯域が区間内に収まるように）
    max_center = points_per_interval - band_width // 2
    min_center = band_width // 2
    band_center = torch.randint(min_center, max_center, (1,)).item()
    
    # ガウシアン分布でノイズの強度を減衰させる
    positions = torch.arange(points_per_interval, dtype=torch.float32)
    # 帯域の中心からの距離
    distances = torch.abs(positions - band_center)
    # ガウシアン分布の標準偏差（帯域幅の1/3程度）
    sigma = band_width / 3.0
    # ガウシアン重み（中心で1.0、外側で減衰）
    weights = torch.exp(-0.5 * (distances / sigma) ** 2)
    
    # ベースノイズを生成
    base_noise = torch.randn(points_per_interval) * interval_mean * noise_level
    
    # ガウシアン重みをかけて帯域集中ノイズを生成
    noise = base_noise * weights
    
    # ノイズを加える
    noisy_data[start_idx:end_idx] += noise
    
    return noisy_data, start_idx, end_idx

