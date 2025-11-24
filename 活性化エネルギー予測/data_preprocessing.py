"""
データ前処理モジュール
ノイズ除去前後のデータの違いを最大化するための前処理
"""

import numpy as np
from scipy.signal import savgol_filter, medfilt
from typing import Tuple


def preprocess_psd_data(
    psd_data: np.ndarray,
    method: str = 'savgol',
    window_length: Optional[int] = None,
) -> np.ndarray:
    """
    PSDデータの前処理
    
    Args:
        psd_data: PSDデータ (3000,)
        method: 前処理方法 ('savgol', 'median', 'none')
        window_length: ウィンドウサイズ（Noneの場合は自動設定）
    
    Returns:
        前処理後のPSDデータ
    """
    if method == 'none':
        return psd_data
    
    if window_length is None:
        window_length = min(51, len(psd_data) // 10)
        if window_length % 2 == 0:
            window_length += 1
    
    if method == 'savgol':
        try:
            polyorder = min(3, window_length - 1)
            return savgol_filter(psd_data, window_length, polyorder)
        except:
            return psd_data
    elif method == 'median':
        kernel_size = min(window_length, len(psd_data) // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return medfilt(psd_data, kernel_size)
    else:
        return psd_data


def enhance_difference_between_noisy_and_denoised(
    noisy_psd: np.ndarray,
    denoised_psd: np.ndarray,
    enhancement_factor: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ノイズ付きデータとノイズ除去後データの違いを強調
    
    Args:
        noisy_psd: ノイズ付きPSDデータ
        denoised_psd: ノイズ除去後PSDデータ
        enhancement_factor: 強調係数
    
    Returns:
        (強調後のノイズ付きデータ, 強調後のノイズ除去後データ)
    """
    # 差分を計算
    diff = noisy_psd - denoised_psd
    
    # 差分を強調
    enhanced_diff = diff * enhancement_factor
    
    # 強調後のデータを計算
    enhanced_noisy = denoised_psd + enhanced_diff
    enhanced_denoised = denoised_psd
    
    return enhanced_noisy, enhanced_denoised


def adaptive_smoothing(
    psd_data: np.ndarray,
    frequencies: np.ndarray,
    noise_level: float = 0.3,
) -> np.ndarray:
    """
    適応的スムージング
    ノイズレベルに応じてスムージングの強度を調整
    """
    # ノイズレベルに応じてウィンドウサイズを調整
    base_window = 51
    window_length = int(base_window * (1 + noise_level))
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, len(psd_data) // 5)
    
    try:
        polyorder = min(3, window_length - 1)
        return savgol_filter(psd_data, window_length, polyorder)
    except:
        return psd_data

