"""
フィッティング最適化モジュール
ノイズ除去前後のデータの違いを最大限に活用して精度を向上
"""

import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import find_peaks, savgol_filter
from typing import Tuple, Optional, List, Dict
import warnings

from activation_energy import psd_model, FREQ_MIN, FREQ_MAX, NUM_POINTS


def robust_fit_psd_model(
    psd_data: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
    use_advanced: bool = True,
) -> Tuple[float, float, float, float, float]:
    """
    ロバストなPSDフィッティング
    
    Args:
        psd_data: PSDデータ (3000,)
        frequencies: 周波数配列 (3000,)
        use_advanced: 高度な手法を使用するか
    
    Returns:
        (A_alpha, f_alpha, A_beta, f_beta, C) のタプル
    """
    if frequencies is None:
        frequencies = np.linspace(FREQ_MIN, FREQ_MAX, NUM_POINTS)
    
    psd_max = np.max(psd_data)
    psd_min = np.min(psd_data)
    psd_mean = np.mean(psd_data)
    psd_std = np.std(psd_data)
    
    # より精密な初期パラメータ推定
    A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init = estimate_initial_parameters_robust(
        psd_data, frequencies
    )
    
    # 境界値（データに基づいて動的に設定）
    f_alpha_lower = max(1.0, f_alpha_init * 0.01)
    f_alpha_upper = min(FREQ_MAX, f_alpha_init * 100.0)
    f_beta_lower = max(10.0, f_beta_init * 0.01)
    f_beta_upper = min(FREQ_MAX, f_beta_init * 100.0)
    
    bounds = (
        [0, f_alpha_lower, 0, f_beta_lower, 0],
        [psd_max * 200, f_alpha_upper, psd_max * 200, f_beta_upper, psd_max * 20],
    )
    
    best_result = None
    best_error = np.inf
    
    # 複数の初期値から試す
    initial_guesses = [
        [A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init],
        [A_alpha_init * 0.5, f_alpha_init * 0.5, A_beta_init * 0.5, f_beta_init * 0.5, C_init],
        [A_alpha_init * 2.0, f_alpha_init * 2.0, A_beta_init * 2.0, f_beta_init * 2.0, C_init],
        [A_alpha_init, f_alpha_init * 0.8, A_beta_init, f_beta_init * 0.8, C_init],
        [A_alpha_init, f_alpha_init * 1.2, A_beta_init, f_beta_init * 1.2, C_init],
        [psd_max * 0.5, 1000.0, psd_max * 0.3, 5000.0, psd_min * 0.1],
    ]
    
    for p0 in initial_guesses:
        try:
            popt, pcov = curve_fit(
                psd_model,
                frequencies,
                psd_data,
                p0=p0,
                bounds=bounds,
                maxfev=100000,  # より多くの評価回数
                method='trf',
                ftol=1e-12,  # より厳しい収束条件
                xtol=1e-12,
            )
            
            # フィッティングの品質を評価（MSEとMAEの両方）
            psd_predicted = psd_model(frequencies, *popt)
            mse = np.mean((psd_data - psd_predicted) ** 2)
            mae = np.mean(np.abs(psd_data - psd_predicted))
            
            # 重み付きエラー（MSEとMAEの組み合わせ）
            error = mse + mae * psd_mean * 0.1
            
            if error < best_error:
                best_error = error
                best_result = popt
        except Exception:
            continue
    
    if best_result is not None:
        return tuple(best_result)
    else:
        # フォールバック: 初期値を返す
        return A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init


def estimate_initial_parameters_robust(
    psd_data: np.ndarray,
    frequencies: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """
    よりロバストな初期パラメータ推定
    """
    psd_max = np.max(psd_data)
    psd_min = np.min(psd_data)
    psd_mean = np.mean(psd_data)
    
    # Savitzky-Golayフィルタでスムージング
    window_length = min(51, len(psd_data) // 10)
    if window_length % 2 == 0:
        window_length += 1
    polyorder = min(3, window_length - 1)
    
    try:
        psd_smooth = savgol_filter(psd_data, window_length, polyorder)
    except:
        # フォールバック
        window_size = max(10, len(psd_data) // 100)
        psd_smooth = np.convolve(psd_data, np.ones(window_size) / window_size, mode='same')
    
    # ピーク検出
    low_freq_mask = frequencies < 5000
    mid_freq_mask = (frequencies >= 5000) & (frequencies < 15000)
    
    # f_alphaの推定
    if np.any(low_freq_mask):
        low_freq_psd = psd_smooth[low_freq_mask]
        low_freq_freq = frequencies[low_freq_mask]
        
        try:
            peaks, _ = find_peaks(
                low_freq_psd,
                height=psd_mean * 0.3,
                distance=max(1, len(low_freq_psd) // 20),
            )
            if len(peaks) > 0:
                peak_heights = low_freq_psd[peaks]
                best_peak_idx = peaks[np.argmax(peak_heights)]
                f_alpha_estimated = low_freq_freq[best_peak_idx]
            else:
                f_alpha_estimated = low_freq_freq[np.argmax(low_freq_psd)]
        except:
            f_alpha_estimated = low_freq_freq[np.argmax(low_freq_psd)]
        
        f_alpha_estimated = np.clip(f_alpha_estimated, 10.0, 5000.0)
    else:
        f_alpha_estimated = 1000.0
    
    # f_betaの推定
    if np.any(mid_freq_mask):
        mid_freq_psd = psd_smooth[mid_freq_mask]
        mid_freq_freq = frequencies[mid_freq_mask]
        
        try:
            peaks, _ = find_peaks(
                mid_freq_psd,
                height=psd_mean * 0.3,
                distance=max(1, len(mid_freq_psd) // 20),
            )
            if len(peaks) > 0:
                peak_heights = mid_freq_psd[peaks]
                best_peak_idx = peaks[np.argmax(peak_heights)]
                f_beta_estimated = mid_freq_freq[best_peak_idx]
            else:
                f_beta_estimated = mid_freq_freq[np.argmax(mid_freq_psd)]
        except:
            f_beta_estimated = mid_freq_freq[np.argmax(mid_freq_psd)]
        
        f_beta_estimated = np.clip(f_beta_estimated, 1000.0, 15000.0)
    else:
        f_beta_estimated = 5000.0
    
    # 振幅の推定
    alpha_peak_value = psd_smooth[np.argmin(np.abs(frequencies - f_alpha_estimated))]
    beta_peak_value = psd_smooth[np.argmin(np.abs(frequencies - f_beta_estimated))]
    
    A_alpha_estimated = max(alpha_peak_value * 2.5, psd_max * 0.15)
    A_beta_estimated = max(beta_peak_value * 2.5, psd_max * 0.15)
    C_estimated = psd_min * 0.3
    
    return A_alpha_estimated, f_alpha_estimated, A_beta_estimated, f_beta_estimated, C_estimated

