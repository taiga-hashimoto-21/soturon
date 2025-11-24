"""
改善されたフィッティングモジュール
ノイズ除去前後のデータの違いを最大限に活用
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks, savgol_filter
from typing import Tuple, Optional, List
import warnings

from activation_energy import psd_model, FREQ_MIN, FREQ_MAX, NUM_POINTS


def fit_psd_model_improved(
    psd_data: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
    reference_psd: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None,
) -> Tuple[float, float, float, float, float]:
    """
    改善されたPSDフィッティング
    
    Args:
        psd_data: PSDデータ (3000,)
        frequencies: 周波数配列 (3000,)
        reference_psd: 参照PSDデータ（ノイズ除去前後の比較用、オプション）
    
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
    A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init = estimate_initial_parameters_improved(
        psd_data, frequencies
    )
    
    # 参照データがある場合は、その情報も活用
    if reference_psd is not None:
        # 参照データからも初期パラメータを推定
        A_alpha_ref, f_alpha_ref, A_beta_ref, f_beta_ref, C_ref = estimate_initial_parameters_improved(
            reference_psd, frequencies
        )
        
        # 参照データと現在のデータの差分を考慮
        diff_ratio = np.mean(np.abs(psd_data - reference_psd)) / (psd_mean + 1e-10)
        
        # 差分が大きい場合は、参照データの情報を重み付け
        if diff_ratio > 0.05:  # 5%以上の差分がある場合
            # 重み付き平均
            weight_current = 0.7
            weight_ref = 0.3
            
            f_alpha_init = f_alpha_init * weight_current + f_alpha_ref * weight_ref
            f_beta_init = f_beta_init * weight_current + f_beta_ref * weight_ref
    
    # 境界値（データに基づいて動的に設定）
    f_alpha_lower = max(1.0, f_alpha_init * 0.001)  # より広い範囲
    f_alpha_upper = min(FREQ_MAX, f_alpha_init * 1000.0)
    f_beta_lower = max(10.0, f_beta_init * 0.001)
    f_beta_upper = min(FREQ_MAX, f_beta_init * 1000.0)
    
    bounds = (
        [0, f_alpha_lower, 0, f_beta_lower, 0],
        [psd_max * 500, f_alpha_upper, psd_max * 500, f_beta_upper, psd_max * 50],
    )
    
    # ランダムシードを設定（データの違いを反映するため）
    if random_seed is not None:
        np.random.seed(random_seed)
    else:
        # データのハッシュ値からシードを生成（データが異なれば異なるシード）
        data_hash = hash(psd_data.tobytes()) % (2**31)
        np.random.seed(data_hash)
    
    best_result = None
    best_error = np.inf
    
    # より多くの初期値から試す（ランダム性を追加）
    base_guesses = [
        [A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init],
        [A_alpha_init * 0.3, f_alpha_init * 0.3, A_beta_init * 0.3, f_beta_init * 0.3, C_init],
        [A_alpha_init * 0.7, f_alpha_init * 0.7, A_beta_init * 0.7, f_beta_init * 0.7, C_init],
        [A_alpha_init * 1.5, f_alpha_init * 1.5, A_beta_init * 1.5, f_beta_init * 1.5, C_init],
        [A_alpha_init * 3.0, f_alpha_init * 3.0, A_beta_init * 3.0, f_beta_init * 3.0, C_init],
        [A_alpha_init, f_alpha_init * 0.5, A_beta_init, f_beta_init * 0.5, C_init],
        [A_alpha_init, f_alpha_init * 2.0, A_beta_init, f_beta_init * 2.0, C_init],
        [psd_max * 0.5, 1000.0, psd_max * 0.3, 5000.0, psd_min * 0.1],
        [psd_max * 0.3, 500.0, psd_max * 0.2, 3000.0, psd_min * 0.5],
        [psd_max * 0.7, 2000.0, psd_max * 0.5, 8000.0, psd_min * 0.05],
    ]
    
    # ランダムな摂動を加えた初期値も追加
    initial_guesses = base_guesses.copy()
    for _ in range(5):  # 5つのランダムな初期値を追加
        noise_factor = np.random.uniform(0.8, 1.2)
        f_noise_factor = np.random.uniform(0.7, 1.3)
        initial_guesses.append([
            A_alpha_init * noise_factor,
            f_alpha_init * f_noise_factor,
            A_beta_init * noise_factor,
            f_beta_init * f_noise_factor,
            C_init * noise_factor,
        ])
    
    for p0 in initial_guesses:
        try:
            popt, pcov = curve_fit(
                psd_model,
                frequencies,
                psd_data,
                p0=p0,
                bounds=bounds,
                maxfev=100000,
                method='trf',
                ftol=1e-12,
                xtol=1e-12,
            )
            
            # フィッティングの品質を評価（複数の指標を使用）
            psd_predicted = psd_model(frequencies, *popt)
            mse = np.mean((psd_data - psd_predicted) ** 2)
            mae = np.mean(np.abs(psd_data - psd_predicted))
            rmse = np.sqrt(mse)
            
            # 重み付きエラー（MSE、MAE、RMSEの組み合わせ）
            error = mse + mae * psd_mean * 0.1 + rmse * psd_std * 0.05
            
            if error < best_error:
                best_error = error
                best_result = popt
        except Exception:
            continue
    
    if best_result is not None:
        return tuple(best_result)
    else:
        return A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init


def estimate_initial_parameters_improved(
    psd_data: np.ndarray,
    frequencies: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """
    より改善された初期パラメータ推定
    """
    psd_max = np.max(psd_data)
    psd_min = np.min(psd_data)
    psd_mean = np.mean(psd_data)
    psd_median = np.median(psd_data)
    
    # より高度なスムージング
    window_length = min(101, len(psd_data) // 5)
    if window_length % 2 == 0:
        window_length += 1
    polyorder = min(5, window_length - 1)
    
    try:
        psd_smooth = savgol_filter(psd_data, window_length, polyorder)
    except:
        window_size = max(20, len(psd_data) // 50)
        psd_smooth = np.convolve(psd_data, np.ones(window_size) / window_size, mode='same')
    
    # ピーク検出（より敏感に）
    low_freq_mask = frequencies < 5000
    mid_freq_mask = (frequencies >= 5000) & (frequencies < 15000)
    
    f_alpha_estimated = None
    f_beta_estimated = None
    
    if np.any(low_freq_mask):
        low_freq_psd = psd_smooth[low_freq_mask]
        low_freq_freq = frequencies[low_freq_mask]
        
        try:
            # より敏感なピーク検出
            peaks, properties = find_peaks(
                low_freq_psd,
                height=psd_median * 0.2,  # より低い閾値
                distance=max(1, len(low_freq_psd) // 30),  # より近いピークも検出
                prominence=psd_std * 0.1,  # プロミネンス条件
            )
            
            if len(peaks) > 0:
                # 最も高いピークを選択
                peak_heights = low_freq_psd[peaks]
                best_peak_idx = peaks[np.argmax(peak_heights)]
                f_alpha_estimated = low_freq_freq[best_peak_idx]
            else:
                # ピークが見つからない場合は最大値の位置
                f_alpha_estimated = low_freq_freq[np.argmax(low_freq_psd)]
        except:
            f_alpha_estimated = low_freq_freq[np.argmax(low_freq_psd)]
        
        f_alpha_estimated = np.clip(f_alpha_estimated, 5.0, 5000.0)
    else:
        f_alpha_estimated = 1000.0
    
    if np.any(mid_freq_mask):
        mid_freq_psd = psd_smooth[mid_freq_mask]
        mid_freq_freq = frequencies[mid_freq_mask]
        
        try:
            peaks, properties = find_peaks(
                mid_freq_psd,
                height=psd_median * 0.2,
                distance=max(1, len(mid_freq_psd) // 30),
                prominence=psd_std * 0.1,
            )
            
            if len(peaks) > 0:
                peak_heights = mid_freq_psd[peaks]
                best_peak_idx = peaks[np.argmax(peak_heights)]
                f_beta_estimated = mid_freq_freq[best_peak_idx]
            else:
                f_beta_estimated = mid_freq_freq[np.argmax(mid_freq_psd)]
        except:
            f_beta_estimated = mid_freq_freq[np.argmax(mid_freq_psd)]
        
        f_beta_estimated = np.clip(f_beta_estimated, 500.0, 15000.0)
    else:
        f_beta_estimated = 5000.0
    
    # 振幅の推定（より精密に）
    alpha_peak_value = psd_smooth[np.argmin(np.abs(frequencies - f_alpha_estimated))]
    beta_peak_value = psd_smooth[np.argmin(np.abs(frequencies - f_beta_estimated))]
    
    # 理論式から逆算: S(f_c) = A/2 + C ≈ A/2
    A_alpha_estimated = max(alpha_peak_value * 3.0, psd_max * 0.2)
    A_beta_estimated = max(beta_peak_value * 3.0, psd_max * 0.2)
    C_estimated = psd_min * 0.2
    
    return A_alpha_estimated, f_alpha_estimated, A_beta_estimated, f_beta_estimated, C_estimated

