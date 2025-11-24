"""
適応的フィッティングモジュール
データの特性に応じてフィッティング手法を動的に選択
"""

import numpy as np
from scipy.optimize import curve_fit, minimize, differential_evolution
from scipy.signal import find_peaks, savgol_filter
from typing import Tuple, Optional, List, Dict
import warnings

from activation_energy import psd_model, FREQ_MIN, FREQ_MAX, NUM_POINTS


def adaptive_fit_psd_model(
    psd_data: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
    reference_psd: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float]:
    """
    適応的PSDフィッティング
    データの特性に応じて最適な手法を選択
    """
    if frequencies is None:
        frequencies = np.linspace(FREQ_MIN, FREQ_MAX, NUM_POINTS)
    
    psd_max = np.max(psd_data)
    psd_min = np.min(psd_data)
    psd_mean = np.mean(psd_data)
    psd_std = np.std(psd_data)
    
    # データの特性を分析
    data_variance = np.var(psd_data)
    data_range = psd_max - psd_min
    data_snr = psd_mean / (psd_std + 1e-10)  # Signal-to-Noise Ratio
    
    # 参照データとの差分を計算
    if reference_psd is not None:
        diff_ratio = np.mean(np.abs(psd_data - reference_psd)) / (psd_mean + 1e-10)
    else:
        diff_ratio = 0.0
    
    # 初期パラメータ推定
    A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init = estimate_initial_parameters_adaptive(
        psd_data, frequencies, reference_psd
    )
    
    # データの特性に応じて境界値を調整
    if diff_ratio > 0.1:  # 大きな差分がある場合
        # より広い範囲を許可
        f_alpha_lower = max(0.1, f_alpha_init * 0.0001)
        f_alpha_upper = min(FREQ_MAX, f_alpha_init * 10000.0)
        f_beta_lower = max(1.0, f_beta_init * 0.0001)
        f_beta_upper = min(FREQ_MAX, f_beta_init * 10000.0)
    else:
        # 通常の範囲
        f_alpha_lower = max(1.0, f_alpha_init * 0.01)
        f_alpha_upper = min(FREQ_MAX, f_alpha_init * 100.0)
        f_beta_lower = max(10.0, f_beta_init * 0.01)
        f_beta_upper = min(FREQ_MAX, f_beta_init * 100.0)
    
    bounds = (
        [0, f_alpha_lower, 0, f_beta_lower, 0],
        [psd_max * 1000, f_alpha_upper, psd_max * 1000, f_beta_upper, psd_max * 100],
    )
    
    best_result = None
    best_error = np.inf
    best_method = None
    
    # データのハッシュ値からシードを生成（データが異なれば異なる結果）
    data_hash = hash(psd_data.tobytes()) % (2**31)
    np.random.seed(data_hash)
    
    # 方法1: 標準的なcurve_fit（複数の初期値から）
    initial_guesses = generate_diverse_initial_guesses(
        A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init,
        psd_max, psd_min, reference_psd, frequencies
    )
    
    for p0 in initial_guesses:
        try:
            popt, pcov = curve_fit(
                psd_model,
                frequencies,
                psd_data,
                p0=p0,
                bounds=bounds,
                maxfev=200000,  # より多くの評価回数
                method='trf',
                ftol=1e-15,  # より厳しい収束条件
                xtol=1e-15,
            )
            
            psd_predicted = psd_model(frequencies, *popt)
            mse = np.mean((psd_data - psd_predicted) ** 2)
            mae = np.mean(np.abs(psd_data - psd_predicted))
            rmse = np.sqrt(mse)
            
            # 複数の指標を組み合わせたエラー
            error = mse + mae * psd_mean * 0.2 + rmse * psd_std * 0.1
            
            if error < best_error:
                best_error = error
                best_result = popt
                best_method = 'curve_fit'
        except Exception:
            continue
    
    # 方法2: L-BFGS-B法（グローバル最適化）
    try:
        def objective(params):
            A_alpha, f_alpha, A_beta, f_beta, C = params
            predicted = psd_model(frequencies, A_alpha, f_alpha, A_beta, f_beta, C)
            mse = np.mean((psd_data - predicted) ** 2)
            mae = np.mean(np.abs(psd_data - predicted))
            return mse + mae * psd_mean * 0.1
        
        p0 = [A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init]
        result = minimize(
            objective,
            p0,
            method='L-BFGS-B',
            bounds=[
                (0, psd_max * 1000),
                (f_alpha_lower, f_alpha_upper),
                (0, psd_max * 1000),
                (f_beta_lower, f_beta_upper),
                (0, psd_max * 100),
            ],
            options={'maxiter': 50000, 'ftol': 1e-15},
        )
        
        if result.success:
            psd_predicted = psd_model(frequencies, *result.x)
            mse = np.mean((psd_data - psd_predicted) ** 2)
            mae = np.mean(np.abs(psd_data - psd_predicted))
            error = mse + mae * psd_mean * 0.2
            
            if error < best_error:
                best_error = error
                best_result = result.x
                best_method = 'L-BFGS-B'
    except Exception:
        pass
    
    if best_result is not None:
        return tuple(best_result)
    else:
        return A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init


def generate_diverse_initial_guesses(
    A_alpha_init: float,
    f_alpha_init: float,
    A_beta_init: float,
    f_beta_init: float,
    C_init: float,
    psd_max: float,
    psd_min: float,
    reference_psd: Optional[np.ndarray],
    frequencies: np.ndarray,
) -> List[List[float]]:
    """
    多様な初期値を生成
    """
    guesses = []
    
    # 基本の初期値
    guesses.append([A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init])
    
    # スケールバリエーション
    for scale in [0.1, 0.3, 0.5, 0.7, 1.5, 2.0, 3.0, 5.0]:
        guesses.append([
            A_alpha_init * scale,
            f_alpha_init * scale,
            A_beta_init * scale,
            f_beta_init * scale,
            C_init * scale,
        ])
    
    # 周波数のみのバリエーション
    for f_scale in [0.1, 0.3, 0.5, 0.7, 1.5, 2.0, 3.0]:
        guesses.append([
            A_alpha_init,
            f_alpha_init * f_scale,
            A_beta_init,
            f_beta_init * f_scale,
            C_init,
        ])
    
    # 参照データがある場合
    if reference_psd is not None:
        from improved_fitting import estimate_initial_parameters_improved
        A_alpha_ref, f_alpha_ref, A_beta_ref, f_beta_ref, C_ref = estimate_initial_parameters_improved(
            reference_psd, frequencies
        )
        
        # 参照データの初期値
        guesses.append([A_alpha_ref, f_alpha_ref, A_beta_ref, f_beta_ref, C_ref])
        
        # 平均値
        guesses.append([
            (A_alpha_init + A_alpha_ref) / 2,
            (f_alpha_init + f_alpha_ref) / 2,
            (A_beta_init + A_beta_ref) / 2,
            (f_beta_init + f_beta_ref) / 2,
            (C_init + C_ref) / 2,
        ])
    
    # ランダムな初期値（データの違いを反映）
    for _ in range(10):
        noise_factor = np.random.uniform(0.5, 2.0)
        f_noise_factor = np.random.uniform(0.3, 3.0)
        guesses.append([
            A_alpha_init * noise_factor,
            f_alpha_init * f_noise_factor,
            A_beta_init * noise_factor,
            f_beta_init * f_noise_factor,
            C_init * noise_factor,
        ])
    
    # デフォルト値
    guesses.extend([
        [psd_max * 0.5, 1000.0, psd_max * 0.3, 5000.0, psd_min * 0.1],
        [psd_max * 0.3, 500.0, psd_max * 0.2, 3000.0, psd_min * 0.5],
        [psd_max * 0.7, 2000.0, psd_max * 0.5, 8000.0, psd_min * 0.05],
    ])
    
    return guesses


def estimate_initial_parameters_adaptive(
    psd_data: np.ndarray,
    frequencies: np.ndarray,
    reference_psd: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float]:
    """
    適応的な初期パラメータ推定
    """
    psd_max = np.max(psd_data)
    psd_min = np.min(psd_data)
    psd_mean = np.mean(psd_data)
    psd_median = np.median(psd_data)
    psd_std = np.std(psd_data)
    
    # より高度なスムージング
    window_length = min(151, len(psd_data) // 3)
    if window_length % 2 == 0:
        window_length += 1
    polyorder = min(7, window_length - 1)
    
    try:
        psd_smooth = savgol_filter(psd_data, window_length, polyorder)
    except:
        window_size = max(30, len(psd_data) // 30)
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
            peaks, properties = find_peaks(
                low_freq_psd,
                height=psd_median * 0.1,
                distance=max(1, len(low_freq_psd) // 50),
                prominence=psd_std * 0.05,
                width=1,
            )
            
            if len(peaks) > 0:
                peak_heights = low_freq_psd[peaks]
                best_peak_idx = peaks[np.argmax(peak_heights)]
                f_alpha_estimated = low_freq_freq[best_peak_idx]
            else:
                f_alpha_estimated = low_freq_freq[np.argmax(low_freq_psd)]
        except:
            f_alpha_estimated = low_freq_freq[np.argmax(low_freq_psd)]
        
        f_alpha_estimated = np.clip(f_alpha_estimated, 1.0, 5000.0)
    else:
        f_alpha_estimated = 1000.0
    
    if np.any(mid_freq_mask):
        mid_freq_psd = psd_smooth[mid_freq_mask]
        mid_freq_freq = frequencies[mid_freq_mask]
        
        try:
            peaks, properties = find_peaks(
                mid_freq_psd,
                height=psd_median * 0.1,
                distance=max(1, len(mid_freq_psd) // 50),
                prominence=psd_std * 0.05,
                width=1,
            )
            
            if len(peaks) > 0:
                peak_heights = mid_freq_psd[peaks]
                best_peak_idx = peaks[np.argmax(peak_heights)]
                f_beta_estimated = mid_freq_freq[best_peak_idx]
            else:
                f_beta_estimated = mid_freq_freq[np.argmax(mid_freq_psd)]
        except:
            f_beta_estimated = mid_freq_freq[np.argmax(mid_freq_psd)]
        
        f_beta_estimated = np.clip(f_beta_estimated, 100.0, 15000.0)
    else:
        f_beta_estimated = 5000.0
    
    # 振幅の推定
    alpha_peak_value = psd_smooth[np.argmin(np.abs(frequencies - f_alpha_estimated))]
    beta_peak_value = psd_smooth[np.argmin(np.abs(frequencies - f_beta_estimated))]
    
    A_alpha_estimated = max(alpha_peak_value * 4.0, psd_max * 0.3)
    A_beta_estimated = max(beta_peak_value * 4.0, psd_max * 0.3)
    C_estimated = psd_min * 0.1
    
    return A_alpha_estimated, f_alpha_estimated, A_beta_estimated, f_beta_estimated, C_estimated

