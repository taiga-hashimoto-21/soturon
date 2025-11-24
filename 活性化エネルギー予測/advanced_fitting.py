"""
高度なフィッティング手法の実装
複数のフィッティング手法を試して最適な結果を選択
"""

import numpy as np
from scipy.optimize import curve_fit, differential_evolution, minimize
from scipy.signal import find_peaks, savgol_filter
from typing import Tuple, Optional, List, Dict
import warnings

from activation_energy import psd_model, FREQ_MIN, FREQ_MAX, NUM_POINTS


def fit_with_differential_evolution(
    psd_data: np.ndarray,
    frequencies: np.ndarray,
    psd_max: float,
    psd_min: float,
    f_alpha_init: float,
    f_beta_init: float,
) -> Tuple[float, float, float, float, float]:
    """
    差分進化アルゴリズムを使用したフィッティング
    グローバル最適化が可能
    """
    # 境界値
    bounds = [
        (0, psd_max * 100),  # A_alpha
        (max(1.0, f_alpha_init * 0.1), min(FREQ_MAX, f_alpha_init * 10.0)),  # f_alpha
        (0, psd_max * 100),  # A_beta
        (max(10.0, f_beta_init * 0.1), min(FREQ_MAX, f_beta_init * 10.0)),  # f_beta
        (0, psd_max * 10),  # C
    ]
    
    def objective(params):
        A_alpha, f_alpha, A_beta, f_beta, C = params
        try:
            predicted = psd_model(frequencies, A_alpha, f_alpha, A_beta, f_beta, C)
            error = np.mean((psd_data - predicted) ** 2)
            return error
        except:
            return np.inf
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=1000,
            popsize=15,
            tol=1e-6,
        )
        
        if result.success:
            A_alpha, f_alpha, A_beta, f_beta, C = result.x
            return A_alpha, f_alpha, A_beta, f_beta, C
    except:
        pass
    
    return None


def fit_with_multiple_methods(
    psd_data: np.ndarray,
    frequencies: np.ndarray,
    psd_max: float,
    psd_min: float,
    A_alpha_init: float,
    f_alpha_init: float,
    A_beta_init: float,
    f_beta_init: float,
    C_init: float,
) -> List[Dict]:
    """
    複数のフィッティング手法を試して結果を返す
    """
    results = []
    
    # 境界値
    f_alpha_lower = max(1.0, f_alpha_init * 0.01)
    f_alpha_upper = min(FREQ_MAX, f_alpha_init * 100.0)
    f_beta_lower = max(10.0, f_beta_init * 0.01)
    f_beta_upper = min(FREQ_MAX, f_beta_init * 100.0)
    
    bounds = (
        [0, f_alpha_lower, 0, f_beta_lower, 0],
        [psd_max * 100, f_alpha_upper, psd_max * 100, f_beta_upper, psd_max * 10],
    )
    
    # 方法1: 標準的なcurve_fit (trf)
    initial_guesses = [
        [A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init],
        [A_alpha_init, f_alpha_init * 0.5, A_beta_init, f_beta_init * 0.5, C_init],
        [A_alpha_init, f_alpha_init * 2.0, A_beta_init, f_beta_init * 2.0, C_init],
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
                maxfev=50000,
                method='trf',
                ftol=1e-10,
                xtol=1e-10,
            )
            
            psd_predicted = psd_model(frequencies, *popt)
            mse = np.mean((psd_data - psd_predicted) ** 2)
            mae = np.mean(np.abs(psd_data - psd_predicted))
            
            results.append({
                'method': 'trf',
                'params': popt,
                'mse': mse,
                'mae': mae,
                'initial': p0,
            })
        except:
            continue
    
    # 方法2: L-BFGS-B法
    try:
        def objective(params):
            A_alpha, f_alpha, A_beta, f_beta, C = params
            predicted = psd_model(frequencies, A_alpha, f_alpha, A_beta, f_beta, C)
            return np.mean((psd_data - predicted) ** 2)
        
        p0 = [A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init]
        result = minimize(
            objective,
            p0,
            method='L-BFGS-B',
            bounds=[
                (0, psd_max * 100),
                (f_alpha_lower, f_alpha_upper),
                (0, psd_max * 100),
                (f_beta_lower, f_beta_upper),
                (0, psd_max * 10),
            ],
            options={'maxiter': 10000, 'ftol': 1e-10},
        )
        
        if result.success:
            psd_predicted = psd_model(frequencies, *result.x)
            mse = np.mean((psd_data - psd_predicted) ** 2)
            mae = np.mean(np.abs(psd_data - psd_predicted))
            
            results.append({
                'method': 'L-BFGS-B',
                'params': result.x,
                'mse': mse,
                'mae': mae,
                'initial': p0,
            })
    except:
        pass
    
    # 方法3: 差分進化
    de_result = fit_with_differential_evolution(
        psd_data, frequencies, psd_max, psd_min, f_alpha_init, f_beta_init
    )
    if de_result is not None:
        A_alpha, f_alpha, A_beta, f_beta, C = de_result
        psd_predicted = psd_model(frequencies, A_alpha, f_alpha, A_beta, f_beta, C)
        mse = np.mean((psd_data - psd_predicted) ** 2)
        mae = np.mean(np.abs(psd_data - psd_predicted))
        
        results.append({
            'method': 'differential_evolution',
            'params': de_result,
            'mse': mse,
            'mae': mae,
            'initial': None,
        })
    
    return results


def select_best_fit(results: List[Dict], criterion: str = 'mse') -> Optional[np.ndarray]:
    """
    複数のフィッティング結果から最適なものを選択
    
    Args:
        results: フィッティング結果のリスト
        criterion: 選択基準 ('mse' or 'mae')
    
    Returns:
        最適なパラメータ配列
    """
    if len(results) == 0:
        return None
    
    if criterion == 'mse':
        best_idx = np.argmin([r['mse'] for r in results])
    else:
        best_idx = np.argmin([r['mae'] for r in results])
    
    return results[best_idx]['params']


def ensemble_fit(results: List[Dict], weights: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    複数のフィッティング結果をアンサンブル（重み付き平均）
    
    Args:
        results: フィッティング結果のリスト
        weights: 各結果の重み（Noneの場合はMSEの逆数を重みとして使用）
    
    Returns:
        アンサンブルされたパラメータ配列
    """
    if len(results) == 0:
        return None
    
    if weights is None:
        # MSEの逆数を重みとして使用
        mses = np.array([r['mse'] for r in results])
        # ゼロ除算を避ける
        mses = np.maximum(mses, 1e-20)
        weights = 1.0 / mses
        weights = weights / weights.sum()  # 正規化
    
    # 重み付き平均
    params_array = np.array([r['params'] for r in results])
    ensemble_params = np.average(params_array, axis=0, weights=weights)
    
    return ensemble_params


def improved_estimate_initial_parameters(
    psd_data: np.ndarray,
    frequencies: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """
    より精密な初期パラメータ推定
    """
    psd_max = np.max(psd_data)
    psd_min = np.min(psd_data)
    psd_mean = np.mean(psd_data)
    
    # より高度なスムージング（Savitzky-Golayフィルタ）
    window_length = min(51, len(psd_data) // 10)
    if window_length % 2 == 0:
        window_length += 1
    polyorder = min(3, window_length - 1)
    
    try:
        psd_smooth = savgol_filter(psd_data, window_length, polyorder)
    except:
        # フォールバック: 単純な移動平均
        window_size = max(10, len(psd_data) // 100)
        psd_smooth = np.convolve(psd_data, np.ones(window_size) / window_size, mode='same')
    
    # ピーク検出を使用
    # 低周波数側と中周波数側で別々にピークを探す
    low_freq_mask = frequencies < 5000
    mid_freq_mask = (frequencies >= 5000) & (frequencies < 15000)
    
    f_alpha_estimated = None
    f_beta_estimated = None
    
    if np.any(low_freq_mask):
        low_freq_psd = psd_smooth[low_freq_mask]
        low_freq_freq = frequencies[low_freq_mask]
        
        # ピーク検出
        peaks, properties = find_peaks(
            low_freq_psd,
            height=psd_mean * 0.5,
            distance=len(low_freq_psd) // 10,
        )
        
        if len(peaks) > 0:
            # 最も高いピークを選択
            peak_heights = low_freq_psd[peaks]
            best_peak_idx = peaks[np.argmax(peak_heights)]
            f_alpha_estimated = low_freq_freq[best_peak_idx]
        else:
            # ピークが見つからない場合は最大値の位置
            f_alpha_estimated = low_freq_freq[np.argmax(low_freq_psd)]
        
        f_alpha_estimated = np.clip(f_alpha_estimated, 10.0, 5000.0)
    else:
        f_alpha_estimated = 1000.0
    
    if np.any(mid_freq_mask):
        mid_freq_psd = psd_smooth[mid_freq_mask]
        mid_freq_freq = frequencies[mid_freq_mask]
        
        # ピーク検出
        peaks, properties = find_peaks(
            mid_freq_psd,
            height=psd_mean * 0.5,
            distance=len(mid_freq_psd) // 10,
        )
        
        if len(peaks) > 0:
            # 最も高いピークを選択
            peak_heights = mid_freq_psd[peaks]
            best_peak_idx = peaks[np.argmax(peak_heights)]
            f_beta_estimated = mid_freq_freq[best_peak_idx]
        else:
            # ピークが見つからない場合は最大値の位置
            f_beta_estimated = mid_freq_freq[np.argmax(mid_freq_psd)]
        
        f_beta_estimated = np.clip(f_beta_estimated, 1000.0, 15000.0)
    else:
        f_beta_estimated = 5000.0
    
    # 振幅を推定
    alpha_peak_value = psd_smooth[np.argmin(np.abs(frequencies - f_alpha_estimated))]
    beta_peak_value = psd_smooth[np.argmin(np.abs(frequencies - f_beta_estimated))]
    
    A_alpha_estimated = max(alpha_peak_value * 2, psd_max * 0.1)
    A_beta_estimated = max(beta_peak_value * 2, psd_max * 0.1)
    C_estimated = psd_min * 0.5
    
    return A_alpha_estimated, f_alpha_estimated, A_beta_estimated, f_beta_estimated, C_estimated

