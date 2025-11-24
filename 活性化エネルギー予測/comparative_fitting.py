"""
比較的フィッティングモジュール
ノイズ除去前後のデータを比較して最適なフィッティングを選択
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Optional
import warnings

from activation_energy import psd_model, FREQ_MIN, FREQ_MAX, NUM_POINTS


def fit_with_comparison(
    psd_data: np.ndarray,
    reference_psd: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float]:
    """
    参照データと比較して最適なフィッティングを選択
    
    Args:
        psd_data: 現在のPSDデータ
        reference_psd: 参照PSDデータ（ノイズ除去前後の比較用）
        frequencies: 周波数配列
    
    Returns:
        (A_alpha, f_alpha, A_beta, f_beta, C) のタプル
    """
    if frequencies is None:
        frequencies = np.linspace(FREQ_MIN, FREQ_MAX, NUM_POINTS)
    
    psd_max = np.max(psd_data)
    psd_min = np.min(psd_data)
    
    # 両方のデータから初期パラメータを推定
    from improved_fitting import estimate_initial_parameters_improved
    
    A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init = estimate_initial_parameters_improved(
        psd_data, frequencies
    )
    
    A_alpha_ref, f_alpha_ref, A_beta_ref, f_beta_ref, C_ref = estimate_initial_parameters_improved(
        reference_psd, frequencies
    )
    
    # データの差分を計算
    data_diff = np.mean(np.abs(psd_data - reference_psd))
    data_diff_ratio = data_diff / (np.mean(psd_data) + 1e-10)
    
    # 差分が大きい場合は、参照データの情報をより重視
    if data_diff_ratio > 0.1:  # 10%以上の差分がある場合
        # 参照データの初期値を重み付け
        weight_current = 0.6
        weight_ref = 0.4
        
        f_alpha_init = f_alpha_init * weight_current + f_alpha_ref * weight_ref
        f_beta_init = f_beta_init * weight_current + f_beta_ref * weight_ref
        A_alpha_init = A_alpha_init * weight_current + A_alpha_ref * weight_ref
        A_beta_init = A_beta_init * weight_current + A_beta_ref * weight_ref
    
    # 境界値
    f_alpha_lower = max(1.0, min(f_alpha_init, f_alpha_ref) * 0.1)
    f_alpha_upper = min(FREQ_MAX, max(f_alpha_init, f_alpha_ref) * 10.0)
    f_beta_lower = max(10.0, min(f_beta_init, f_beta_ref) * 0.1)
    f_beta_upper = min(FREQ_MAX, max(f_beta_init, f_beta_ref) * 10.0)
    
    bounds = (
        [0, f_alpha_lower, 0, f_beta_lower, 0],
        [psd_max * 500, f_alpha_upper, psd_max * 500, f_beta_upper, psd_max * 50],
    )
    
    best_result = None
    best_error = np.inf
    
    # 現在のデータと参照データの両方から初期値を生成
    initial_guesses = [
        [A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init],
        [A_alpha_ref, f_alpha_ref, A_beta_ref, f_beta_ref, C_ref],
        [(A_alpha_init + A_alpha_ref) / 2, (f_alpha_init + f_alpha_ref) / 2,
         (A_beta_init + A_beta_ref) / 2, (f_beta_init + f_beta_ref) / 2, (C_init + C_ref) / 2],
        [A_alpha_init * 0.5, f_alpha_init * 0.5, A_beta_init * 0.5, f_beta_init * 0.5, C_init],
        [A_alpha_init * 2.0, f_alpha_init * 2.0, A_beta_init * 2.0, f_beta_init * 2.0, C_init],
    ]
    
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
            
            psd_predicted = psd_model(frequencies, *popt)
            mse = np.mean((psd_data - psd_predicted) ** 2)
            mae = np.mean(np.abs(psd_data - psd_predicted))
            
            # 参照データとの一貫性も考慮
            ref_predicted = psd_model(frequencies, *popt)
            ref_mse = np.mean((reference_psd - ref_predicted) ** 2)
            
            # 重み付きエラー（現在のデータの誤差 + 参照データとの一貫性）
            error = mse + mae * np.mean(psd_data) * 0.1 + ref_mse * 0.1
            
            if error < best_error:
                best_error = error
                best_result = popt
        except Exception:
            continue
    
    if best_result is not None:
        return tuple(best_result)
    else:
        return A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init

