"""
差分ベースのフィッティングモジュール
ノイズ除去前後のデータの差分を直接活用して精度を向上
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from typing import Tuple, Optional
import warnings

from activation_energy import psd_model, FREQ_MIN, FREQ_MAX, NUM_POINTS


def fit_with_differential_emphasis(
    psd_data: np.ndarray,
    reference_psd: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float]:
    """
    参照データとの差分を強調してフィッティング
    
    Args:
        psd_data: 現在のPSDデータ
        reference_psd: 参照PSDデータ（ノイズ除去前後の比較用）
        frequencies: 周波数配列
    
    Returns:
        (A_alpha, f_alpha, A_beta, f_beta, C) のタプル
    """
    if frequencies is None:
        frequencies = np.linspace(FREQ_MIN, FREQ_MAX, NUM_POINTS)
    
    # 差分を計算
    diff = psd_data - reference_psd
    diff_ratio = np.mean(np.abs(diff)) / (np.mean(np.abs(psd_data)) + 1e-10)
    
    # 差分が大きい領域を強調
    if diff_ratio > 0.05:  # 5%以上の差分がある場合
        # 差分を強調したデータを作成
        enhancement_factor = 2.0  # 差分を2倍に強調
        emphasized_data = reference_psd + diff * enhancement_factor
        
        # 強調したデータでフィッティング
        psd_max = np.max(emphasized_data)
        psd_min = np.min(emphasized_data)
    else:
        # 差分が小さい場合は通常のデータを使用
        emphasized_data = psd_data
        psd_max = np.max(psd_data)
        psd_min = np.min(psd_data)
    
    # 初期パラメータ推定
    from activation_energy import estimate_initial_parameters_old
    
    A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init = estimate_initial_parameters_old(
        emphasized_data, frequencies
    )
    
    # 参照データからも初期パラメータを推定
    A_alpha_ref, f_alpha_ref, A_beta_ref, f_beta_ref, C_ref = estimate_initial_parameters_old(
        reference_psd, frequencies
    )
    
    # データのハッシュ値からシードを生成（データが異なれば異なる結果）
    data_hash = hash(emphasized_data.tobytes()) % (2**31)
    np.random.seed(data_hash)
    
    # 差分に応じて初期値を調整
    if diff_ratio > 0.1:  # 10%以上の差分がある場合
        # 参照データの情報を重み付け
        weight_current = 0.7
        weight_ref = 0.3
        
        f_alpha_init = f_alpha_init * weight_current + f_alpha_ref * weight_ref
        f_beta_init = f_beta_init * weight_current + f_beta_ref * weight_ref
    
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
    
    # 複数の初期値から試す
    initial_guesses = [
        [A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init],
        [A_alpha_ref, f_alpha_ref, A_beta_ref, f_beta_ref, C_ref],
        [(A_alpha_init + A_alpha_ref) / 2, (f_alpha_init + f_alpha_ref) / 2,
         (A_beta_init + A_beta_ref) / 2, (f_beta_init + f_beta_ref) / 2, (C_init + C_ref) / 2],
    ]
    
    # データ依存のランダムな初期値
    for _ in range(15):
        noise_factor = np.random.uniform(0.5, 2.0)
        f_noise_factor = np.random.uniform(0.3, 3.0)
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
                emphasized_data,
                p0=p0,
                bounds=bounds,
                maxfev=100000,
                method='trf',
                ftol=1e-12,
                xtol=1e-12,
            )
            
            psd_predicted = psd_model(frequencies, *popt)
            mse = np.mean((emphasized_data - psd_predicted) ** 2)
            mae = np.mean(np.abs(emphasized_data - psd_predicted))
            
            # 参照データとの一貫性も考慮
            ref_predicted = psd_model(frequencies, *popt)
            ref_mse = np.mean((reference_psd - ref_predicted) ** 2)
            
            # 重み付きエラー
            error = mse + mae * np.mean(emphasized_data) * 0.1 + ref_mse * 0.05
            
            if error < best_error:
                best_error = error
                best_result = popt
        except Exception:
            continue
    
    if best_result is not None:
        return tuple(best_result)
    else:
        return A_alpha_init, f_alpha_init, A_beta_init, f_beta_init, C_init

