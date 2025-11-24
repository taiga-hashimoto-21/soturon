"""
アンサンブル予測モジュール
複数のフィッティング結果を組み合わせて精度を向上
"""

import numpy as np
from typing import List, Tuple, Optional
from activation_energy import (
    calculate_activation_energy_from_tau,
    SIGMA_ALPHA_0, SIGMA_BETA_0, NT_MEAN, V,
    E_ALPHA_MIN, E_ALPHA_MAX, E_BETA_MIN, E_BETA_MAX,
)


def ensemble_activation_energy_prediction(
    f_alpha_values: List[float],
    f_beta_values: List[float],
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    複数のフィッティング結果から活性化エネルギーをアンサンブル予測
    
    Args:
        f_alpha_values: f_alphaの値のリスト
        f_beta_values: f_betaの値のリスト
        weights: 各結果の重み（Noneの場合は均等重み）
    
    Returns:
        (E_alpha, E_beta) のタプル
    """
    if len(f_alpha_values) == 0 or len(f_beta_values) == 0:
        return E_ALPHA_MIN, E_BETA_MIN
    
    if weights is None:
        weights = np.ones(len(f_alpha_values)) / len(f_alpha_values)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()  # 正規化
    
    # 各f_alpha, f_betaから活性化エネルギーを計算
    E_alpha_values = []
    E_beta_values = []
    
    for f_alpha, f_beta in zip(f_alpha_values, f_beta_values):
        tau_alpha = 1.0 / (2 * np.pi * f_alpha) if f_alpha > 0 else np.inf
        tau_beta = 1.0 / (2 * np.pi * f_beta) if f_beta > 0 else np.inf
        
        E_alpha = calculate_activation_energy_from_tau(tau_alpha, SIGMA_ALPHA_0, NT_MEAN, V)
        E_beta = calculate_activation_energy_from_tau(tau_beta, SIGMA_BETA_0, NT_MEAN, V)
        
        # 範囲外の値は除外
        if E_ALPHA_MIN <= E_alpha <= E_ALPHA_MAX:
            E_alpha_values.append(E_alpha)
        if E_BETA_MIN <= E_beta <= E_BETA_MAX:
            E_beta_values.append(E_beta)
    
    if len(E_alpha_values) == 0:
        E_alpha = E_ALPHA_MIN
    else:
        # 重み付き平均
        E_alpha = np.average(E_alpha_values, weights=weights[:len(E_alpha_values)])
        E_alpha = np.clip(E_alpha, E_ALPHA_MIN, E_ALPHA_MAX)
    
    if len(E_beta_values) == 0:
        E_beta = E_BETA_MIN
    else:
        # 重み付き平均
        E_beta = np.average(E_beta_values, weights=weights[:len(E_beta_values)])
        E_beta = np.clip(E_beta, E_BETA_MIN, E_BETA_MAX)
    
    return E_alpha, E_beta

