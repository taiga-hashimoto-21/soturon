"""
PSDデータから活性化エネルギーを計算する理論式の実装

理論式:
- 式(1): S_I(f) = (A_α(I_bias)I_bias^2) / (1 + (f/f_α)^2) + (A_β(I_bias)I_bias^2) / (1 + (f/f_β)^2) + C
- 式(2): τ(T) = 1 / (2πf(T)) = 1 / (σ^(0) × exp(-E / (k_B T)) × N_T × ν)
"""

import torch
import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Optional


# 物理定数
KB = 8.6e-5  # eV/K (ボルツマン定数)
T = 300.0  # K (温度)
V = 1e7  # cm/s
SIGMA_ALPHA_0 = 1e-23  # cm²
SIGMA_BETA_0 = SIGMA_ALPHA_0 * 830  # cm²
NT_MIN = (4e11) ** 1.5 * 1.0  # /cm³
NT_MAX = (4e11) ** 1.5 * 1.3  # /cm³
NT_MEAN = (NT_MIN + NT_MAX) / 2.0  # /cm³

# 活性化エネルギーの範囲（meV）
E_ALPHA_MIN = 0.1  # meV
E_ALPHA_MAX = 20.0  # meV
E_BETA_MIN = 12.5  # meV
E_BETA_MAX = 32.5  # meV

# 周波数範囲（Hz）
FREQ_MIN = 0.0  # Hz
FREQ_MAX = 15000.0  # Hz
NUM_POINTS = 3000


def psd_model(f: np.ndarray, A_alpha: float, f_alpha: float, A_beta: float, f_beta: float, C: float) -> np.ndarray:
    """
    PSDの理論式（式(1)）
    
    Args:
        f: 周波数配列 (Hz)
        A_alpha: α成分の振幅係数
        f_alpha: α成分の特徴周波数 (Hz)
        A_beta: β成分の振幅係数
        f_beta: β成分の特徴周波数 (Hz)
        C: 定数項
    
    Returns:
        PSD値の配列
    """
    term_alpha = A_alpha / (1 + (f / f_alpha) ** 2)
    term_beta = A_beta / (1 + (f / f_beta) ** 2)
    return term_alpha + term_beta + C


def calculate_relaxation_time(E: float, sigma_0: float, N_T: float, nu: float) -> float:
    """
    緩和時間を計算（式(2)）
    
    Args:
        E: 活性化エネルギー (meV)
        sigma_0: 断面積の前因子 (cm²)
        N_T: 欠陥密度 (/cm³)
        nu: 速度 (cm/s)
    
    Returns:
        緩和時間 (s)
    """
    # EをeVに変換（meV → eV）
    E_eV = E * 1e-3
    
    # exp(-E / (k_B T))
    exp_term = np.exp(-E_eV / (KB * T))
    
    # τ = 1 / (σ^(0) × exp(-E / (k_B T)) × N_T × ν)
    tau = 1.0 / (sigma_0 * exp_term * N_T * nu)
    
    return tau


def calculate_activation_energy_from_tau(tau: float, sigma_0: float, N_T: float, nu: float) -> float:
    """
    緩和時間から活性化エネルギーを逆算（式(2)の逆変換）
    
    Args:
        tau: 緩和時間 (s)
        sigma_0: 断面積の前因子 (cm²)
        N_T: 欠陥密度 (/cm³)
        nu: 速度 (cm/s)
    
    Returns:
        活性化エネルギー (meV)
    """
    # τ = 1 / (σ^(0) × exp(-E / (k_B T)) × N_T × ν)
    # → exp(-E / (k_B T)) = 1 / (τ × σ^(0) × N_T × ν)
    # → -E / (k_B T) = ln(1 / (τ × σ^(0) × N_T × ν))
    # → E = -k_B T × ln(1 / (τ × σ^(0) × N_T × ν))
    
    denominator = tau * sigma_0 * N_T * nu
    if denominator <= 0:
        return np.nan
    
    E_eV = -KB * T * np.log(1.0 / denominator)
    E_meV = E_eV * 1000.0  # eV → meV
    
    return E_meV


def fit_psd_model(psd_data: np.ndarray, frequencies: Optional[np.ndarray] = None) -> Tuple[float, float, float, float, float]:
    """
    PSDデータに理論式をフィッティングしてパラメータを推定
    
    Args:
        psd_data: PSDデータ (3000,)
        frequencies: 周波数配列 (3000,)（Noneの場合は自動生成）
    
    Returns:
        (A_alpha, f_alpha, A_beta, f_beta, C) のタプル
    """
    if frequencies is None:
        frequencies = np.linspace(FREQ_MIN, FREQ_MAX, NUM_POINTS)
    
    # 初期値の設定
    psd_max = np.max(psd_data)
    psd_min = np.min(psd_data)
    psd_mean = np.mean(psd_data)
    
    # 初期パラメータ
    p0 = [
        psd_max * 0.5,  # A_alpha
        FREQ_MAX * 0.1,  # f_alpha (低周波数側)
        psd_max * 0.5,  # A_beta
        FREQ_MAX * 0.5,  # f_beta (中周波数側)
        psd_min,  # C
    ]
    
    # 境界値
    bounds = (
        [0, FREQ_MIN, 0, FREQ_MIN, 0],  # 下限
        [psd_max * 10, FREQ_MAX, psd_max * 10, FREQ_MAX, psd_max],  # 上限
    )
    
    try:
        # フィッティング
        popt, _ = curve_fit(
            psd_model,
            frequencies,
            psd_data,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
        
        A_alpha, f_alpha, A_beta, f_beta, C = popt
        return A_alpha, f_alpha, A_beta, f_beta, C
    except Exception as e:
        print(f"フィッティングに失敗: {e}")
        # デフォルト値を返す
        return psd_max * 0.5, FREQ_MAX * 0.1, psd_max * 0.5, FREQ_MAX * 0.5, psd_min


def calculate_activation_energy_from_psd(
    psd_data: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
    N_T: Optional[float] = None,
) -> Tuple[float, float]:
    """
    PSDデータから活性化エネルギーを計算
    
    Args:
        psd_data: PSDデータ (3000,) または torch.Tensor
        frequencies: 周波数配列 (3000,)（Noneの場合は自動生成）
        N_T: 欠陥密度 (/cm³)（Noneの場合は平均値を使用）
    
    Returns:
        (E_alpha, E_beta) のタプル（meV）
    """
    # torch.Tensorの場合はnumpyに変換
    if isinstance(psd_data, torch.Tensor):
        psd_data = psd_data.cpu().numpy()
    
    # 1次元配列に変換
    if psd_data.ndim > 1:
        psd_data = psd_data.flatten()
    
    # 周波数配列を生成
    if frequencies is None:
        frequencies = np.linspace(FREQ_MIN, FREQ_MAX, len(psd_data))
    
    # 欠陥密度を設定
    if N_T is None:
        N_T = NT_MEAN
    
    # PSDモデルをフィッティング
    A_alpha, f_alpha, A_beta, f_beta, C = fit_psd_model(psd_data, frequencies)
    
    # 緩和時間を計算（τ = 1 / (2πf)）
    tau_alpha = 1.0 / (2 * np.pi * f_alpha) if f_alpha > 0 else np.inf
    tau_beta = 1.0 / (2 * np.pi * f_beta) if f_beta > 0 else np.inf
    
    # 活性化エネルギーを逆算
    E_alpha = calculate_activation_energy_from_tau(tau_alpha, SIGMA_ALPHA_0, N_T, V)
    E_beta = calculate_activation_energy_from_tau(tau_beta, SIGMA_BETA_0, N_T, V)
    
    # 範囲内にクリップ
    E_alpha = np.clip(E_alpha, E_ALPHA_MIN, E_ALPHA_MAX)
    E_beta = np.clip(E_beta, E_BETA_MIN, E_BETA_MAX)
    
    return E_alpha, E_beta


def convert_to_y_format(E_alpha: float, E_beta: float) -> Tuple[float, float]:
    """
    活性化エネルギーをy形式に変換（meV → y形式: E / 10）
    
    Args:
        E_alpha: 活性化エネルギーα (meV)
        E_beta: 活性化エネルギーβ (meV)
    
    Returns:
        (y_alpha, y_beta) のタプル
    """
    y_alpha = E_alpha / 10.0
    y_beta = E_beta / 10.0
    return y_alpha, y_beta

