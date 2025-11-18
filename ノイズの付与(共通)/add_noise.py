"""
測定系由来のノイズを付与するモジュール
3つのパターンのノイズ生成関数を提供
統合インターフェースとして機能
"""

# 相対インポートを絶対インポートに変更（Colab対応）
from frequency_band_noise import add_frequency_band_noise
from localized_spike_noise import add_localized_spike_noise
from amplitude_dependent_noise import add_amplitude_dependent_noise


def add_noise_to_interval(psd_data, interval_idx, noise_type='frequency_band', **kwargs):
    """
    統一インターフェース: ノイズタイプを指定してノイズを付与
    
    Args:
        psd_data: PSDデータ
        interval_idx: ノイズを加える区間のインデックス
        noise_type: 'frequency_band', 'localized_spike', 'amplitude_dependent' のいずれか
        **kwargs: 各ノイズ関数への追加パラメータ
    
    Returns:
        ノイズを加えたPSDデータ, 開始インデックス, 終了インデックス
    """
    if noise_type == 'frequency_band':
        return add_frequency_band_noise(psd_data, interval_idx, **kwargs)
    elif noise_type == 'localized_spike':
        return add_localized_spike_noise(psd_data, interval_idx, **kwargs)
    elif noise_type == 'amplitude_dependent':
        return add_amplitude_dependent_noise(psd_data, interval_idx, **kwargs)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}. Choose from 'frequency_band', 'localized_spike', 'amplitude_dependent'")
