"""
実験者のスキルに依存するノイズを付与するモジュール
3つのパターンのノイズ生成関数を提供
統合インターフェースとして機能
"""

# 相対インポート（同じディレクトリ内のモジュール）
from .power_supply_noise import add_power_supply_noise
from .interference_noise import add_interference_noise
from .clock_leakage_noise import add_clock_leakage_noise


def add_noise_to_interval(psd_data, interval_idx, noise_type='power_supply', **kwargs):
    """
    統一インターフェース: ノイズタイプを指定してノイズを付与
    
    Args:
        psd_data: PSDデータ
        interval_idx: ノイズを加える区間のインデックス（新しいノイズは全体に適用されるため無視される）
        noise_type: 'power_supply', 'interference', 'clock_leakage' のいずれか
        **kwargs: 各ノイズ関数への追加パラメータ
    
    Returns:
        ノイズを加えたPSDデータ, 開始インデックス, 終了インデックス
    """
    if noise_type == 'power_supply':
        return add_power_supply_noise(psd_data, interval_idx, **kwargs)
    elif noise_type == 'interference':
        return add_interference_noise(psd_data, interval_idx, **kwargs)
    elif noise_type == 'clock_leakage':
        return add_clock_leakage_noise(psd_data, interval_idx, **kwargs)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}. Choose from 'power_supply', 'interference', 'clock_leakage'")
