"""
ノイズ強度の計算をデバッグするスクリプト
実際のデータを見て、どこで問題が起きているか確認
"""
import sys
import os
import numpy as np
import torch

# パスを追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '自己教師あり学習'))

# ノイズモジュールを登録
noise_folder_path = os.path.join(project_root, 'ノイズの付与(共通)')
if noise_folder_path not in sys.path:
    sys.path.insert(0, noise_folder_path)

import importlib.util
init_path = os.path.join(noise_folder_path, '__init__.py')
init_spec = importlib.util.spec_from_file_location("noise", init_path)
init_module = importlib.util.module_from_spec(init_spec)
sys.modules['noise'] = init_module
init_module.__file__ = init_path
init_module.__path__ = [noise_folder_path]
init_spec.loader.exec_module(init_module)

add_noise_path = os.path.join(noise_folder_path, 'add_noise.py')
add_noise_spec = importlib.util.spec_from_file_location("noise.add_noise", add_noise_path)
add_noise_module = importlib.util.module_from_spec(add_noise_spec)
sys.modules['noise.add_noise'] = add_noise_module
add_noise_module.__file__ = add_noise_path
add_noise_spec.loader.exec_module(add_noise_module)
sys.modules['add_noise'] = add_noise_module

from add_noise import add_noise_to_interval
from task.dataset import add_structured_noise

def debug_noise_calculation():
    """ノイズ強度の計算をデバッグ"""
    print("=" * 80)
    print("ノイズ強度計算のデバッグ")
    print("=" * 80)
    
    # データセットから直接取得（dataset.pyと同じ方法）
    from task.dataset import Task4Dataset
    pickle_path = os.path.join(project_root, 'data_lowF_noise.pickle')
    
    dataset = Task4Dataset(
        pickle_path=pickle_path,
        num_intervals=30,
        noise_type='power_supply',
        use_random_noise=False,
        noise_level=0.3,
        add_structured_noise_flag=True,
    )
    
    # データセットの内部データを確認
    print(f"\nデータセットの内部データ形状: {dataset.x.shape}")
    print(f"データセットの内部データ統計:")
    print(f"  平均: {dataset.x.mean():.6f}")
    print(f"  標準偏差: {dataset.x.std():.6f}")
    print(f"  最小値: {dataset.x.min():.6f}")
    print(f"  最大値: {dataset.x.max():.6f}")
    
    # 1つのサンプルを取得（dataset.pyと同じ方法）
    idx = 0
    base_psd = dataset.x[idx].astype(np.float32)
    base_psd = np.nan_to_num(base_psd, nan=0.0, posinf=0.0, neginf=0.0)
    L = base_psd.shape[0]
    
    print(f"\nbase_psdの統計:")
    print(f"  形状: {base_psd.shape}")
    print(f"  平均: {base_psd.mean():.6f}")
    print(f"  標準偏差: {base_psd.std():.6f}")
    print(f"  最小値: {base_psd.min():.6f}")
    print(f"  最大値: {base_psd.max():.6f}")
    
    # 構造化ノイズを付与（dataset.pyと同じ処理）
    from task.dataset import add_structured_noise
    base_psd_tensor = torch.from_numpy(base_psd)
    structured_noisy_psd = add_structured_noise(
        base_psd_tensor,
        clip_range=0.5,
        smoothing_factor=0.1
    )
    
    print(f"\nstructured_noisy_psdの統計:")
    print(f"  平均: {structured_noisy_psd.numpy().mean():.6f}")
    print(f"  標準偏差: {structured_noisy_psd.numpy().std():.6f}")
    
    # 構造化ノイズの差分
    structured_noise_diff = structured_noisy_psd.numpy() - base_psd
    print(f"\n構造化ノイズの差分:")
    print(f"  平均: {structured_noise_diff.mean():.6f}")
    print(f"  標準偏差: {structured_noise_diff.std():.6f}")
    print(f"  絶対値の平均: {np.abs(structured_noise_diff).mean():.6f}")
    
    # 区間ノイズを付与（power_supply、区間4に相当する周波数）
    noise_interval = 4
    num_intervals = 30
    points_per_interval = L // num_intervals
    
    # 周波数から区間インデックスを計算（dataset.pyと同じ）
    freq_min = 0.0
    freq_max = 15000.0
    main_freq = 2000.0  # power_supplyの主要周波数
    freq_to_point = main_freq / (freq_max / L)
    point_idx = int(freq_to_point)
    calculated_noise_interval = min(point_idx // points_per_interval, num_intervals - 1)
    
    print(f"\nノイズ区間の計算:")
    print(f"  主要周波数: {main_freq} Hz")
    print(f"  ポイントインデックス: {point_idx}")
    print(f"  計算されたノイズ区間: {calculated_noise_interval}")
    print(f"  指定されたノイズ区間: {noise_interval}")
    
    # 区間ノイズを付与
    noisy_psd_tensor, start_idx, end_idx = add_noise_to_interval(
        structured_noisy_psd,
        noise_interval,
        noise_type='power_supply',
        noise_level=0.3,
        num_intervals=num_intervals
    )
    noisy_psd = noisy_psd_tensor.numpy()
    
    print(f"\nnoisy_psdの統計:")
    print(f"  平均: {noisy_psd.mean():.6f}")
    print(f"  標準偏差: {noisy_psd.std():.6f}")
    
    # ノイズ強度の計算（修正前: structured_noisy_psdとの差分）
    noise_diff_old = noisy_psd - structured_noisy_psd.numpy()
    print(f"\n【修正前】noisy_psd - structured_noisy_psd:")
    print(f"  平均: {noise_diff_old.mean():.6f}")
    print(f"  標準偏差: {noise_diff_old.std():.6f}")
    print(f"  絶対値の平均: {np.abs(noise_diff_old).mean():.6f}")
    
    # ノイズ強度の計算（修正後: base_psdとの差分）
    noise_diff_new = noisy_psd - base_psd
    print(f"\n【修正後】noisy_psd - base_psd:")
    print(f"  平均: {noise_diff_new.mean():.6f}")
    print(f"  標準偏差: {noise_diff_new.std():.6f}")
    print(f"  絶対値の平均: {np.abs(noise_diff_new).mean():.6f}")
    
    # 各区間のノイズ強度を計算（修正前）
    noise_strength_per_interval_old = []
    for i in range(num_intervals):
        start_idx = i * points_per_interval
        end_idx = min(start_idx + points_per_interval, L)
        interval_noise = noise_diff_old[start_idx:end_idx]
        noise_strength = np.abs(interval_noise).mean()
        noise_strength_per_interval_old.append(noise_strength)
    
    noise_strength_per_interval_old = np.array(noise_strength_per_interval_old)
    total_strength_old = noise_strength_per_interval_old.sum()
    if total_strength_old > 1e-10:
        normalized_noise_strength_old = noise_strength_per_interval_old / total_strength_old
    else:
        normalized_noise_strength_old = np.ones(num_intervals, dtype=np.float32) / num_intervals
    
    print(f"\n【修正前】ノイズ強度（正規化済み）:")
    print(f"  合計: {normalized_noise_strength_old.sum():.6f}")
    print(f"  最小値: {normalized_noise_strength_old.min():.6f}")
    print(f"  最大値: {normalized_noise_strength_old.max():.6f}")
    print(f"  平均値: {normalized_noise_strength_old.mean():.6f}")
    print(f"  標準偏差: {normalized_noise_strength_old.std():.6f}")
    print(f"  ノイズ区間({noise_interval})の値: {normalized_noise_strength_old[noise_interval]:.6f}")
    
    # 各区間のノイズ強度を計算（修正後）
    noise_strength_per_interval_new = []
    for i in range(num_intervals):
        start_idx = i * points_per_interval
        end_idx = min(start_idx + points_per_interval, L)
        interval_noise = noise_diff_new[start_idx:end_idx]
        noise_strength = np.abs(interval_noise).mean()
        noise_strength_per_interval_new.append(noise_strength)
    
    noise_strength_per_interval_new = np.array(noise_strength_per_interval_new)
    total_strength_new = noise_strength_per_interval_new.sum()
    if total_strength_new > 1e-10:
        normalized_noise_strength_new = noise_strength_per_interval_new / total_strength_new
    else:
        normalized_noise_strength_new = np.ones(num_intervals, dtype=np.float32) / num_intervals
    
    print(f"\n【修正後】ノイズ強度（正規化済み）:")
    print(f"  合計: {normalized_noise_strength_new.sum():.6f}")
    print(f"  最小値: {normalized_noise_strength_new.min():.6f}")
    print(f"  最大値: {normalized_noise_strength_new.max():.6f}")
    print(f"  平均値: {normalized_noise_strength_new.mean():.6f}")
    print(f"  標準偏差: {normalized_noise_strength_new.std():.6f}")
    print(f"  ノイズ区間({noise_interval})の値: {normalized_noise_strength_new[noise_interval]:.6f}")
    
    # ノイズ区間の周波数範囲を確認
    noise_start_idx = noise_interval * points_per_interval
    noise_end_idx = min(noise_start_idx + points_per_interval, L)
    noise_freq_start = freq_min + (noise_start_idx / L) * (freq_max - freq_min)
    noise_freq_end = freq_min + (noise_end_idx / L) * (freq_max - freq_min)
    
    print(f"\nノイズ区間({noise_interval})の周波数範囲:")
    print(f"  {noise_freq_start:.1f} Hz - {noise_freq_end:.1f} Hz")
    print(f"  主要周波数({main_freq} Hz)はこの範囲内: {noise_freq_start <= main_freq <= noise_freq_end}")
    
    # ノイズ差分を可視化（ノイズ区間周辺）
    print(f"\nノイズ差分の詳細（区間{noise_interval-1}から{noise_interval+1}まで）:")
    for i in range(max(0, noise_interval-1), min(num_intervals, noise_interval+2)):
        start_idx = i * points_per_interval
        end_idx = min(start_idx + points_per_interval, L)
        interval_noise_old = noise_diff_old[start_idx:end_idx]
        interval_noise_new = noise_diff_new[start_idx:end_idx]
        print(f"  区間{i}:")
        print(f"    修正前（structured差分）: 平均={np.abs(interval_noise_old).mean():.6f}, 最大={np.abs(interval_noise_old).max():.6f}")
        print(f"    修正後（base差分）: 平均={np.abs(interval_noise_new).mean():.6f}, 最大={np.abs(interval_noise_new).max():.6f}")

if __name__ == "__main__":
    import pickle
    debug_noise_calculation()

