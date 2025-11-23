"""
ノイズが実際に付与されているか確認するスクリプト
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
from task.dataset import Task4Dataset, add_structured_noise

def debug_noise_addition():
    """ノイズが実際に付与されているか確認"""
    print("=" * 80)
    print("ノイズ付与のデバッグ")
    print("=" * 80)
    
    pickle_path = os.path.join(project_root, 'data_lowF_noise.pickle')
    
    # データセットを作成
    dataset = Task4Dataset(
        pickle_path=pickle_path,
        num_intervals=30,
        noise_type='power_supply',
        use_random_noise=False,
        noise_level=0.3,
        add_structured_noise_flag=True,
    )
    
    # 1つのサンプルを取得
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
    
    # 構造化ノイズを付与
    base_psd_tensor = torch.from_numpy(base_psd)
    structured_noisy_psd = add_structured_noise(
        base_psd_tensor,
        clip_range=0.5,
        smoothing_factor=0.1
    )
    
    print(f"\nstructured_noisy_psdの統計:")
    print(f"  平均: {structured_noisy_psd.numpy().mean():.6f}")
    print(f"  標準偏差: {structured_noisy_psd.numpy().std():.6f}")
    print(f"  最小値: {structured_noisy_psd.numpy().min():.6f}")
    print(f"  最大値: {structured_noisy_psd.numpy().max():.6f}")
    
    # スケールファクターを適用
    scale_factor = dataset.scale_factor
    print(f"\nスケールファクター: {scale_factor}")
    
    scaled_structured_noisy_psd = structured_noisy_psd * scale_factor
    print(f"\nscaled_structured_noisy_psdの統計:")
    print(f"  平均: {scaled_structured_noisy_psd.numpy().mean():.6f}")
    print(f"  標準偏差: {scaled_structured_noisy_psd.numpy().std():.6f}")
    
    # ノイズ区間を計算
    noise_interval = 4  # power_supplyの場合
    noise_level = 0.3
    
    # ノイズを付与（スケールされたデータで）
    print(f"\nノイズを付与中...")
    print(f"  ノイズタイプ: power_supply")
    print(f"  ノイズ区間: {noise_interval}")
    print(f"  ノイズレベル: {noise_level}")
    print(f"  入力データの平均: {scaled_structured_noisy_psd.mean():.6f}")
    
    noisy_psd_tensor, start_idx, end_idx = add_noise_to_interval(
        scaled_structured_noisy_psd,
        noise_interval,
        noise_type='power_supply',
        noise_level=noise_level,
        num_intervals=30
    )
    
    # スケールを戻す
    noisy_psd_tensor = noisy_psd_tensor / scale_factor
    noisy_psd = noisy_psd_tensor.numpy()
    
    print(f"\nnoisy_psdの統計:")
    print(f"  平均: {noisy_psd.mean():.6f}")
    print(f"  標準偏差: {noisy_psd.std():.6f}")
    print(f"  最小値: {noisy_psd.min():.6f}")
    print(f"  最大値: {noisy_psd.max():.6f}")
    
    # ノイズ差分を計算
    noise_diff = noisy_psd - structured_noisy_psd.numpy()
    print(f"\nノイズ差分（noisy_psd - structured_noisy_psd）の統計:")
    print(f"  平均: {noise_diff.mean():.6f}")
    print(f"  標準偏差: {noise_diff.std():.6f}")
    print(f"  最小値: {noise_diff.min():.6f}")
    print(f"  最大値: {noise_diff.max():.6f}")
    print(f"  絶対値の平均: {np.abs(noise_diff).mean():.6f}")
    print(f"  絶対値の最大値: {np.abs(noise_diff).max():.6f}")
    
    # ノイズ区間（区間4）のノイズ差分を確認
    points_per_interval = L // 30
    noise_start_idx = noise_interval * points_per_interval
    noise_end_idx = min(noise_start_idx + points_per_interval, L)
    
    noise_interval_diff = noise_diff[noise_start_idx:noise_end_idx]
    print(f"\nノイズ区間({noise_interval})のノイズ差分:")
    print(f"  範囲: インデックス{noise_start_idx}から{noise_end_idx}")
    print(f"  平均: {noise_interval_diff.mean():.6f}")
    print(f"  標準偏差: {noise_interval_diff.std():.6f}")
    print(f"  絶対値の平均: {np.abs(noise_interval_diff).mean():.6f}")
    print(f"  絶対値の最大値: {np.abs(noise_interval_diff).max():.6f}")
    
    # 他の区間との比較
    other_intervals = [0, 10, 20]
    print(f"\n他の区間との比較:")
    for other_interval in other_intervals:
        other_start_idx = other_interval * points_per_interval
        other_end_idx = min(other_start_idx + points_per_interval, L)
        other_interval_diff = noise_diff[other_start_idx:other_end_idx]
        print(f"  区間{other_interval}: 絶対値の平均={np.abs(other_interval_diff).mean():.6f}, "
              f"最大値={np.abs(other_interval_diff).max():.6f}")
    
    # ノイズ強度を計算
    noise_strength_per_interval = []
    for i in range(30):
        start_idx = i * points_per_interval
        end_idx = min(start_idx + points_per_interval, L)
        interval_noise = noise_diff[start_idx:end_idx]
        noise_strength = np.abs(interval_noise).mean()
        noise_strength_per_interval.append(noise_strength)
    
    noise_strength_per_interval = np.array(noise_strength_per_interval)
    total_strength = noise_strength_per_interval.sum()
    
    print(f"\nノイズ強度（正規化前）:")
    print(f"  合計: {total_strength:.6f}")
    print(f"  最小値: {noise_strength_per_interval.min():.6f}")
    print(f"  最大値: {noise_strength_per_interval.max():.6f}")
    print(f"  平均値: {noise_strength_per_interval.mean():.6f}")
    print(f"  標準偏差: {noise_strength_per_interval.std():.6f}")
    print(f"  ノイズ区間({noise_interval})の値: {noise_strength_per_interval[noise_interval]:.6f}")
    
    if total_strength > 1e-10:
        normalized_noise_strength = noise_strength_per_interval / total_strength
        print(f"\nノイズ強度（正規化後）:")
        print(f"  合計: {normalized_noise_strength.sum():.6f}")
        print(f"  最小値: {normalized_noise_strength.min():.6f}")
        print(f"  最大値: {normalized_noise_strength.max():.6f}")
        print(f"  平均値: {normalized_noise_strength.mean():.6f}")
        print(f"  標準偏差: {normalized_noise_strength.std():.6f}")
        print(f"  ノイズ区間({noise_interval})の値: {normalized_noise_strength[noise_interval]:.6f}")
    else:
        print(f"\n⚠️ ノイズ強度の合計が非常に小さい（{total_strength:.6f}）")
        print(f"   ノイズが検出できていません")

if __name__ == "__main__":
    debug_noise_addition()

