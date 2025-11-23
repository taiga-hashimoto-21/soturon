"""
3種類のノイズタイプそれぞれでノイズ強度をテスト
"""
import sys
import os
import numpy as np

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

from task.dataset import Task4Dataset

def test_all_noise_types():
    """3種類のノイズタイプそれぞれでノイズ強度をテスト"""
    print("=" * 80)
    print("3種類のノイズタイプでのノイズ強度テスト")
    print("=" * 80)
    
    pickle_path = os.path.join(project_root, 'data_lowF_noise.pickle')
    
    # 3種類のノイズタイプをテスト
    noise_types = ['power_supply', 'interference', 'clock_leakage']
    
    for noise_type in noise_types:
        print(f"\n{'=' * 80}")
        print(f"ノイズタイプ: {noise_type}")
        print(f"{'=' * 80}")
        
        dataset = Task4Dataset(
            pickle_path=pickle_path,
            num_intervals=30,
            noise_type=noise_type,
            use_random_noise=False,  # 固定のノイズタイプでテスト
            noise_level=0.3,
            add_structured_noise_flag=True,
        )
        
        # 複数のサンプルをテスト
        num_test_samples = 3
        print(f"\n{num_test_samples}個のサンプルをテストします...\n")
        
        for i in range(min(num_test_samples, len(dataset))):
            sample = dataset[i]
            
            noise_strength = sample['noise_strength'].numpy()  # (30,)
            noise_interval = sample['noise_interval'].item()
            
            print(f"\n  サンプル {i+1}:")
            print(f"    ノイズ区間インデックス: {noise_interval}")
            print(f"    ノイズ強度の最小値: {noise_strength.min():.6f}")
            print(f"    ノイズ強度の最大値: {noise_strength.max():.6f}")
            print(f"    ノイズ強度の平均値: {noise_strength.mean():.6f}")
            print(f"    ノイズ強度の標準偏差: {noise_strength.std():.6f}")
            print(f"    ノイズ区間({noise_interval})のノイズ強度: {noise_strength[noise_interval]:.6f}")
            
            # ノイズ強度が高い区間トップ5
            top5_indices = np.argsort(noise_strength)[::-1][:5]
            print(f"    ノイズ強度が高い区間トップ5:")
            for rank, idx in enumerate(top5_indices, 1):
                marker = " ← ノイズ区間" if idx == noise_interval else ""
                print(f"      {rank}. 区間{idx}: {noise_strength[idx]:.6f}{marker}")
            
            # 全ての区間で同じ値になっていないか確認
            if noise_strength.std() < 1e-6:
                print(f"    ❌ 警告: ノイズ強度の標準偏差が非常に小さい（{noise_strength.std():.6f}）")
            else:
                print(f"    ✅ ノイズ強度にばらつきがあります（標準偏差: {noise_strength.std():.6f}）")
                # ノイズ区間がトップ5に入っているか確認
                if noise_interval in top5_indices:
                    rank = list(top5_indices).index(noise_interval) + 1
                    print(f"    ✅ ノイズ区間({noise_interval})はノイズ強度が高い区間の{rank}位です")
                else:
                    print(f"    ⚠️ ノイズ区間({noise_interval})はノイズ強度が高い区間トップ5に入っていません")
        
        # ノイズタイプに応じた期待されるノイズ区間を確認
        if noise_type == 'power_supply':
            expected_interval = 4  # 2000Hz → 区間4
        elif noise_type == 'interference':
            expected_interval = 6  # 3000Hz → 区間6
        elif noise_type == 'clock_leakage':
            expected_interval = 10  # 5000Hz → 区間10
        
        print(f"\n  期待されるノイズ区間: {expected_interval}")
        print(f"  実際のノイズ区間: {sample['noise_interval'].item()}")
        if sample['noise_interval'].item() == expected_interval:
            print(f"  ✅ ノイズ区間が正しく計算されています")
        else:
            print(f"  ⚠️ ノイズ区間が期待値と異なります")
    
    # ランダムノイズの場合もテスト
    print(f"\n{'=' * 80}")
    print(f"ランダムノイズ（use_random_noise=True）")
    print(f"{'=' * 80}")
    
    dataset_random = Task4Dataset(
        pickle_path=pickle_path,
        num_intervals=30,
        noise_type='power_supply',  # use_random_noise=Trueの場合は無視される
        use_random_noise=True,  # 3種類のノイズをランダムに使用
        noise_level=0.3,
        add_structured_noise_flag=True,
    )
    
    print(f"\n10個のサンプルをテストします...\n")
    noise_type_counts = {'power_supply': 0, 'interference': 0, 'clock_leakage': 0}
    
    for i in range(min(10, len(dataset_random))):
        sample = dataset_random[i]
        noise_strength = sample['noise_strength'].numpy()
        noise_interval = sample['noise_interval'].item()
        
        # ノイズ区間からノイズタイプを推測
        if noise_interval == 4:
            inferred_type = 'power_supply'
        elif noise_interval == 6:
            inferred_type = 'interference'
        elif noise_interval == 10:
            inferred_type = 'clock_leakage'
        else:
            inferred_type = 'unknown'
        
        if inferred_type in noise_type_counts:
            noise_type_counts[inferred_type] += 1
        
        print(f"  サンプル {i+1}: ノイズ区間={noise_interval} (推測: {inferred_type}), "
              f"ノイズ強度の標準偏差={noise_strength.std():.6f}")
    
    print(f"\n  ノイズタイプの分布:")
    for noise_type, count in noise_type_counts.items():
        print(f"    {noise_type}: {count}回")
    
    print(f"\n{'=' * 80}")
    print("テスト完了")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    test_all_noise_types()

