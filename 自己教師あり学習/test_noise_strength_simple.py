"""
ノイズ強度の計算を簡易的にテスト
実際のデータセットからサンプルを取得して確認
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

def test_noise_strength_simple():
    """ノイズ強度の計算を簡易的にテスト"""
    print("=" * 80)
    print("ノイズ強度計算の簡易テスト")
    print("=" * 80)
    
    # データセットを作成
    pickle_path = os.path.join(project_root, 'data_lowF_noise.pickle')
    dataset = Task4Dataset(
        pickle_path=pickle_path,
        num_intervals=30,
        noise_type='power_supply',
        use_random_noise=False,
        noise_level=0.3,
        add_structured_noise_flag=True,
    )
    
    # 複数のサンプルをテスト
    num_test_samples = 3
    print(f"\n{num_test_samples}個のサンプルをテストします...\n")
    
    for i in range(min(num_test_samples, len(dataset))):
        print(f"\n{'=' * 80}")
        print(f"サンプル {i+1}/{num_test_samples}")
        print(f"{'=' * 80}")
        
        sample = dataset[i]
        
        noise_strength = sample['noise_strength'].numpy()  # (30,)
        noise_interval = sample['noise_interval'].item()
        
        print(f"\nノイズ区間インデックス: {noise_interval}")
        print(f"ノイズ強度の形状: {noise_strength.shape}")
        print(f"ノイズ強度の合計: {noise_strength.sum():.6f}")
        print(f"ノイズ強度の最小値: {noise_strength.min():.6f}")
        print(f"ノイズ強度の最大値: {noise_strength.max():.6f}")
        print(f"ノイズ強度の平均値: {noise_strength.mean():.6f}")
        print(f"ノイズ強度の標準偏差: {noise_strength.std():.6f}")
        
        # ノイズ区間のノイズ強度
        noise_interval_strength = noise_strength[noise_interval]
        print(f"\nノイズ区間({noise_interval})のノイズ強度: {noise_interval_strength:.6f}")
        
        # ノイズ強度が高い区間トップ5
        top5_indices = np.argsort(noise_strength)[::-1][:5]
        print(f"\nノイズ強度が高い区間トップ5:")
        for rank, idx in enumerate(top5_indices, 1):
            marker = " ← ノイズ区間" if idx == noise_interval else ""
            print(f"  {rank}. 区間{idx}: {noise_strength[idx]:.6f}{marker}")
        
        # 全ての区間で同じ値になっていないか確認
        if noise_strength.std() < 1e-6:
            print(f"\n❌ 警告: ノイズ強度の標準偏差が非常に小さい（{noise_strength.std():.6f}）")
            print(f"   全ての区間でほぼ同じ値になっています")
            print(f"   これは、ノイズが全体に均等に広がっているか、計算方法に問題がある可能性があります")
        else:
            print(f"\n✅ ノイズ強度にばらつきがあります（標準偏差: {noise_strength.std():.6f}）")
            # ノイズ区間がトップ5に入っているか確認
            if noise_interval in top5_indices:
                rank = list(top5_indices).index(noise_interval) + 1
                print(f"✅ ノイズ区間({noise_interval})はノイズ強度が高い区間の{rank}位です")
            else:
                print(f"⚠️ ノイズ区間({noise_interval})はノイズ強度が高い区間トップ5に入っていません")
    
    print(f"\n{'=' * 80}")
    print("テスト完了")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    test_noise_strength_simple()

