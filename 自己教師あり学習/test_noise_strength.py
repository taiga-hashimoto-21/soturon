"""
ノイズ強度の計算が正しくできているか確認するスクリプト
"""
import sys
import os
import numpy as np
import torch

# パスを追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '自己教師あり学習'))

# ノイズモジュールを登録（dataset.pyと同じ方法）
noise_folder_path = os.path.join(project_root, 'ノイズの付与(共通)')
if noise_folder_path not in sys.path:
    sys.path.insert(0, noise_folder_path)

# ノイズパッケージの__init__.pyをロード
import importlib.util
init_path = os.path.join(noise_folder_path, '__init__.py')
init_spec = importlib.util.spec_from_file_location("noise", init_path)
init_module = importlib.util.module_from_spec(init_spec)
sys.modules['noise'] = init_module
init_module.__file__ = init_path
init_module.__path__ = [noise_folder_path]
init_spec.loader.exec_module(init_module)

# add_noise.pyをロード
add_noise_path = os.path.join(noise_folder_path, 'add_noise.py')
add_noise_spec = importlib.util.spec_from_file_location("noise.add_noise", add_noise_path)
add_noise_module = importlib.util.module_from_spec(add_noise_spec)
sys.modules['noise.add_noise'] = add_noise_module
add_noise_module.__file__ = add_noise_path
add_noise_spec.loader.exec_module(add_noise_module)

# add_noiseモジュールを直接登録（task/dataset.pyが`from add_noise import`を使うため）
sys.modules['add_noise'] = add_noise_module

from task.dataset import Task4Dataset

def test_noise_strength():
    """ノイズ強度の計算をテスト"""
    print("=" * 80)
    print("ノイズ強度の計算テスト")
    print("=" * 80)
    
    # データセットを作成
    pickle_path = os.path.join(project_root, 'data_lowF_noise.pickle')
    if not os.path.exists(pickle_path):
        print(f"❌ データファイルが見つかりません: {pickle_path}")
        return
    
    dataset = Task4Dataset(
        pickle_path=pickle_path,
        num_intervals=30,
        noise_type='power_supply',
        use_random_noise=False,  # 固定のノイズタイプでテスト
        noise_level=0.3,
        add_structured_noise_flag=True,
    )
    
    print(f"\nデータセットサイズ: {len(dataset)}")
    
    # 複数のサンプルをテスト
    num_test_samples = 5
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
        print(f"ノイズ強度の合計: {noise_strength.sum():.6f} (正規化されているので1.0に近いはず)")
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
            print(f"  {rank}. 区間{idx}: {noise_strength[idx]:.6f} {'(ノイズ区間)' if idx == noise_interval else ''}")
        
        # ノイズ強度の逆数（目標アテンションウェイト）
        target_attention = 1.0 / (noise_strength + 1e-6)  # ゼロ除算を防ぐ
        target_attention_normalized = target_attention / target_attention.sum()  # 正規化
        
        print(f"\n目標アテンションウェイト（1/ノイズ強度、正規化済み）:")
        print(f"  ノイズ区間({noise_interval}): {target_attention_normalized[noise_interval]:.6f}")
        print(f"  平均値: {target_attention_normalized.mean():.6f}")
        print(f"  標準偏差: {target_attention_normalized.std():.6f}")
        
        # ノイズ区間の目標アテンションが小さいか確認
        if noise_interval_strength > noise_strength.mean():
            print(f"\n✅ ノイズ区間のノイズ強度が平均より高い（{noise_interval_strength:.6f} > {noise_strength.mean():.6f}）")
            print(f"✅ ノイズ区間の目標アテンションが平均より小さい（{target_attention_normalized[noise_interval]:.6f} < {target_attention_normalized.mean():.6f}）")
        else:
            print(f"\n⚠️ ノイズ区間のノイズ強度が平均より低い（{noise_interval_strength:.6f} < {noise_strength.mean():.6f}）")
            print(f"⚠️ ノイズ区間の目標アテンションが平均より大きい（{target_attention_normalized[noise_interval]:.6f} > {target_attention_normalized.mean():.6f}）")
        
        # ノイズ強度の分布を確認
        print(f"\nノイズ強度の分布:")
        print(f"  0.0-0.01: {(noise_strength < 0.01).sum()}区間")
        print(f"  0.01-0.05: {((noise_strength >= 0.01) & (noise_strength < 0.05)).sum()}区間")
        print(f"  0.05-0.1: {((noise_strength >= 0.05) & (noise_strength < 0.1)).sum()}区間")
        print(f"  0.1以上: {(noise_strength >= 0.1).sum()}区間")
        
        # 全ての区間で同じ値になっていないか確認
        if noise_strength.std() < 1e-6:
            print(f"\n❌ 警告: ノイズ強度の標準偏差が非常に小さい（{noise_strength.std():.6f}）")
            print(f"   全ての区間でほぼ同じ値になっている可能性があります")
        else:
            print(f"\n✅ ノイズ強度にばらつきがあります（標準偏差: {noise_strength.std():.6f}）")
    
    print(f"\n{'=' * 80}")
    print("テスト完了")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    test_noise_strength()

