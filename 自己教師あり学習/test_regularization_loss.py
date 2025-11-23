"""
正則化損失の計算をテスト
ノイズタイプによって目標アテンションウェイトが変わるか確認
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

from task.dataset import Task4Dataset

def test_regularization_loss():
    """正則化損失の計算をテスト"""
    print("=" * 80)
    print("正則化損失の計算テスト")
    print("=" * 80)
    
    pickle_path = os.path.join(project_root, 'data_lowF_noise.pickle')
    
    # 3種類のノイズタイプをテスト
    noise_types = ['power_supply', 'interference', 'clock_leakage']
    
    print("\n各ノイズタイプでのノイズ強度と目標アテンションウェイトを確認します...\n")
    
    for noise_type in noise_types:
        print(f"{'=' * 80}")
        print(f"ノイズタイプ: {noise_type}")
        print(f"{'=' * 80}")
        
        dataset = Task4Dataset(
            pickle_path=pickle_path,
            num_intervals=30,
            noise_type=noise_type,
            use_random_noise=False,
            noise_level=0.3,
            add_structured_noise_flag=True,
        )
        
        # 複数のサンプルをテスト
        num_test_samples = 3
        all_noise_strengths = []
        all_target_attentions = []
        
        for i in range(min(num_test_samples, len(dataset))):
            sample = dataset[i]
            noise_strength = sample['noise_strength'].numpy()  # (30,)
            noise_interval = sample['noise_interval'].item()
            
            # 目標アテンションウェイトを計算（1 / ノイズ強度）
            epsilon = 1e-6
            target_attention = 1.0 / (noise_strength + epsilon)
            # 正規化（合計=1）
            target_attention_normalized = target_attention / target_attention.sum()
            
            all_noise_strengths.append(noise_strength)
            all_target_attentions.append(target_attention_normalized)
            
            print(f"\n  サンプル {i+1}:")
            print(f"    ノイズ区間: {noise_interval}")
            print(f"    ノイズ強度:")
            print(f"      最小値: {noise_strength.min():.6f}")
            print(f"      最大値: {noise_strength.max():.6f}")
            print(f"      平均値: {noise_strength.mean():.6f}")
            print(f"      標準偏差: {noise_strength.std():.6f}")
            print(f"      ノイズ区間({noise_interval})の値: {noise_strength[noise_interval]:.6f}")
            print(f"    目標アテンションウェイト（1/ノイズ強度、正規化済み）:")
            print(f"      最小値: {target_attention_normalized.min():.6f}")
            print(f"      最大値: {target_attention_normalized.max():.6f}")
            print(f"      平均値: {target_attention_normalized.mean():.6f}")
            print(f"      標準偏差: {target_attention_normalized.std():.6f}")
            print(f"      ノイズ区間({noise_interval})の値: {target_attention_normalized[noise_interval]:.6f}")
            
            # ノイズ区間の目標アテンションが小さいか確認
            if target_attention_normalized[noise_interval] < target_attention_normalized.mean():
                print(f"    ✅ ノイズ区間の目標アテンションが平均より小さい（正しい）")
            else:
                print(f"    ⚠️ ノイズ区間の目標アテンションが平均より大きい（問題あり）")
        
        # 平均を計算
        avg_noise_strength = np.mean(all_noise_strengths, axis=0)
        avg_target_attention = np.mean(all_target_attentions, axis=0)
        
        # ノイズ区間のインデックスを取得（最初のサンプルから）
        noise_interval_idx = sample['noise_interval'].item()
        
        print(f"\n  平均的なノイズ強度:")
        print(f"    ノイズ区間({noise_interval_idx})の値: {avg_noise_strength[noise_interval_idx]:.6f}")
        print(f"    標準偏差: {avg_noise_strength.std():.6f}")
        print(f"  平均的な目標アテンションウェイト:")
        print(f"    ノイズ区間({noise_interval_idx})の値: {avg_target_attention[noise_interval_idx]:.6f}")
        print(f"    標準偏差: {avg_target_attention.std():.6f}")
    
    # ノイズタイプ間の比較
    print(f"\n{'=' * 80}")
    print("ノイズタイプ間の比較")
    print(f"{'=' * 80}")
    
    noise_type_stats = {}
    for noise_type in noise_types:
        dataset = Task4Dataset(
            pickle_path=pickle_path,
            num_intervals=30,
            noise_type=noise_type,
            use_random_noise=False,
            noise_level=0.3,
            add_structured_noise_flag=True,
        )
        
        sample = dataset[0]
        noise_strength = sample['noise_strength'].numpy()
        noise_interval = sample['noise_interval'].item()
        
        epsilon = 1e-6
        target_attention = 1.0 / (noise_strength + epsilon)
        target_attention_normalized = target_attention / target_attention.sum()
        
        # デバッグ情報
        print(f"\n  デバッグ情報:")
        print(f"    ノイズ区間インデックス: {noise_interval}")
        print(f"    ノイズ強度（ノイズ区間）: {noise_strength[noise_interval]:.6f}")
        print(f"    ノイズ強度の最大値: {noise_strength.max():.6f}")
        print(f"    ノイズ強度が最大の区間: {noise_strength.argmax()}")
        print(f"    1/ノイズ強度（ノイズ区間、正規化前）: {target_attention[noise_interval]:.6f}")
        print(f"    1/ノイズ強度（最大値の区間、正規化前）: {target_attention[noise_strength.argmax()]:.6f}")
        print(f"    目標アテンション（ノイズ区間、正規化後）: {target_attention_normalized[noise_interval]:.6f}")
        print(f"    目標アテンション（最大値の区間、正規化後）: {target_attention_normalized[noise_strength.argmax()]:.6f}")
        print(f"    目標アテンションの合計: {target_attention_normalized.sum():.6f}")
        print(f"    目標アテンションの最小値: {target_attention_normalized.min():.6f}")
        print(f"    目標アテンションの最大値: {target_attention_normalized.max():.6f}")
        print(f"    目標アテンションが最大の区間: {target_attention_normalized.argmax()}")
        
        noise_type_stats[noise_type] = {
            'noise_interval': noise_interval,
            'noise_strength_at_interval': noise_strength[noise_interval],
            'target_attention_at_interval': target_attention_normalized[noise_interval],
            'noise_strength_std': noise_strength.std(),
            'target_attention_std': target_attention_normalized.std(),
        }
    
    print(f"\n各ノイズタイプの統計:")
    for noise_type, stats in noise_type_stats.items():
        print(f"  {noise_type}:")
        print(f"    ノイズ区間: {stats['noise_interval']}")
        print(f"    ノイズ区間のノイズ強度: {stats['noise_strength_at_interval']:.6f}")
        print(f"    ノイズ区間の目標アテンション: {stats['target_attention_at_interval']:.6f}")
        print(f"    ノイズ強度の標準偏差: {stats['noise_strength_std']:.6f}")
        print(f"    目標アテンションの標準偏差: {stats['target_attention_std']:.6f}")
    
    # ノイズタイプによって目標アテンションが異なるか確認
    print(f"\nノイズタイプ間の違い:")
    power_supply_attention = noise_type_stats['power_supply']['target_attention_at_interval']
    interference_attention = noise_type_stats['interference']['target_attention_at_interval']
    clock_leakage_attention = noise_type_stats['clock_leakage']['target_attention_at_interval']
    
    print(f"  power_supplyの目標アテンション: {power_supply_attention:.6f}")
    print(f"  interferenceの目標アテンション: {interference_attention:.6f}")
    print(f"  clock_leakageの目標アテンション: {clock_leakage_attention:.6f}")
    
    if abs(power_supply_attention - interference_attention) > 1e-6:
        print(f"  ✅ power_supplyとinterferenceで目標アテンションが異なる")
    else:
        print(f"  ⚠️ power_supplyとinterferenceで目標アテンションが同じ（問題あり）")
    
    if abs(power_supply_attention - clock_leakage_attention) > 1e-6:
        print(f"  ✅ power_supplyとclock_leakageで目標アテンションが異なる")
    else:
        print(f"  ⚠️ power_supplyとclock_leakageで目標アテンションが同じ（問題あり）")
    
    if abs(interference_attention - clock_leakage_attention) > 1e-6:
        print(f"  ✅ interferenceとclock_leakageで目標アテンションが異なる")
    else:
        print(f"  ⚠️ interferenceとclock_leakageで目標アテンションが同じ（問題あり）")
    
    print(f"\n{'=' * 80}")
    print("テスト完了")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    test_regularization_loss()

