"""
正解のアテンションウェイト（1/ノイズ強度）がノイズタイプによって異なるか確認
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

def test_target_attention():
    """正解のアテンションウェイトを確認"""
    print("=" * 80)
    print("正解のアテンションウェイト（1/ノイズ強度）の確認")
    print("=" * 80)
    
    pickle_path = os.path.join(project_root, 'data_lowF_noise.pickle')
    
    # 3種類のノイズタイプをテスト
    noise_types = ['power_supply', 'interference', 'clock_leakage']
    
    results = {}
    
    for noise_type in noise_types:
        print(f"\n{'=' * 80}")
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
        
        # 複数のサンプルを平均
        num_samples = 5
        all_noise_strengths = []
        all_target_attentions = []
        
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            noise_strength = sample['noise_strength'].numpy()  # (30,)
            noise_interval = sample['noise_interval'].item()
            
            # 正解のアテンションウェイト = 1 / ノイズ強度
            epsilon = 1e-6
            target_attention = 1.0 / (noise_strength + epsilon)
            # 正規化（合計=1）
            target_attention_normalized = target_attention / target_attention.sum()
            
            all_noise_strengths.append(noise_strength)
            all_target_attentions.append(target_attention_normalized)
        
        # 平均を計算
        avg_noise_strength = np.mean(all_noise_strengths, axis=0)
        avg_target_attention = np.mean(all_target_attentions, axis=0)
        
        # ノイズ区間のインデックス（最初のサンプルから）
        noise_interval = dataset[0]['noise_interval'].item()
        
        print(f"\nノイズ区間: {noise_interval}")
        print(f"\nノイズ強度（平均）:")
        print(f"  ノイズ区間({noise_interval})の値: {avg_noise_strength[noise_interval]:.6f}")
        print(f"  最小値: {avg_noise_strength.min():.6f}")
        print(f"  最大値: {avg_noise_strength.max():.6f}")
        print(f"  標準偏差: {avg_noise_strength.std():.6f}")
        
        print(f"\n正解のアテンションウェイト（1/ノイズ強度、正規化済み）:")
        print(f"  ノイズ区間({noise_interval})の値: {avg_target_attention[noise_interval]:.10f}")
        print(f"  最小値: {avg_target_attention.min():.10f}")
        print(f"  最大値: {avg_target_attention.max():.10f}")
        print(f"  標準偏差: {avg_target_attention.std():.10f}")
        
        # ノイズ区間の目標アテンションが小さいか確認
        if avg_target_attention[noise_interval] < avg_target_attention.mean():
            print(f"  ✅ ノイズ区間の目標アテンションが平均より小さい（正しい）")
        else:
            print(f"  ⚠️ ノイズ区間の目標アテンションが平均より大きい（問題あり）")
        
        # ノイズ強度が大きい区間の目標アテンションを確認
        max_noise_strength_idx = avg_noise_strength.argmax()
        print(f"\nノイズ強度が最大の区間({max_noise_strength_idx}):")
        print(f"  ノイズ強度: {avg_noise_strength[max_noise_strength_idx]:.6f}")
        print(f"  目標アテンション: {avg_target_attention[max_noise_strength_idx]:.10f}")
        
        # ノイズ強度が小さい区間の目標アテンションを確認
        min_noise_strength_idx = avg_noise_strength.argmin()
        print(f"\nノイズ強度が最小の区間({min_noise_strength_idx}):")
        print(f"  ノイズ強度: {avg_noise_strength[min_noise_strength_idx]:.6f}")
        print(f"  目標アテンション: {avg_target_attention[min_noise_strength_idx]:.10f}")
        
        results[noise_type] = {
            'noise_interval': noise_interval,
            'noise_strength_at_interval': avg_noise_strength[noise_interval],
            'target_attention_at_interval': avg_target_attention[noise_interval],
            'noise_strength_std': avg_noise_strength.std(),
            'target_attention_std': avg_target_attention.std(),
        }
    
    # ノイズタイプ間の比較
    print(f"\n{'=' * 80}")
    print("ノイズタイプ間の比較")
    print(f"{'=' * 80}")
    
    print(f"\n各ノイズタイプのノイズ区間での値:")
    for noise_type, stats in results.items():
        print(f"  {noise_type}:")
        print(f"    ノイズ区間: {stats['noise_interval']}")
        print(f"    ノイズ強度: {stats['noise_strength_at_interval']:.6f}")
        print(f"    目標アテンション: {stats['target_attention_at_interval']:.10f}")
    
    # ノイズタイプによって目標アテンションが異なるか確認
    print(f"\nノイズタイプ間の違い:")
    power_supply_attention = results['power_supply']['target_attention_at_interval']
    interference_attention = results['interference']['target_attention_at_interval']
    clock_leakage_attention = results['clock_leakage']['target_attention_at_interval']
    
    print(f"  power_supplyの目標アテンション: {power_supply_attention:.10f}")
    print(f"  interferenceの目標アテンション: {interference_attention:.10f}")
    print(f"  clock_leakageの目標アテンション: {clock_leakage_attention:.10f}")
    
    if abs(power_supply_attention - interference_attention) > 1e-10:
        print(f"  ✅ power_supplyとinterferenceで目標アテンションが異なる")
        print(f"     差: {abs(power_supply_attention - interference_attention):.10f}")
    else:
        print(f"  ⚠️ power_supplyとinterferenceで目標アテンションが同じ（問題あり）")
    
    if abs(power_supply_attention - clock_leakage_attention) > 1e-10:
        print(f"  ✅ power_supplyとclock_leakageで目標アテンションが異なる")
        print(f"     差: {abs(power_supply_attention - clock_leakage_attention):.10f}")
    else:
        print(f"  ⚠️ power_supplyとclock_leakageで目標アテンションが同じ（問題あり）")
    
    if abs(interference_attention - clock_leakage_attention) > 1e-10:
        print(f"  ✅ interferenceとclock_leakageで目標アテンションが異なる")
        print(f"     差: {abs(interference_attention - clock_leakage_attention):.10f}")
    else:
        print(f"  ⚠️ interferenceとclock_leakageで目標アテンションが同じ（問題あり）")
    
    print(f"\n{'=' * 80}")
    print("結論:")
    print(f"{'=' * 80}")
    print(f"ノイズタイプによってノイズ強度が異なるため、")
    print(f"正解のアテンションウェイト（1/ノイズ強度）も異なります。")
    print(f"これにより、正則化損失の計算時に、")
    print(f"ノイズタイプに応じた適切な目標アテンションが使用されます。")
    
    print(f"\n{'=' * 80}")
    print("テスト完了")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    test_target_attention()

