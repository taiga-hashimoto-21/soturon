"""
目標アテンションと予測アテンションの差を分析し、
どうすれば一致させられるかを計算するスクリプト
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

def analyze_target_vs_predicted(attention_data_path):
    """
    目標アテンションと予測アテンションの差を分析
    
    Args:
        attention_data_path: attention_data.pklのパス
    """
    print("=" * 80)
    print("目標アテンションと予測アテンションの差の分析")
    print("=" * 80)
    
    with open(attention_data_path, 'rb') as f:
        data = pickle.load(f)
    
    predicted_attention = data['predicted_attention']  # (N, 30) 正規化後
    predicted_attention_before_norm = data['predicted_attention_before_normalization']  # (N, 30) 正規化前
    target_attention = data['target_attention']  # (N, 30) 正規化後
    target_attention_before_norm = data['target_attention_before_normalization']  # (N, 30) 正規化前
    true_noise_intervals = data['true_noise_intervals']  # (N,)
    predicted_noise_intervals = data['predicted_noise_intervals']  # (N,)
    noise_strength = data['noise_strength']  # (N, 30)
    
    N = len(true_noise_intervals)
    
    print(f"\nサンプル数: {N}")
    print(f"予測精度: {(predicted_noise_intervals == true_noise_intervals).sum() / N * 100:.2f}%")
    
    # 1. 正規化後の差を分析
    print("\n【1. 正規化後のアテンションの差】")
    diff_normalized = predicted_attention - target_attention  # (N, 30)
    
    # ノイズ区間と正常区間で分けて分析
    noise_diff_list = []
    normal_diff_list = []
    
    for i in range(N):
        noise_idx = true_noise_intervals[i]
        noise_diff = diff_normalized[i, noise_idx]
        noise_diff_list.append(noise_diff)
        
        normal_indices = [j for j in range(30) if j != noise_idx]
        normal_diffs = diff_normalized[i, normal_indices]
        normal_diff_list.extend(normal_diffs.tolist())
    
    noise_diff = np.array(noise_diff_list)
    normal_diff = np.array(normal_diff_list)
    
    print(f"\nノイズ区間の差（予測 - 目標）:")
    print(f"  平均: {noise_diff.mean():.8f}")
    print(f"  標準偏差: {noise_diff.std():.8f}")
    print(f"  最小: {noise_diff.min():.8f}")
    print(f"  最大: {noise_diff.max():.8f}")
    
    print(f"\n正常区間の差（予測 - 目標）:")
    print(f"  平均: {normal_diff.mean():.8f}")
    print(f"  標準偏差: {normal_diff.std():.8f}")
    print(f"  最小: {normal_diff.min():.8f}")
    print(f"  最大: {normal_diff.max():.8f}")
    
    # 2. 正規化前の差を分析
    print("\n【2. 正規化前のアテンションの差】")
    diff_before_norm = predicted_attention_before_norm - target_attention_before_norm  # (N, 30)
    
    noise_diff_before_list = []
    normal_diff_before_list = []
    
    for i in range(N):
        noise_idx = true_noise_intervals[i]
        noise_diff = diff_before_norm[i, noise_idx]
        noise_diff_before_list.append(noise_diff)
        
        normal_indices = [j for j in range(30) if j != noise_idx]
        normal_diffs = diff_before_norm[i, normal_indices]
        normal_diff_before_list.extend(normal_diffs.tolist())
    
    noise_diff_before = np.array(noise_diff_before_list)
    normal_diff_before = np.array(normal_diff_before_list)
    
    print(f"\nノイズ区間の差（予測 - 目標、正規化前）:")
    print(f"  平均: {noise_diff_before.mean():.8f}")
    print(f"  標準偏差: {noise_diff_before.std():.8f}")
    print(f"  最小: {noise_diff_before.min():.8f}")
    print(f"  最大: {noise_diff_before.max():.8f}")
    
    print(f"\n正常区間の差（予測 - 目標、正規化前）:")
    print(f"  平均: {normal_diff_before.mean():.8f}")
    print(f"  標準偏差: {normal_diff_before.std():.8f}")
    print(f"  最小: {normal_diff_before.min():.8f}")
    print(f"  最大: {normal_diff_before.max():.8f}")
    
    # 3. ノイズ強度とアテンションの関係を分析
    print("\n【3. ノイズ強度とアテンションの関係】")
    
    # 目標アテンション = 1 / noise_strength（理論値）
    # 予測アテンションと目標アテンションの比を計算
    ratio_list = []
    for i in range(N):
        noise_idx = true_noise_intervals[i]
        target_attn = target_attention_before_norm[i, noise_idx]
        predicted_attn = predicted_attention_before_norm[i, noise_idx]
        
        if target_attn > 0:
            ratio = predicted_attn / target_attn
            ratio_list.append(ratio)
    
    ratio_array = np.array(ratio_list)
    print(f"\n予測アテンション / 目標アテンション（ノイズ区間、正規化前）:")
    print(f"  平均: {ratio_array.mean():.8f}")
    print(f"  標準偏差: {ratio_array.std():.8f}")
    print(f"  最小: {ratio_array.min():.8f}")
    print(f"  最大: {ratio_array.max():.8f}")
    print(f"  理想値: 1.0（完全一致）")
    
    if ratio_array.mean() < 1.0:
        scale_factor = 1.0 / ratio_array.mean()
        print(f"\n  ⚠️ 予測アテンションが目標より小さい")
        print(f"  推奨: 予測アテンションを {scale_factor:.4f} 倍にスケールすると目標に近づく")
    elif ratio_array.mean() > 1.0:
        scale_factor = 1.0 / ratio_array.mean()
        print(f"\n  ⚠️ 予測アテンションが目標より大きい")
        print(f"  推奨: 予測アテンションを {scale_factor:.4f} 倍にスケールすると目標に近づく")
    else:
        print(f"\n  ✅ 予測アテンションと目標が一致している")
    
    # 4. 予測が外れたサンプルでの差を分析
    print("\n【4. 予測が外れたサンプルでの差の分析】")
    wrong_mask = predicted_noise_intervals != true_noise_intervals
    wrong_indices = np.where(wrong_mask)[0]
    
    if len(wrong_indices) > 0:
        print(f"\n予測が外れたサンプル数: {len(wrong_indices)} / {N}")
        
        # 予測が外れたサンプルでのノイズ区間の差
        wrong_noise_diff = []
        wrong_normal_diff = []
        
        for i in wrong_indices:
            noise_idx = true_noise_intervals[i]
            noise_diff = diff_before_norm[i, noise_idx]
            wrong_noise_diff.append(noise_diff)
            
            # 予測された区間の差
            pred_idx = predicted_noise_intervals[i]
            pred_diff = diff_before_norm[i, pred_idx]
            wrong_normal_diff.append(pred_diff)
        
        wrong_noise_diff = np.array(wrong_noise_diff)
        wrong_normal_diff = np.array(wrong_normal_diff)
        
        print(f"\n予測が外れたサンプルでのノイズ区間の差（正規化前）:")
        print(f"  平均: {wrong_noise_diff.mean():.8f}")
        print(f"  標準偏差: {wrong_noise_diff.std():.8f}")
        
        print(f"\n予測が外れたサンプルでの予測区間の差（正規化前）:")
        print(f"  平均: {wrong_normal_diff.mean():.8f}")
        print(f"  標準偏差: {wrong_normal_diff.std():.8f}")
        
        # ノイズ区間と予測区間のアテンションを比較
        noise_attn_wrong = []
        pred_attn_wrong = []
        
        for i in wrong_indices:
            noise_idx = true_noise_intervals[i]
            pred_idx = predicted_noise_intervals[i]
            
            noise_attn = predicted_attention_before_norm[i, noise_idx]
            pred_attn = predicted_attention_before_norm[i, pred_idx]
            
            noise_attn_wrong.append(noise_attn)
            pred_attn_wrong.append(pred_attn)
        
        noise_attn_wrong = np.array(noise_attn_wrong)
        pred_attn_wrong = np.array(pred_attn_wrong)
        
        print(f"\n予測が外れたサンプルでのアテンション比較（正規化前）:")
        print(f"  ノイズ区間の平均アテンション: {noise_attn_wrong.mean():.8f}")
        print(f"  予測区間の平均アテンション: {pred_attn_wrong.mean():.8f}")
        print(f"  差（ノイズ - 予測）: {(noise_attn_wrong - pred_attn_wrong).mean():.8f}")
        
        if (noise_attn_wrong - pred_attn_wrong).mean() > 0:
            print(f"\n  ⚠️ 問題: ノイズ区間のアテンションが予測区間より高い")
            print(f"  原因: ノイズ区間のアテンションが十分に低く学習されていない")
            print(f"  解決策: ノイズ区間のアテンションをさらに低くする必要がある")
        else:
            print(f"\n  ✅ ノイズ区間のアテンションは予測区間より低い")
            print(f"  原因: 予測区間のアテンションが偶然低くなっている")
            print(f"  解決策: 正常区間のアテンションを均一にする必要がある")
    
    # 5. 損失関数の重みの推奨値を計算
    print("\n【5. 損失関数の重みの推奨値】")
    
    # MSE損失を計算
    mse_loss_normalized = ((predicted_attention - target_attention) ** 2).mean()
    mse_loss_before_norm = ((predicted_attention_before_norm - target_attention_before_norm) ** 2).mean()
    
    print(f"\nMSE損失（正規化後）: {mse_loss_normalized:.8f}")
    print(f"MSE損失（正規化前）: {mse_loss_before_norm:.8f}")
    
    # ノイズ区間のみのMSE損失
    noise_mse_list = []
    for i in range(N):
        noise_idx = true_noise_intervals[i]
        noise_mse = (predicted_attention_before_norm[i, noise_idx] - target_attention_before_norm[i, noise_idx]) ** 2
        noise_mse_list.append(noise_mse)
    
    noise_mse = np.mean(noise_mse_list)
    print(f"ノイズ区間のみのMSE損失（正規化前）: {noise_mse:.8f}")
    
    # 現在のlambda_regとlambda_noise_intervalを考慮した推奨値
    current_lambda_reg = 1.0
    current_lambda_noise_interval = 50.0
    
    # ノイズ区間の損失が全体の損失に占める割合を計算
    noise_loss_ratio = noise_mse / mse_loss_before_norm if mse_loss_before_norm > 0 else 0
    
    print(f"\nノイズ区間の損失が全体の損失に占める割合: {noise_loss_ratio:.4f}")
    
    if noise_loss_ratio < 0.5:
        recommended_lambda_noise_interval = current_lambda_noise_interval * (0.5 / noise_loss_ratio)
        print(f"\n推奨: lambda_noise_intervalを {current_lambda_noise_interval:.1f} → {recommended_lambda_noise_interval:.1f} に増やす")
        print(f"理由: ノイズ区間の損失が全体の損失に占める割合が小さいため")
    
    print("\n" + "=" * 80)
    print("分析完了")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        attention_data_path = sys.argv[1]
    else:
        attention_data_path = "/Users/hashimototaiga/Desktop/attention_data.pkl"
    
    analyze_target_vs_predicted(attention_data_path)

