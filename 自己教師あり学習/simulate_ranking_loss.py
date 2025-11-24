"""
attention_data.pklを使って、ランキング損失を追加した場合の
予測精度をシミュレーションするスクリプト
"""

import pickle
import numpy as np

def simulate_ranking_loss_effect(attention_data_path, ranking_margin=0.01):
    """
    ランキング損失を追加した場合の予測精度をシミュレーション
    
    Args:
        attention_data_path: attention_data.pklのパス
        ranking_margin: ランキング損失のマージン
    """
    print("=" * 80)
    print("ランキング損失の効果をシミュレーション")
    print("=" * 80)
    
    with open(attention_data_path, 'rb') as f:
        data = pickle.load(f)
    
    predicted_attention_before_norm = data['predicted_attention_before_normalization']
    predicted_attention = data['predicted_attention']
    true_noise_intervals = data['true_noise_intervals']
    predicted_noise_intervals = data['predicted_noise_intervals']
    
    N = len(true_noise_intervals)
    
    print(f"\nサンプル数: {N}")
    print(f"現在の予測精度: {(predicted_noise_intervals == true_noise_intervals).sum() / N * 100:.2f}%")
    
    # 1. 現在のランキング損失を計算（正規化前）
    print("\n【1. 現在のランキング損失（正規化前）】")
    ranking_losses_before = []
    violations_before = []  # ランキング違反のサンプル
    
    for i in range(N):
        noise_idx = true_noise_intervals[i]
        noise_attn = predicted_attention_before_norm[i, noise_idx]
        
        normal_indices = [j for j in range(30) if j != noise_idx]
        normal_attns = predicted_attention_before_norm[i, normal_indices]
        normal_min_attn = normal_attns.min()
        
        # ランキング損失
        ranking_loss = max(0, noise_attn - normal_min_attn + ranking_margin)
        ranking_losses_before.append(ranking_loss)
        
        # ランキング違反（ノイズ区間のアテンション > 正常区間の最小アテンション）
        if noise_attn > normal_min_attn:
            violations_before.append(i)
    
    ranking_losses_before = np.array(ranking_losses_before)
    print(f"平均ランキング損失: {ranking_losses_before.mean():.10f}")
    print(f"ランキング違反のサンプル数: {len(violations_before)} / {N} ({len(violations_before)/N*100:.2f}%)")
    print(f"ランキング違反の平均損失: {ranking_losses_before[violations_before].mean():.10f}" if len(violations_before) > 0 else "ランキング違反なし")
    
    # 2. 現在のランキング損失を計算（正規化後）
    print("\n【2. 現在のランキング損失（正規化後）】")
    ranking_losses_after = []
    violations_after = []
    
    for i in range(N):
        noise_idx = true_noise_intervals[i]
        noise_attn = predicted_attention[i, noise_idx]
        
        normal_indices = [j for j in range(30) if j != noise_idx]
        normal_attns = predicted_attention[i, normal_indices]
        normal_min_attn = normal_attns.min()
        
        ranking_loss = max(0, noise_attn - normal_min_attn + ranking_margin)
        ranking_losses_after.append(ranking_loss)
        
        if noise_attn > normal_min_attn:
            violations_after.append(i)
    
    ranking_losses_after = np.array(ranking_losses_after)
    print(f"平均ランキング損失: {ranking_losses_after.mean():.10f}")
    print(f"ランキング違反のサンプル数: {len(violations_after)} / {N} ({len(violations_after)/N*100:.2f}%)")
    print(f"ランキング違反の平均損失: {ranking_losses_after[violations_after].mean():.10f}" if len(violations_after) > 0 else "ランキング違反なし")
    
    # 3. ランキング損失が0になった場合の予測精度をシミュレーション
    print("\n【3. ランキング損失が0になった場合の予測精度（シミュレーション）】")
    
    # ランキング違反を修正した場合の予測精度
    # 方法: ノイズ区間のアテンションを正常区間の最小アテンションより低くする
    corrected_predictions = 0
    
    for i in range(N):
        noise_idx = true_noise_intervals[i]
        noise_attn = predicted_attention[i, noise_idx]  # 正規化後
        
        normal_indices = [j for j in range(30) if j != noise_idx]
        normal_attns = predicted_attention[i, normal_indices]
        normal_min_attn = normal_attns.min()
        
        # ランキング違反がある場合
        if noise_attn > normal_min_attn:
            # ノイズ区間のアテンションを正常区間の最小アテンションより低くする
            # （シミュレーション: ノイズ区間のアテンションを正常区間の最小値より低く設定）
            corrected_noise_attn = normal_min_attn - ranking_margin
            
            # 修正後のアテンションで予測
            corrected_attention = predicted_attention[i].copy()
            corrected_attention[noise_idx] = corrected_noise_attn
            
            # 正規化（合計が1になるように）
            corrected_attention = corrected_attention / corrected_attention.sum()
            
            # 予測
            predicted_interval = corrected_attention.argmin()
            
            if predicted_interval == noise_idx:
                corrected_predictions += 1
        else:
            # ランキング違反がない場合、現在の予測が正しいか確認
            predicted_interval = predicted_attention[i].argmin()
            if predicted_interval == noise_idx:
                corrected_predictions += 1
    
    corrected_accuracy = corrected_predictions / N
    print(f"ランキング損失が0になった場合の予測精度: {corrected_accuracy:.4f} ({corrected_accuracy*100:.2f}%)")
    print(f"現在の予測精度: {(predicted_noise_intervals == true_noise_intervals).sum() / N:.4f} ({(predicted_noise_intervals == true_noise_intervals).sum() / N*100:.2f}%)")
    print(f"改善: {corrected_accuracy - (predicted_noise_intervals == true_noise_intervals).sum() / N:.4f} ({(corrected_accuracy - (predicted_noise_intervals == true_noise_intervals).sum() / N)*100:.2f}%)")
    
    # 4. ランキング違反のサンプルの詳細
    print("\n【4. ランキング違反のサンプルの詳細（最初の10サンプル）】")
    for i in violations_after[:10]:
        noise_idx = true_noise_intervals[i]
        pred_idx = predicted_noise_intervals[i]
        
        noise_attn = predicted_attention[i, noise_idx]
        
        normal_indices = [j for j in range(30) if j != noise_idx]
        normal_attns = predicted_attention[i, normal_indices]
        normal_min_attn = normal_attns.min()
        
        print(f"\nサンプル {i}:")
        print(f"  正解区間: {noise_idx}, 予測区間: {pred_idx}")
        print(f"  ノイズ区間のアテンション: {noise_attn:.10f}")
        print(f"  正常区間の最小アテンション: {normal_min_attn:.10f}")
        print(f"  差（ノイズ - 正常最小）: {noise_attn - normal_min_attn:.10f}")
        print(f"  ランキング損失: {max(0, noise_attn - normal_min_attn + ranking_margin):.10f}")
        print(f"  必要な削減量: {noise_attn - normal_min_attn + ranking_margin:.10f}")
    
    # 5. 結論
    print("\n【5. 結論】")
    if len(violations_after) > 0:
        print(f"✅ ランキング損失を追加することで、{len(violations_after)}サンプルのランキング違反を修正できます")
        print(f"✅ 予測精度が {(corrected_accuracy - (predicted_noise_intervals == true_noise_intervals).sum() / N)*100:.2f}% 向上する可能性があります")
        print(f"✅ 最終的な予測精度: {corrected_accuracy*100:.2f}%")
    else:
        print(f"✅ ランキング違反はありません。現在の予測精度は既に最適です。")
    
    print("\n" + "=" * 80)
    print("シミュレーション完了")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        attention_data_path = sys.argv[1]
    else:
        attention_data_path = "/Users/hashimototaiga/Desktop/attention_data.pkl"
    
    ranking_margin = 0.01
    if len(sys.argv) > 2:
        ranking_margin = float(sys.argv[2])
    
    simulate_ranking_loss_effect(attention_data_path, ranking_margin)

