"""
アテンションウェイトの詳細分析
予測が外れる原因を特定する
"""

import pickle
import numpy as np

def detailed_analysis(attention_data_path):
    """詳細な分析を実行"""
    print("=" * 80)
    print("アテンションウェイトの詳細分析")
    print("=" * 80)
    
    with open(attention_data_path, 'rb') as f:
        data = pickle.load(f)
    
    predicted_attention = data['predicted_attention']  # (N, 30) 正規化済み
    predicted_attention_before_norm = data.get('predicted_attention_before_normalization', None)  # (N, 30) 正規化前
    true_noise_intervals = data['true_noise_intervals']  # (N,)
    predicted_noise_intervals = data['predicted_noise_intervals']  # (N,)
    predicted_noise_intervals_before_norm = data.get('predicted_noise_intervals_before_normalization', None)  # (N,)
    noise_strength = data['noise_strength']  # (N, 30)
    
    N = len(true_noise_intervals)
    
    print(f"\nサンプル数: {N}")
    print(f"予測精度: {(predicted_noise_intervals == true_noise_intervals).sum() / N * 100:.2f}%")
    
    # 問題の核心を分析
    print("\n【問題の核心分析】")
    print("予測が外れたサンプルで、正解区間と予測区間のアテンションを比較:")
    
    wrong_mask = predicted_noise_intervals != true_noise_intervals
    wrong_indices = np.where(wrong_mask)[0]
    
    print(f"\n予測が外れたサンプル数: {len(wrong_indices)} / {N}")
    
    # 最初の10サンプルの詳細を表示
    print("\n【予測が外れたサンプルの詳細（最初の10サンプル）】")
    for idx in wrong_indices[:10]:
        true_interval = true_noise_intervals[idx]
        pred_interval = predicted_noise_intervals[idx]
        
        true_attn = predicted_attention[idx, true_interval]
        pred_attn = predicted_attention[idx, pred_interval]
        
        # 全区間のアテンションを取得
        all_attn = predicted_attention[idx]
        min_attn = all_attn.min()
        min_interval = all_attn.argmin()
        
        print(f"\nサンプル {idx}:")
        print(f"  正解区間: {true_interval}, 予測区間: {pred_interval}")
        print(f"  正解区間のアテンション: {true_attn:.8f}")
        print(f"  予測区間のアテンション: {pred_attn:.8f}")
        print(f"  最小アテンション: {min_attn:.8f} (区間{min_interval})")
        print(f"  正解区間のアテンションが最小か: {true_interval == min_interval}")
        print(f"  予測区間のアテンションが最小か: {pred_interval == min_interval}")
        
        # 正常区間の平均アテンション
        normal_indices = [j for j in range(30) if j != true_interval]
        normal_attn_mean = predicted_attention[idx, normal_indices].mean()
        print(f"  正常区間の平均アテンション: {normal_attn_mean:.8f}")
        print(f"  正解区間のアテンション < 正常区間の平均: {true_attn < normal_attn_mean}")
        
        # 正解区間よりアテンションが低い区間の数
        lower_than_true = (all_attn < true_attn).sum()
        print(f"  正解区間よりアテンションが低い区間数: {lower_than_true} / 30")
    
    # 統計的分析
    print("\n【統計的分析】")
    
    # 正解区間のアテンションが最小かどうか
    correct_is_min = []
    for i in range(N):
        true_interval = true_noise_intervals[i]
        true_attn = predicted_attention[i, true_interval]
        min_attn = predicted_attention[i].min()
        correct_is_min.append(true_attn == min_attn)
    
    correct_is_min = np.array(correct_is_min)
    print(f"正解区間のアテンションが最小のサンプル数: {correct_is_min.sum()} / {N} ({correct_is_min.sum()/N*100:.2f}%)")
    
    # 正解区間よりアテンションが低い区間の数
    num_lower_intervals = []
    for i in range(N):
        true_interval = true_noise_intervals[i]
        true_attn = predicted_attention[i, true_interval]
        lower_count = (predicted_attention[i] < true_attn).sum()
        num_lower_intervals.append(lower_count)
    
    num_lower_intervals = np.array(num_lower_intervals)
    print(f"正解区間よりアテンションが低い区間の平均: {num_lower_intervals.mean():.2f}")
    print(f"正解区間よりアテンションが低い区間の最大: {num_lower_intervals.max()}")
    print(f"正解区間よりアテンションが低い区間の最小: {num_lower_intervals.min()}")
    
    # 正解区間のアテンションが最小でない場合、なぜ最小でないのか
    print("\n【正解区間のアテンションが最小でない理由】")
    wrong_cases = ~correct_is_min
    if wrong_cases.sum() > 0:
        print(f"正解区間のアテンションが最小でないサンプル数: {wrong_cases.sum()}")
        
        # 正解区間のアテンションと最小アテンションの差
        attn_diffs = []
        for i in np.where(wrong_cases)[0]:
            true_interval = true_noise_intervals[i]
            true_attn = predicted_attention[i, true_interval]
            min_attn = predicted_attention[i].min()
            attn_diffs.append(true_attn - min_attn)
        
        attn_diffs = np.array(attn_diffs)
        print(f"正解区間のアテンション - 最小アテンションの差:")
        print(f"  平均: {attn_diffs.mean():.8f}")
        print(f"  最小: {attn_diffs.min():.8f}")
        print(f"  最大: {attn_diffs.max():.8f}")
        
        # 正規化の問題を確認
        print("\n【正規化の問題の可能性】")
        print("L1正規化により、各サンプルのアテンションの合計が1になります。")
        print("ノイズ区間のアテンションが非常に小さい場合、正常区間のアテンションがほぼ均一になります。")
        print("その結果、正常区間のどれかが偶然ノイズ区間より低くなる可能性があります。")
        
        # 正規化前のアテンションを分析
        print("\n【正規化前のアテンションの分析】")
        if predicted_attention_before_norm is not None:
            print("✅ 正規化前のアテンションウェイトが保存されています")
            
            # 正規化前の予測精度
            if predicted_noise_intervals_before_norm is not None:
                correct_before_norm = (predicted_noise_intervals_before_norm == true_noise_intervals).sum()
                accuracy_before_norm = correct_before_norm / N
                print(f"\n正規化前の予測精度: {accuracy_before_norm:.4f} ({accuracy_before_norm*100:.2f}%)")
                print(f"正規化後の予測精度: {(predicted_noise_intervals == true_noise_intervals).sum() / N:.4f} ({(predicted_noise_intervals == true_noise_intervals).sum() / N * 100:.2f}%)")
                
                if accuracy_before_norm > (predicted_noise_intervals == true_noise_intervals).sum() / N:
                    print("⚠️ 正規化前の方が予測精度が高い！正規化が問題の可能性があります。")
            
            # 正規化前のアテンションウェイトの詳細
            for i in wrong_indices[:5]:
                true_interval = true_noise_intervals[i]
                pred_interval = predicted_noise_intervals[i]
                
                print(f"\nサンプル {i}:")
                print(f"  正解区間: {true_interval}, 予測区間（正規化後）: {pred_interval}")
                
                if predicted_noise_intervals_before_norm is not None:
                    pred_interval_before = predicted_noise_intervals_before_norm[i]
                    print(f"  予測区間（正規化前）: {pred_interval_before}")
                
                # 正規化前のアテンション
                attn_before = predicted_attention_before_norm[i]
                true_attn_before = attn_before[true_interval]
                pred_attn_before = attn_before[pred_interval]
                min_attn_before = attn_before.min()
                min_interval_before = attn_before.argmin()
                
                print(f"  正規化前のアテンション（正解区間{true_interval}）: {true_attn_before:.8f}")
                print(f"  正規化前のアテンション（予測区間{pred_interval}）: {pred_attn_before:.8f}")
                print(f"  正規化前の最小アテンション: {min_attn_before:.8f} (区間{min_interval_before})")
                print(f"  正規化前で正解区間が最小か: {true_interval == min_interval_before}")
                
                # 正規化後のアテンション
                attn_after = predicted_attention[i]
                true_attn_after = attn_after[true_interval]
                pred_attn_after = attn_after[pred_interval]
                min_attn_after = attn_after.min()
                min_interval_after = attn_after.argmin()
                
                print(f"  正規化後のアテンション（正解区間{true_interval}）: {true_attn_after:.8f}")
                print(f"  正規化後のアテンション（予測区間{pred_interval}）: {pred_attn_after:.8f}")
                print(f"  正規化後の最小アテンション: {min_attn_after:.8f} (区間{min_interval_after})")
                print(f"  正規化後で正解区間が最小か: {true_interval == min_interval_after}")
                
                # ノイズ強度との関係
                noise_str = noise_strength[i, true_interval]
                print(f"  ノイズ強度（正解区間）: {noise_str:.6f}")
                print(f"  ノイズ強度の逆数（理論値）: {1.0/noise_str:.6f}")
                print(f"  正規化前のアテンション / ノイズ強度の逆数: {true_attn_before / (1.0/noise_str):.6f}")
        else:
            print("⚠️ 正規化前のアテンションウェイトが保存されていません。")
            print("   学習コードを更新して、正規化前の値も保存するようにしてください。")
    
    print("\n" + "=" * 80)
    print("分析完了")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        attention_data_path = sys.argv[1]
    else:
        attention_data_path = "/Users/hashimototaiga/Desktop/attention_data.pkl"
    
    detailed_analysis(attention_data_path)

