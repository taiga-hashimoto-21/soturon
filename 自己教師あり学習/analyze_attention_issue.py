"""
アテンションウェイトの問題を分析するスクリプト

学習時と評価時でアテンションウェイトの計算方法が同じかどうかを確認し、
ノイズ区間予測精度が低い原因を特定する
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_attention_issue(attention_data_path, train_accuracies_path, eval_output_dir=None):
    """
    アテンションウェイトの問題を分析
    
    Args:
        attention_data_path: attention_data.pklのパス
        train_accuracies_path: train_noise_interval_accuracies.pklのパス
        eval_output_dir: 評価結果のディレクトリ（オプション）
    """
    print("=" * 80)
    print("アテンションウェイトの問題分析")
    print("=" * 80)
    
    # 1. 学習時のノイズ区間予測精度を確認
    print("\n【1. 学習時のノイズ区間予測精度】")
    if Path(train_accuracies_path).exists():
        with open(train_accuracies_path, 'rb') as f:
            train_accuracies = pickle.load(f)
        print(f"  エポック数: {len(train_accuracies)}")
        print(f"  最終エポックの精度: {train_accuracies[-1]:.4f} ({train_accuracies[-1]*100:.2f}%)")
        print(f"  平均精度: {np.mean(train_accuracies):.4f} ({np.mean(train_accuracies)*100:.2f}%)")
        print(f"  最大精度: {np.max(train_accuracies):.4f} ({np.max(train_accuracies)*100:.2f}%)")
        print(f"  最小精度: {np.min(train_accuracies):.4f} ({np.min(train_accuracies)*100:.2f}%)")
        
        # ランダム予測との比較
        random_accuracy = 1.0 / 30.0
        print(f"\n  ランダム予測の期待値: {random_accuracy:.4f} ({random_accuracy*100:.2f}%)")
        if train_accuracies[-1] < random_accuracy:
            print(f"  ⚠️ 警告: 学習時の精度がランダム予測より低い！")
        else:
            print(f"  ✅ 学習時の精度はランダム予測より高い")
    else:
        print(f"  ⚠️ ファイルが見つかりません: {train_accuracies_path}")
    
    # 2. 学習時に保存されたアテンション情報を分析
    print("\n【2. 学習時に保存されたアテンション情報の分析】")
    if Path(attention_data_path).exists():
        with open(attention_data_path, 'rb') as f:
            attention_data = pickle.load(f)
        
        predicted_attention = attention_data['predicted_attention']  # (N, 30) 正規化済み
        predicted_attention_before_norm = attention_data.get('predicted_attention_before_normalization', None)  # (N, 30) 正規化前
        target_attention = attention_data['target_attention']  # (N, 30) 正規化済み
        target_attention_before_norm = attention_data.get('target_attention_before_normalization', None)  # (N, 30) 正規化前
        true_noise_intervals = attention_data['true_noise_intervals']  # (N,)
        predicted_noise_intervals = attention_data['predicted_noise_intervals']  # (N,)
        predicted_noise_intervals_before_norm = attention_data.get('predicted_noise_intervals_before_normalization', None)  # (N,)
        noise_strength = attention_data['noise_strength']  # (N, 30)
        
        N = len(true_noise_intervals)
        print(f"  サンプル数: {N}")
        print(f"  区間数: {predicted_attention.shape[1]}")
        
        # 予測精度を計算
        correct_predictions = (predicted_noise_intervals == true_noise_intervals).sum()
        accuracy = correct_predictions / N
        print(f"\n  予測精度: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  正しく予測できたサンプル数: {correct_predictions} / {N}")
        
        # ノイズ区間と正常区間のアテンションウェイトを比較
        noise_attention_list = []
        normal_attention_list = []
        
        for i in range(N):
            true_interval = true_noise_intervals[i]
            noise_attn = predicted_attention[i, true_interval]
            noise_attention_list.append(noise_attn)
            
            normal_indices = [j for j in range(30) if j != true_interval]
            normal_attn = predicted_attention[i, normal_indices].mean()
            normal_attention_list.append(normal_attn)
        
        noise_attention = np.array(noise_attention_list)
        normal_attention = np.array(normal_attention_list)
        
        print(f"\n  ノイズ区間のアテンションウェイト:")
        print(f"    平均: {noise_attention.mean():.6f}")
        print(f"    最小: {noise_attention.min():.6f}")
        print(f"    最大: {noise_attention.max():.6f}")
        print(f"    標準偏差: {noise_attention.std():.6f}")
        
        print(f"\n  正常区間のアテンションウェイト:")
        print(f"    平均: {normal_attention.mean():.6f}")
        print(f"    最小: {normal_attention.min():.6f}")
        print(f"    最大: {normal_attention.max():.6f}")
        print(f"    標準偏差: {normal_attention.std():.6f}")
        
        print(f"\n  差（正常 - ノイズ）: {normal_attention.mean() - noise_attention.mean():.6f}")
        
        # 問題のあるサンプルを特定
        print(f"\n【3. 問題のあるサンプルの分析】")
        wrong_predictions = predicted_noise_intervals != true_noise_intervals
        num_wrong = wrong_predictions.sum()
        print(f"  予測が外れたサンプル数: {num_wrong} / {N} ({num_wrong/N*100:.2f}%)")
        
        if num_wrong > 0:
            wrong_noise_attention = noise_attention[wrong_predictions]
            wrong_normal_attention = normal_attention[wrong_predictions]
            
            print(f"\n  予測が外れたサンプルでのアテンションウェイト:")
            print(f"    ノイズ区間の平均: {wrong_noise_attention.mean():.6f}")
            print(f"    正常区間の平均: {wrong_normal_attention.mean():.6f}")
            print(f"    差（正常 - ノイズ）: {wrong_normal_attention.mean() - wrong_noise_attention.mean():.6f}")
            
            # ノイズ区間のアテンションが正常区間より高いサンプル数を確認
            noise_higher_than_normal = noise_attention > normal_attention
            num_noise_higher = noise_higher_than_normal.sum()
            print(f"\n  ノイズ区間のアテンション > 正常区間のアテンション のサンプル数: {num_noise_higher} / {N} ({num_noise_higher/N*100:.2f}%)")
            
            if num_noise_higher > 0:
                print(f"  ⚠️ 問題: {num_noise_higher}サンプルでノイズ区間のアテンションが正常区間より高い！")
                print(f"     これが予測精度が低い主な原因です。")
            
            # 正常区間のアテンションがノイズ区間より低いサンプル数を確認
            normal_lower_than_noise = normal_attention < noise_attention
            num_normal_lower = normal_lower_than_noise.sum()
            print(f"\n  正常区間のアテンション < ノイズ区間のアテンション のサンプル数: {num_normal_lower} / {N} ({num_normal_lower/N*100:.2f}%)")
            
            if num_normal_lower > 0:
                print(f"  ⚠️ 問題: {num_normal_lower}サンプルで正常区間のアテンションがノイズ区間より低い！")
                print(f"     これも予測精度が低い原因です。")
        
        # 正規化前のアテンションウェイトを確認
        print(f"\n【4. 正規化前のアテンションウェイトの分析】")
        if predicted_attention_before_norm is not None:
            print(f"  ✅ 正規化前のアテンションウェイトが保存されています")
            
            # 正規化前のノイズ区間と正常区間のアテンションを比較
            noise_attention_before_norm_list = []
            normal_attention_before_norm_list = []
            
            for i in range(N):
                true_interval = true_noise_intervals[i]
                noise_attn = predicted_attention_before_norm[i, true_interval]
                noise_attention_before_norm_list.append(noise_attn)
                
                normal_indices = [j for j in range(30) if j != true_interval]
                normal_attn = predicted_attention_before_norm[i, normal_indices].mean()
                normal_attention_before_norm_list.append(normal_attn)
            
            noise_attention_before_norm = np.array(noise_attention_before_norm_list)
            normal_attention_before_norm = np.array(normal_attention_before_norm_list)
            
            print(f"\n  正規化前のノイズ区間のアテンションウェイト:")
            print(f"    平均: {noise_attention_before_norm.mean():.8f}")
            print(f"    最小: {noise_attention_before_norm.min():.8f}")
            print(f"    最大: {noise_attention_before_norm.max():.8f}")
            print(f"    標準偏差: {noise_attention_before_norm.std():.8f}")
            
            print(f"\n  正規化前の正常区間のアテンションウェイト:")
            print(f"    平均: {normal_attention_before_norm.mean():.8f}")
            print(f"    最小: {normal_attention_before_norm.min():.8f}")
            print(f"    最大: {normal_attention_before_norm.max():.8f}")
            print(f"    標準偏差: {normal_attention_before_norm.std():.8f}")
            
            print(f"\n  差（正常 - ノイズ）: {normal_attention_before_norm.mean() - noise_attention_before_norm.mean():.8f}")
            
            # 正規化前の予測精度
            if predicted_noise_intervals_before_norm is not None:
                correct_before_norm = (predicted_noise_intervals_before_norm == true_noise_intervals).sum()
                accuracy_before_norm = correct_before_norm / N
                print(f"\n  正規化前の予測精度: {accuracy_before_norm:.4f} ({accuracy_before_norm*100:.2f}%)")
                print(f"  正規化前の正しく予測できたサンプル数: {correct_before_norm} / {N}")
                
                # 正規化前と正規化後の予測精度を比較
                print(f"\n  正規化後の予測精度: {accuracy:.4f} ({accuracy*100:.2f}%)")
                if accuracy_before_norm > accuracy:
                    print(f"  ⚠️ 正規化前の方が予測精度が高い！正規化が問題の可能性があります。")
                elif accuracy_before_norm < accuracy:
                    print(f"  ✅ 正規化後の方が予測精度が高い")
                else:
                    print(f"  → 正規化前後で予測精度は同じ")
            
            # 正規化前で正解区間のアテンションが最小かどうか
            correct_is_min_before_norm = []
            for i in range(N):
                true_interval = true_noise_intervals[i]
                true_attn = predicted_attention_before_norm[i, true_interval]
                min_attn = predicted_attention_before_norm[i].min()
                correct_is_min_before_norm.append(true_attn == min_attn)
            
            correct_is_min_before_norm = np.array(correct_is_min_before_norm)
            print(f"\n  正規化前で正解区間のアテンションが最小のサンプル数: {correct_is_min_before_norm.sum()} / {N} ({correct_is_min_before_norm.sum()/N*100:.2f}%)")
            
            # 正規化前と正規化後で正解区間が最小かどうかを比較
            correct_is_min_after_norm = []
            for i in range(N):
                true_interval = true_noise_intervals[i]
                true_attn = predicted_attention[i, true_interval]
                min_attn = predicted_attention[i].min()
                correct_is_min_after_norm.append(true_attn == min_attn)
            
            correct_is_min_after_norm = np.array(correct_is_min_after_norm)
            print(f"  正規化後で正解区間のアテンションが最小のサンプル数: {correct_is_min_after_norm.sum()} / {N} ({correct_is_min_after_norm.sum()/N*100:.2f}%)")
            
            if correct_is_min_before_norm.sum() > correct_is_min_after_norm.sum():
                print(f"  ⚠️ 正規化前の方が正解区間が最小になるサンプルが多い！")
                print(f"     正規化が問題を引き起こしている可能性が高いです。")
        else:
            print(f"  ⚠️ 正規化前のアテンションウェイトが保存されていません。")
            print(f"     学習コードを更新して、正規化前の値も保存するようにしてください。")
        
        # ノイズ強度とアテンションウェイトの関係を確認
        print(f"\n【5. ノイズ強度とアテンションウェイトの関係】")
        for i in range(min(5, N)):
            true_interval = true_noise_intervals[i]
            pred_interval = predicted_noise_intervals[i]
            noise_str = noise_strength[i, true_interval]
            noise_attn = predicted_attention[i, true_interval]
            pred_attn = predicted_attention[i, pred_interval]
            
            is_correct = "✓" if pred_interval == true_interval else "✗"
            print(f"  サンプル {i}: 正解区間={true_interval}, 予測区間={pred_interval} {is_correct}")
            print(f"    ノイズ強度（正解区間）: {noise_str:.6f}")
            print(f"    アテンション（正解区間）: {noise_attn:.6f}")
            print(f"    アテンション（予測区間）: {pred_attn:.6f}")
            if pred_interval != true_interval:
                print(f"    ⚠️ 予測が外れています。予測区間のアテンションが正解区間より低い: {pred_attn < noise_attn}")
        
    else:
        print(f"  ⚠️ ファイルが見つかりません: {attention_data_path}")
    
    print("\n" + "=" * 80)
    print("分析完了")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    # デフォルトパス
    if len(sys.argv) > 1:
        # 引数がファイルパスの場合
        if sys.argv[1].endswith('.pkl'):
            attention_data_path = Path(sys.argv[1])
        else:
            # ディレクトリパスの場合
            output_dir = sys.argv[1]
            attention_data_path = Path(output_dir) / "attention_data.pkl"
    else:
        output_dir = "task4_output"
        attention_data_path = Path(output_dir) / "attention_data.pkl"
    
    # train_accuracies_pathはオプション
    if len(sys.argv) > 2:
        train_accuracies_path = Path(sys.argv[2])
    else:
        # デフォルトではattention_dataと同じディレクトリを探す
        train_accuracies_path = attention_data_path.parent / "train_noise_interval_accuracies.pkl"
    
    analyze_attention_issue(attention_data_path, train_accuracies_path)

