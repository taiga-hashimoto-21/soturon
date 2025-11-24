"""
学習時と評価時でアテンションウェイトの計算方法が同じかどうかを確認するスクリプト
"""

import torch
import torch.nn.functional as F
import numpy as np

def compare_attention_calculation_methods():
    """
    学習時と評価時でアテンションウェイトの計算方法を比較
    
    学習時（train.py）:
    1. cls_attention_full = attention_weights[:, :, 0, 1:]  # (B, n_heads, L)
    2. cls_attention_mean = cls_attention_full.mean(dim=1)  # (B, L)
    3. 区間ごとに平均化して interval_attention (B, num_intervals)
    4. interval_attention_normalized = F.normalize(interval_attention, p=1, dim=1)
    
    評価時（eval.py）:
    1. get_cls_attention_to_intervals を使うか、手動計算
    2. cls_attention_full = attention_weights[:, :, 0, 1:]  # (B, n_heads, L)
    3. cls_attention_mean = cls_attention_full.mean(dim=1)  # (B, L)
    4. 区間ごとに平均化して cls_attention (B, num_intervals)
    5. L1正規化（numpyで手動実装）
    """
    print("=" * 80)
    print("学習時と評価時のアテンションウェイト計算方法の比較")
    print("=" * 80)
    
    # ダミーデータでテスト
    B = 4
    n_heads = 4
    L = 3000
    num_intervals = 30
    points_per_interval = L // num_intervals
    
    # ダミーのアテンションウェイト（L+1, L+1）を生成
    # attention_weights: (B, n_heads, L+1, L+1)
    attention_weights = torch.randn(B, n_heads, L+1, L+1)
    attention_weights = F.softmax(attention_weights, dim=-1)  # 正規化
    
    print("\n【学習時の計算方法（train.py）】")
    # 学習時の計算
    cls_attention_full_train = attention_weights[:, :, 0, 1:]  # (B, n_heads, L)
    cls_attention_mean_train = cls_attention_full_train.mean(dim=1)  # (B, L)
    
    interval_attention_list_train = []
    for i in range(B):
        interval_attn = []
        for j in range(num_intervals):
            start_idx = j * points_per_interval
            end_idx = min(start_idx + points_per_interval, L)
            attn = cls_attention_mean_train[i, start_idx:end_idx].mean()
            interval_attn.append(attn)
        interval_attention_list_train.append(torch.stack(interval_attn))
    
    interval_attention_train = torch.stack(interval_attention_list_train)  # (B, num_intervals)
    interval_attention_normalized_train = F.normalize(interval_attention_train, p=1, dim=1)  # (B, num_intervals)
    
    print(f"  interval_attention_train.shape: {interval_attention_train.shape}")
    print(f"  interval_attention_normalized_train.shape: {interval_attention_normalized_train.shape}")
    print(f"  正規化後の合計（各サンプル）: {interval_attention_normalized_train.sum(dim=1)}")
    
    print("\n【評価時の計算方法（eval.py - get_cls_attention_to_intervals）】")
    # 評価時の計算（get_cls_attention_to_intervalsメソッドを使用）
    cls_attention_full_eval = attention_weights[:, :, 0, 1:]  # (B, n_heads, L)
    cls_attention_mean_eval = cls_attention_full_eval.mean(dim=1)  # (B, L)
    
    cls_attention_intervals_eval = []
    for i in range(num_intervals):
        start_idx = i * points_per_interval
        end_idx = min(start_idx + points_per_interval, L)
        interval_attn = cls_attention_mean_eval[:, start_idx:end_idx].mean(dim=1)
        cls_attention_intervals_eval.append(interval_attn)
    
    cls_attention_eval = torch.stack(cls_attention_intervals_eval, dim=1)  # (B, num_intervals)
    
    # numpyでL1正規化（eval.pyと同じ方法）
    cls_attention_eval_np = cls_attention_eval.numpy()
    cls_attention_normalized_eval = np.zeros_like(cls_attention_eval_np)
    for i in range(B):
        row_sum = np.sum(np.abs(cls_attention_eval_np[i]))
        if row_sum > 0:
            cls_attention_normalized_eval[i] = cls_attention_eval_np[i] / row_sum
        else:
            cls_attention_normalized_eval[i] = cls_attention_eval_np[i]
    
    print(f"  cls_attention_eval.shape: {cls_attention_eval.shape}")
    print(f"  cls_attention_normalized_eval.shape: {cls_attention_normalized_eval.shape}")
    print(f"  正規化後の合計（各サンプル）: {cls_attention_normalized_eval.sum(axis=1)}")
    
    print("\n【比較】")
    # 正規化前の値を比較
    diff_before_normalization = torch.abs(interval_attention_train - cls_attention_eval).max().item()
    print(f"  正規化前の最大差: {diff_before_normalization:.10f}")
    if diff_before_normalization < 1e-6:
        print(f"  ✅ 正規化前の値は一致しています")
    else:
        print(f"  ⚠️ 警告: 正規化前の値が異なります！")
        print(f"     学習時と評価時で計算方法が異なる可能性があります。")
    
    # 正規化後の値を比較
    diff_after_normalization = np.abs(interval_attention_normalized_train.numpy() - cls_attention_normalized_eval).max()
    print(f"  正規化後の最大差: {diff_after_normalization:.10f}")
    if diff_after_normalization < 1e-6:
        print(f"  ✅ 正規化後の値は一致しています")
    else:
        print(f"  ⚠️ 警告: 正規化後の値が異なります！")
        print(f"     正規化の実装が異なる可能性があります。")
    
    # F.normalizeとnumpyのL1正規化の比較
    print("\n【F.normalizeとnumpyのL1正規化の比較】")
    test_tensor = torch.randn(B, num_intervals)
    normalized_pytorch = F.normalize(test_tensor, p=1, dim=1)
    
    test_numpy = test_tensor.numpy()
    normalized_numpy = np.zeros_like(test_numpy)
    for i in range(B):
        row_sum = np.sum(np.abs(test_numpy[i]))
        if row_sum > 0:
            normalized_numpy[i] = test_numpy[i] / row_sum
        else:
            normalized_numpy[i] = test_numpy[i]
    
    diff_normalization = np.abs(normalized_pytorch.numpy() - normalized_numpy).max()
    print(f"  F.normalizeとnumpyのL1正規化の最大差: {diff_normalization:.10f}")
    if diff_normalization < 1e-6:
        print(f"  ✅ F.normalizeとnumpyのL1正規化は一致しています")
    else:
        print(f"  ⚠️ 警告: F.normalizeとnumpyのL1正規化が異なります！")
    
    print("\n" + "=" * 80)
    print("比較完了")
    print("=" * 80)

if __name__ == "__main__":
    compare_attention_calculation_methods()

