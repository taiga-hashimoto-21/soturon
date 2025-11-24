"""
ランキング損失の最終的な実装

添付画像の式を正しく解釈し、ランキング違反が解消される実装を作成します。
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Desktopのデータを読み込む
desktop_path = Path("/Users/hashimototaiga/Desktop")
attention_data_path = desktop_path / "attention_data.pkl"

print("=" * 60)
print("【ランキング損失の最終的な実装】")
print("=" * 60)

# データを読み込む
with open(attention_data_path, 'rb') as f:
    attention_data = pickle.load(f)

predicted_attention_before = attention_data['predicted_attention_before_normalization']
true_intervals = attention_data['true_noise_intervals']

# アテンション値をテンソルに変換
attention_tensor = torch.tensor(predicted_attention_before, dtype=torch.float32)
true_intervals_tensor = torch.tensor(true_intervals, dtype=torch.long)

num_samples = attention_tensor.shape[0]
num_intervals = attention_tensor.shape[1]

print(f"\nサンプル数: {num_samples}")
print(f"区間数: {num_intervals}")


def compute_ranking_loss_formula_based(attention, noise_intervals, margin=0.01):
    """
    添付画像の式に基づいた実装（修正版）
    L_rank(A, Y_dot) = Σ (1 - Y_dot_ij * Y_dot_ik) * max(m, Y_dot_ij * a_ij + Y_dot_ik * a_ik)
    Y_dot_ij = -1 if Y_hat_i = Y_hat_j, +1 if Y_hat_i ≠ Y_hat_j
    
    ここで、a_ijは区間iのアテンション値、a_ikは区間kのアテンション値と解釈します。
    しかし、添付画像の式の本質的な意味は：
    - ノイズ区間のアテンション < 正常区間の最小アテンションを保証すること
    
    そのため、以下のように解釈します：
    - i=正常区間、j=ノイズ区間、k=正常区間（i≠k）の場合
    - Y_dot_ij = +1（iとjは異なるクラス）
    - Y_dot_ik = -1（iとkは同じクラス）
    - term = Y_dot_ij * a_i + Y_dot_ik * a_k = a_i - a_k
    - これは、正常区間iのアテンション > 正常区間kのアテンションを保証
    
    - i=正常区間、j=正常区間（i≠j）、k=ノイズ区間の場合
    - Y_dot_ij = -1（iとjは同じクラス）
    - Y_dot_ik = +1（iとkは異なるクラス）
    - term = Y_dot_ij * a_i + Y_dot_ik * a_k = -a_i + a_k
    - これは、ノイズ区間kのアテンション < 正常区間iのアテンションを保証
    
    しかし、これでは複雑すぎるので、本質的な意味を捉えた実装にします。
    """
    B = attention.shape[0]
    ranking_loss = torch.tensor(0.0, device=attention.device)
    
    for b in range(B):
        noise_idx = noise_intervals[b].item()
        attns = attention[b]
        
        # Y_hat_iを定義（1=ノイズ, 0=正常）
        Y_hat = torch.zeros(num_intervals, dtype=torch.long, device=attention.device)
        Y_hat[noise_idx] = 1
        
        # 重みが0でない場合のみ計算（Y_dot_ij * Y_dot_ik = -1 の場合）
        # これは以下の場合に発生：
        # 1. i=正常, j=ノイズ, k=正常（i≠k）: Y_dot_ij=+1, Y_dot_ik=-1
        # 2. i=正常, j=正常（i≠j）, k=ノイズ: Y_dot_ij=-1, Y_dot_ik=+1
        
        total_loss = torch.tensor(0.0, device=attention.device)
        count = 0
        
        # ケース1: i=正常, j=ノイズ, k=正常（i≠k）
        normal_indices = [i for i in range(num_intervals) if i != noise_idx]
        for i_idx, i in enumerate(normal_indices):
            for k_idx, k in enumerate(normal_indices):
                if i == k:
                    continue
                
                Y_dot_ij = 1  # iとj（ノイズ）は異なるクラス
                Y_dot_ik = -1  # iとkは同じクラス（正常）
                
                # 重み (1 - Y_dot_ij * Y_dot_ik) = 1 - (1 * -1) = 2
                weight = 2.0
                
                # term = Y_dot_ij * a_i + Y_dot_ik * a_k = a_i - a_k
                term = Y_dot_ij * attns[i] + Y_dot_ik * attns[k]
                
                # max(m, term)
                loss_term = torch.maximum(torch.tensor(margin, device=attention.device), term)
                
                total_loss = total_loss + weight * loss_term
                count += 1
        
        # ケース2: i=正常, j=正常（i≠j）, k=ノイズ
        for i_idx, i in enumerate(normal_indices):
            for j_idx, j in enumerate(normal_indices):
                if i == j:
                    continue
                
                Y_dot_ij = -1  # iとjは同じクラス（正常）
                Y_dot_ik = 1  # iとk（ノイズ）は異なるクラス
                
                # 重み (1 - Y_dot_ij * Y_dot_ik) = 1 - (-1 * 1) = 2
                weight = 2.0
                
                # term = Y_dot_ij * a_i + Y_dot_ik * a_k = -a_i + a_k
                term = Y_dot_ij * attns[i] + Y_dot_ik * attns[noise_idx]
                
                # max(m, term)
                loss_term = torch.maximum(torch.tensor(margin, device=attention.device), term)
                
                total_loss = total_loss + weight * loss_term
                count += 1
        
        if count > 0:
            ranking_loss = ranking_loss + total_loss / count
    
    return ranking_loss / B if B > 0 else torch.tensor(0.0, device=attention.device)


def check_ranking_violations(attention, noise_intervals):
    """ランキング違反をチェック"""
    violations = []
    for i in range(attention.shape[0]):
        noise_idx = int(noise_intervals[i])
        noise_attn = attention[i, noise_idx].item()
        normal_indices = [j for j in range(num_intervals) if j != noise_idx]
        normal_attns = attention[i, normal_indices].detach().numpy()
        normal_min_attn = normal_attns.min()
        
        if noise_attn > normal_min_attn:
            violations.append(i)
    
    return violations


# 学習をシミュレート
print("\n" + "=" * 60)
print("【学習シミュレーション】")
print("=" * 60)

# アテンション値をパラメータとして扱う
attention_param = nn.Parameter(attention_tensor.clone())

# オプティマイザーを設定
optimizer = optim.Adam([attention_param], lr=0.1)

# ランキング損失の重み
lambda_ranking = 300.0
margin = 0.01

# 初期状態を確認
initial_violations = check_ranking_violations(attention_param, true_intervals_tensor)
print(f"\n初期状態:")
print(f"  ランキング違反: {len(initial_violations)} / {num_samples} ({len(initial_violations)/num_samples*100:.2f}%)")

num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # ランキング損失を計算
    ranking_loss = compute_ranking_loss_formula_based(attention_param, true_intervals_tensor, margin)
    
    # 総損失
    total_loss = lambda_ranking * ranking_loss
    
    # バックプロパゲーション
    total_loss.backward()
    
    # 勾配を確認
    grad_norm = torch.norm(attention_param.grad).item()
    
    # 勾配クリッピング
    torch.nn.utils.clip_grad_norm_([attention_param], max_norm=1.0)
    
    # オプティマイザーステップ
    optimizer.step()
    
    # ランキング違反をチェック
    violations = check_ranking_violations(attention_param, true_intervals_tensor)
    
    print(f"\nエポック {epoch+1}:")
    print(f"  ランキング損失: {ranking_loss.item():.9f}")
    print(f"  総損失: {total_loss.item():.9f}")
    print(f"  勾配ノルム: {grad_norm:.9f}")
    print(f"  ランキング違反: {len(violations)} / {num_samples} ({len(violations)/num_samples*100:.2f}%)")
    
    if len(violations) == 0:
        print(f"  ✅ ランキング違反が解消されました！")
        break

print("\n" + "=" * 60)
print("【結果】")
print("=" * 60)

final_violations = check_ranking_violations(attention_param, true_intervals_tensor)
print(f"最終的なランキング違反の数: {len(final_violations)} / {num_samples} ({len(final_violations)/num_samples*100:.2f}%)")

if len(final_violations) == 0:
    print("✅ ランキング違反が解消されました！")
    print("\nこの実装を使用して、train.pyを修正します。")
else:
    print("❌ ランキング違反が残っています。")
    print("\n別のアプローチを試します...")

print("\n" + "=" * 60)

