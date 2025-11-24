"""
ランキング損失の詳細分析と修正

添付画像の式を正しく実装し、勾配が正しく計算されるか確認します。
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
print("【ランキング損失の詳細分析と修正】")
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


def compute_ranking_loss_correct(attention, noise_intervals, margin=0.01):
    """
    添付画像の式に基づいた正しい実装
    L_rank(A, Y_dot) = Σ (1 - Y_dot_ij * Y_dot_ik) * max(m, Y_dot_ij * a_ij + Y_dot_ik * a_ik)
    Y_dot_ij = -1 if Y_hat_i = Y_hat_j, +1 if Y_hat_i ≠ Y_hat_j
    
    ここで、a_ijは区間iのアテンション値、a_ikは区間kのアテンション値と解釈します。
    """
    B = attention.shape[0]
    ranking_loss = torch.tensor(0.0, device=attention.device)
    
    for b in range(B):
        noise_idx = noise_intervals[b].item()
        attns = attention[b]
        
        # Y_hat_iを定義（1=ノイズ, 0=正常）
        Y_hat = torch.zeros(num_intervals, dtype=torch.long, device=attention.device)
        Y_hat[noise_idx] = 1
        
        # すべての(i, j, k)の組み合わせを計算
        total_loss = torch.tensor(0.0, device=attention.device)
        count = 0
        
        for i in range(num_intervals):
            for j in range(num_intervals):
                if i == j:
                    continue
                for k in range(num_intervals):
                    if i == k or j == k:
                        continue
                    
                    # Y_dot_ijとY_dot_ikを計算
                    Y_dot_ij = -1 if Y_hat[i] == Y_hat[j] else 1
                    Y_dot_ik = -1 if Y_hat[i] == Y_hat[k] else 1
                    
                    # 重み (1 - Y_dot_ij * Y_dot_ik) を計算
                    weight = 1 - Y_dot_ij * Y_dot_ik
                    
                    # 重みが0でない場合のみ計算
                    if weight != 0:
                        # term = Y_dot_ij * a_i + Y_dot_ik * a_k
                        # ここで、a_iは区間iのアテンション値、a_kは区間kのアテンション値
                        term = Y_dot_ij * attns[i] + Y_dot_ik * attns[k]
                        
                        # max(m, term)
                        loss_term = torch.maximum(torch.tensor(margin, device=attention.device), term)
                        
                        # 重みを掛ける
                        total_loss = total_loss + weight * loss_term
                        count += 1
        
        if count > 0:
            ranking_loss = ranking_loss + total_loss / count
    
    return ranking_loss / B if B > 0 else torch.tensor(0.0, device=attention.device)


def compute_ranking_loss_simple_effective(attention, noise_intervals, margin=0.01):
    """
    シンプルで効果的な実装：ノイズ区間のアテンション < 正常区間の最小アテンションを保証
    これは、添付画像の式の本質的な意味を捉えた実装です。
    """
    B = attention.shape[0]
    ranking_loss = torch.tensor(0.0, device=attention.device)
    
    for b in range(B):
        noise_idx = noise_intervals[b].item()
        attns = attention[b]
        
        noise_attn = attns[noise_idx]
        normal_indices = [i for i in range(num_intervals) if i != noise_idx]
        normal_attns = attns[normal_indices]
        
        if len(normal_attns) == 0:
            continue
        
        # 正常区間の最小アテンション値
        normal_min_attn = normal_attns.min()
        
        # ノイズ区間のアテンション < 正常区間の最小アテンションを保証
        # term = noise_attn - normal_min_attn + margin
        # max(0, term) で損失を計算
        term = noise_attn - normal_min_attn + margin
        loss = torch.maximum(torch.tensor(0.0, device=attention.device), term)
        
        ranking_loss = ranking_loss + loss
    
    return ranking_loss / B if B > 0 else torch.tensor(0.0, device=attention.device)


# 勾配をテスト
print("\n" + "=" * 60)
print("【勾配のテスト】")
print("=" * 60)

# アテンション値をパラメータとして扱う
attention_param = nn.Parameter(attention_tensor.clone())

# オプティマイザーを設定
optimizer = optim.Adam([attention_param], lr=0.1)

# ランキング損失の重み
lambda_ranking = 300.0
margin = 0.01

# シンプルな実装でテスト
print("\nシンプルな実装でテスト:")
for epoch in range(5):
    optimizer.zero_grad()
    
    # ランキング損失を計算
    ranking_loss = compute_ranking_loss_simple_effective(attention_param, true_intervals_tensor, margin)
    
    # 総損失
    total_loss = lambda_ranking * ranking_loss
    
    # バックプロパゲーション
    total_loss.backward()
    
    # 勾配を確認
    grad_norm = torch.norm(attention_param.grad).item()
    
    # オプティマイザーステップ
    optimizer.step()
    
    # ランキング違反をチェック
    violations = []
    for i in range(num_samples):
        noise_idx = int(true_intervals[i])
        noise_attn = attention_param[i, noise_idx].item()
        normal_indices = [j for j in range(num_intervals) if j != noise_idx]
        normal_attns = attention_param[i, normal_indices].detach().numpy()
        normal_min_attn = normal_attns.min()
        
        if noise_attn > normal_min_attn:
            violations.append(i)
    
    print(f"エポック {epoch+1}:")
    print(f"  ランキング損失: {ranking_loss.item():.9f}")
    print(f"  総損失: {total_loss.item():.9f}")
    print(f"  勾配ノルム: {grad_norm:.9f}")
    print(f"  ランキング違反: {len(violations)} / {num_samples} ({len(violations)/num_samples*100:.2f}%)")
    
    if len(violations) == 0:
        print(f"  ✅ ランキング違反が解消されました！")
        break

print("\n" + "=" * 60)
print("【添付画像の式に基づいた実装でテスト】")
print("=" * 60)

# アテンション値を再初期化
attention_param = nn.Parameter(attention_tensor.clone())
optimizer = optim.Adam([attention_param], lr=0.1)

print("\n添付画像の式に基づいた実装でテスト:")
for epoch in range(5):
    optimizer.zero_grad()
    
    # ランキング損失を計算
    ranking_loss = compute_ranking_loss_correct(attention_param, true_intervals_tensor, margin)
    
    # 総損失
    total_loss = lambda_ranking * ranking_loss
    
    # バックプロパゲーション
    total_loss.backward()
    
    # 勾配を確認
    grad_norm = torch.norm(attention_param.grad).item()
    
    # オプティマイザーステップ
    optimizer.step()
    
    # ランキング違反をチェック
    violations = []
    for i in range(num_samples):
        noise_idx = int(true_intervals[i])
        noise_attn = attention_param[i, noise_idx].item()
        normal_indices = [j for j in range(num_intervals) if j != noise_idx]
        normal_attns = attention_param[i, normal_indices].detach().numpy()
        normal_min_attn = normal_attns.min()
        
        if noise_attn > normal_min_attn:
            violations.append(i)
    
    print(f"エポック {epoch+1}:")
    print(f"  ランキング損失: {ranking_loss.item():.9f}")
    print(f"  総損失: {total_loss.item():.9f}")
    print(f"  勾配ノルム: {grad_norm:.9f}")
    print(f"  ランキング違反: {len(violations)} / {num_samples} ({len(violations)/num_samples*100:.2f}%)")
    
    if len(violations) == 0:
        print(f"  ✅ ランキング違反が解消されました！")
        break

print("\n" + "=" * 60)

