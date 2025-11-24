"""
ランキング損失の仮説検証スクリプト

添付画像の式に基づいて、ランキング損失が正しく機能するかテストします。
L_rank(A, Y_dot) = Σ (1 - Y_dot_ij * Y_dot_ik) * max(m, Y_dot_ij * a_ij + Y_dot_ik * a_ik)
Y_dot_ij = -1 if Y_hat_i = Y_hat_j, +1 if Y_hat_i ≠ Y_hat_j
"""

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Desktopのデータを読み込む
desktop_path = Path("/Users/hashimototaiga/Desktop")
attention_data_path = desktop_path / "attention_data.pkl"
accuracies_path = desktop_path / "train_noise_interval_accuracies.pkl"

print("=" * 60)
print("【ランキング損失の仮説検証】")
print("=" * 60)

# データを読み込む
with open(attention_data_path, 'rb') as f:
    attention_data = pickle.load(f)

predicted_attention_before = attention_data['predicted_attention_before_normalization']
true_intervals = attention_data['true_noise_intervals']
predicted_intervals = attention_data['predicted_noise_intervals_before_normalization']

print(f"\nデータ形状:")
print(f"  アテンション値: {predicted_attention_before.shape}")
print(f"  真のノイズ区間: {true_intervals.shape}")
print(f"  予測されたノイズ区間: {predicted_intervals.shape}")

# アテンション値をテンソルに変換
attention_tensor = torch.tensor(predicted_attention_before, dtype=torch.float32)
true_intervals_tensor = torch.tensor(true_intervals, dtype=torch.long)
predicted_intervals_tensor = torch.tensor(predicted_intervals, dtype=torch.long)

num_samples = attention_tensor.shape[0]
num_intervals = attention_tensor.shape[1]

print(f"\nサンプル数: {num_samples}")
print(f"区間数: {num_intervals}")


def compute_ranking_loss_v1(attention, noise_intervals, margin=0.01):
    """
    現在の実装（ケース2とケース3のみ）
    """
    B = attention.shape[0]
    ranking_loss = torch.tensor(0.0)
    
    for b in range(B):
        noise_idx = noise_intervals[b].item()
        attns = attention[b]
        
        normal_indices_list = [i for i in range(num_intervals) if i != noise_idx]
        normal_indices = torch.tensor(normal_indices_list, dtype=torch.long)
        normal_attns = attns[normal_indices]
        noise_attn = attns[noise_idx]
        num_normal = len(normal_indices)
        
        if num_normal < 2:
            continue
        
        # ケース2: i=正常, j=ノイズ, k=正常（iとkは異なる正常区間）
        i_indices = normal_indices.unsqueeze(1).expand(-1, num_normal)
        k_indices = normal_indices.unsqueeze(0).expand(num_normal, -1)
        mask = i_indices != k_indices
        
        if mask.any():
            a_i_expanded = normal_attns.unsqueeze(1).expand(-1, num_normal)
            a_k_expanded = normal_attns.unsqueeze(0).expand(num_normal, -1)
            terms_case2 = (a_i_expanded - a_k_expanded)[mask]
            loss_case2 = torch.maximum(
                torch.tensor(margin), 
                terms_case2
            ).sum()
        else:
            loss_case2 = torch.tensor(0.0)
        
        # ケース3: i=正常, j=正常, k=ノイズ（iとjは異なる正常区間）
        if mask.any():
            a_i_expanded = normal_attns.unsqueeze(1).expand(-1, num_normal)
            terms_case3 = (-a_i_expanded + noise_attn)[mask]
            loss_case3 = torch.maximum(
                torch.tensor(margin), 
                terms_case3
            ).sum()
        else:
            loss_case3 = torch.tensor(0.0)
        
        num_pairs = mask.sum().item() * 2
        if num_pairs > 0:
            sample_loss = (loss_case2 + loss_case3) / num_pairs
            ranking_loss += sample_loss
    
    return ranking_loss / B if B > 0 else torch.tensor(0.0)


def compute_ranking_loss_v2(attention, noise_intervals, margin=0.01):
    """
    添付画像の式に基づいた実装（完全版）
    L_rank(A, Y_dot) = Σ (1 - Y_dot_ij * Y_dot_ik) * max(m, Y_dot_ij * a_ij + Y_dot_ik * a_ik)
    Y_dot_ij = -1 if Y_hat_i = Y_hat_j, +1 if Y_hat_i ≠ Y_hat_j
    
    ここで、Y_hat_iは区間iがノイズ区間かどうかを示す（1=ノイズ, 0=正常）
    """
    B = attention.shape[0]
    ranking_loss = torch.tensor(0.0)
    
    for b in range(B):
        noise_idx = noise_intervals[b].item()
        attns = attention[b]
        
        # Y_hat_iを定義（1=ノイズ, 0=正常）
        Y_hat = torch.zeros(num_intervals, dtype=torch.long)
        Y_hat[noise_idx] = 1
        
        # すべての(i, j, k)の組み合わせを計算
        total_loss = torch.tensor(0.0)
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
                        # term = Y_dot_ij * a_ij + Y_dot_ik * a_ik
                        term = Y_dot_ij * attns[i] + Y_dot_ik * attns[k]
                        
                        # max(m, term)
                        loss_term = torch.maximum(torch.tensor(margin), term)
                        
                        # 重みを掛ける
                        total_loss += weight * loss_term
                        count += 1
        
        if count > 0:
            ranking_loss += total_loss / count
    
    return ranking_loss / B if B > 0 else torch.tensor(0.0)


def compute_ranking_loss_v3(attention, noise_intervals, margin=0.01):
    """
    最適化版：ノイズ区間と正常区間のペアのみを計算
    式を簡略化して、ノイズ区間のアテンション < 正常区間の最小アテンションを保証
    """
    B = attention.shape[0]
    ranking_loss = torch.tensor(0.0)
    
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
        loss = torch.maximum(torch.tensor(0.0), term)
        
        ranking_loss += loss
    
    return ranking_loss / B if B > 0 else torch.tensor(0.0)


def compute_ranking_loss_v4(attention, noise_intervals, margin=0.01):
    """
    添付画像の式に基づいた実装（最適化版）
    重みが0でない場合のみ計算（Y_dot_ij * Y_dot_ik = -1 の場合）
    """
    B = attention.shape[0]
    ranking_loss = torch.tensor(0.0)
    
    for b in range(B):
        noise_idx = noise_intervals[b].item()
        attns = attention[b]
        
        # Y_hat_iを定義（1=ノイズ, 0=正常）
        Y_hat = torch.zeros(num_intervals, dtype=torch.long)
        Y_hat[noise_idx] = 1
        
        # 重みが0でない場合のみ計算（Y_dot_ij * Y_dot_ik = -1 の場合）
        # これは以下の場合に発生：
        # 1. Y_dot_ij = +1, Y_dot_ik = -1 → iとjは異なるクラス、iとkは同じクラス
        # 2. Y_dot_ij = -1, Y_dot_ik = +1 → iとjは同じクラス、iとkは異なるクラス
        
        total_loss = torch.tensor(0.0)
        count = 0
        
        for i in range(num_intervals):
            for j in range(num_intervals):
                if i == j:
                    continue
                for k in range(num_intervals):
                    if i == k or j == k:
                        continue
                    
                    Y_dot_ij = -1 if Y_hat[i] == Y_hat[j] else 1
                    Y_dot_ik = -1 if Y_hat[i] == Y_hat[k] else 1
                    
                    # 重みが0でない場合のみ計算
                    if Y_dot_ij * Y_dot_ik == -1:
                        weight = 1 - Y_dot_ij * Y_dot_ik  # = 2
                        term = Y_dot_ij * attns[i] + Y_dot_ik * attns[k]
                        loss_term = torch.maximum(torch.tensor(margin), term)
                        total_loss += weight * loss_term
                        count += 1
        
        if count > 0:
            ranking_loss += total_loss / count
    
    return ranking_loss / B if B > 0 else torch.tensor(0.0)


# 各実装をテスト
print("\n" + "=" * 60)
print("【各実装のテスト】")
print("=" * 60)

margin = 0.01

print("\n1. 現在の実装（v1）:")
loss_v1 = compute_ranking_loss_v1(attention_tensor, true_intervals_tensor, margin)
print(f"   損失: {loss_v1.item():.9f}")

print("\n2. 添付画像の式に基づいた実装（完全版、v2）:")
loss_v2 = compute_ranking_loss_v2(attention_tensor, true_intervals_tensor, margin)
print(f"   損失: {loss_v2.item():.9f}")

print("\n3. 最適化版（v3）:")
loss_v3 = compute_ranking_loss_v3(attention_tensor, true_intervals_tensor, margin)
print(f"   損失: {loss_v3.item():.9f}")

print("\n4. 添付画像の式に基づいた実装（最適化版、v4）:")
loss_v4 = compute_ranking_loss_v4(attention_tensor, true_intervals_tensor, margin)
print(f"   損失: {loss_v4.item():.9f}")

# ランキング違反をチェック
print("\n" + "=" * 60)
print("【ランキング違反のチェック】")
print("=" * 60)

ranking_violations = []
for i in range(num_samples):
    noise_idx = int(true_intervals[i])
    noise_attn = attention_tensor[i, noise_idx]
    normal_indices = [j for j in range(num_intervals) if j != noise_idx]
    normal_attns = attention_tensor[i, normal_indices]
    normal_min_attn = normal_attns.min()
    
    if noise_attn > normal_min_attn:
        ranking_violations.append(i)

print(f"\nランキング違反の数: {len(ranking_violations)} / {num_samples} ({len(ranking_violations)/num_samples*100:.2f}%)")

# 各実装での損失とランキング違反の関係を確認
print("\n" + "=" * 60)
print("【ランキング違反サンプルでの損失】")
print("=" * 60)

if len(ranking_violations) > 0:
    violation_indices = torch.tensor(ranking_violations)
    violation_attention = attention_tensor[ranking_violations]
    violation_true_intervals = true_intervals_tensor[ranking_violations]
    
    print(f"\nランキング違反サンプルでの損失:")
    loss_v1_violation = compute_ranking_loss_v1(violation_attention, violation_true_intervals, margin)
    loss_v2_violation = compute_ranking_loss_v2(violation_attention, violation_true_intervals, margin)
    loss_v3_violation = compute_ranking_loss_v3(violation_attention, violation_true_intervals, margin)
    loss_v4_violation = compute_ranking_loss_v4(violation_attention, violation_true_intervals, margin)
    
    print(f"  v1: {loss_v1_violation.item():.9f}")
    print(f"  v2: {loss_v2_violation.item():.9f}")
    print(f"  v3: {loss_v3_violation.item():.9f}")
    print(f"  v4: {loss_v4_violation.item():.9f}")

print("\n" + "=" * 60)

