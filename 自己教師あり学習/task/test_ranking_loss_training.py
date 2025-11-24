"""
ランキング損失の学習シミュレーション

ランキング損失が正しく機能するか、簡単な学習をシミュレートしてテストします。
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
print("【ランキング損失の学習シミュレーション】")
print("=" * 60)

# データを読み込む
with open(attention_data_path, 'rb') as f:
    attention_data = pickle.load(f)

predicted_attention_before = attention_data['predicted_attention_before_normalization']
true_intervals = attention_data['true_noise_intervals']

# アテンション値をテンソルに変換
attention_tensor = torch.tensor(predicted_attention_before, dtype=torch.float32, requires_grad=True)
true_intervals_tensor = torch.tensor(true_intervals, dtype=torch.long)

num_samples = attention_tensor.shape[0]
num_intervals = attention_tensor.shape[1]

print(f"\nサンプル数: {num_samples}")
print(f"区間数: {num_intervals}")


def compute_ranking_loss_optimized(attention, noise_intervals, margin=0.01):
    """
    添付画像の式に基づいた実装（最適化版）
    重みが0でない場合のみ計算（Y_dot_ij * Y_dot_ik = -1 の場合）
    """
    B = attention.shape[0]
    ranking_loss = torch.tensor(0.0, device=attention.device, requires_grad=True)
    
    for b in range(B):
        noise_idx = noise_intervals[b].item()
        attns = attention[b]
        
        # Y_hat_iを定義（1=ノイズ, 0=正常）
        Y_hat = torch.zeros(num_intervals, dtype=torch.long, device=attention.device)
        Y_hat[noise_idx] = 1
        
        # 重みが0でない場合のみ計算（Y_dot_ij * Y_dot_ik = -1 の場合）
        # これは以下の場合に発生：
        # 1. Y_dot_ij = +1, Y_dot_ik = -1 → iとjは異なるクラス、iとkは同じクラス
        # 2. Y_dot_ij = -1, Y_dot_ik = +1 → iとjは同じクラス、iとkは異なるクラス
        
        total_loss = torch.tensor(0.0, device=attention.device, requires_grad=True)
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
                        loss_term = torch.maximum(torch.tensor(margin, device=attention.device), term)
                        total_loss = total_loss + weight * loss_term
                        count += 1
        
        if count > 0:
            ranking_loss = ranking_loss + total_loss / count
    
    return ranking_loss / B if B > 0 else torch.tensor(0.0, device=attention.device)


def compute_ranking_loss_simple(attention, noise_intervals, margin=0.01):
    """
    シンプルな実装：ノイズ区間のアテンション < 正常区間の最小アテンションを保証
    """
    B = attention.shape[0]
    ranking_loss = torch.tensor(0.0, device=attention.device, requires_grad=True)
    
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


# 初期状態を確認
print("\n" + "=" * 60)
print("【初期状態の確認】")
print("=" * 60)

initial_violations = check_ranking_violations(attention_tensor, true_intervals_tensor)
print(f"ランキング違反の数: {len(initial_violations)} / {num_samples} ({len(initial_violations)/num_samples*100:.2f}%)")

# 学習をシミュレート
print("\n" + "=" * 60)
print("【学習シミュレーション】")
print("=" * 60)

# アテンション値をパラメータとして扱う
attention_param = nn.Parameter(attention_tensor.clone())

# オプティマイザーを設定
optimizer = optim.Adam([attention_param], lr=0.01)

# ランキング損失の重み
lambda_ranking = 300.0
margin = 0.01

num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # ランキング損失を計算（添付画像の式を使用）
    ranking_loss = compute_ranking_loss_optimized(attention_param, true_intervals_tensor, margin)
    
    # 総損失
    total_loss = lambda_ranking * ranking_loss
    
    # バックプロパゲーション
    total_loss.backward()
    
    # 勾配クリッピング
    torch.nn.utils.clip_grad_norm_([attention_param], max_norm=1.0)
    
    # オプティマイザーステップ
    optimizer.step()
    
    # ランキング違反をチェック
    violations = check_ranking_violations(attention_param, true_intervals_tensor)
    
    print(f"エポック {epoch+1}:")
    print(f"  ランキング損失: {ranking_loss.item():.9f}")
    print(f"  総損失: {total_loss.item():.9f}")
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
else:
    print("❌ ランキング違反が残っています。")
    print("\n原因を調査します...")
    
    # 違反サンプルの詳細を確認
    print(f"\n違反サンプルの詳細（最初の5つ）:")
    for i in final_violations[:5]:
        noise_idx = int(true_intervals[i])
        noise_attn = attention_param[i, noise_idx].item()
        normal_indices = [j for j in range(num_intervals) if j != noise_idx]
        normal_attns = attention_param[i, normal_indices].detach().numpy()
        normal_min_attn = normal_attns.min()
        
        print(f"  サンプル {i}:")
        print(f"    ノイズ区間: {noise_idx}, アテンション値: {noise_attn:.9f}")
        print(f"    正常区間の最小アテンション値: {normal_min_attn:.9f}")
        print(f"    差: {noise_attn - normal_min_attn:.9f}")

print("\n" + "=" * 60)

