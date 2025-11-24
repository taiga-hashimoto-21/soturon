"""
シンプルなランキング損失の実装でノイズ区間予測がうまくいくかテスト

シンプルな実装：
- ノイズ区間のアテンション < 正常区間の最小アテンションを保証
- term = noise_attn - normal_min_attn + margin
- loss = max(0, term)
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
print("【シンプルなランキング損失でノイズ区間予測をテスト】")
print("=" * 60)

# データを読み込む
with open(attention_data_path, 'rb') as f:
    attention_data = pickle.load(f)

predicted_attention_before = attention_data['predicted_attention_before_normalization']
true_intervals = attention_data['true_noise_intervals']
predicted_intervals = attention_data['predicted_noise_intervals_before_normalization']

# アテンション値をテンソルに変換
attention_tensor = torch.tensor(predicted_attention_before, dtype=torch.float32)
true_intervals_tensor = torch.tensor(true_intervals, dtype=torch.long)
predicted_intervals_tensor = torch.tensor(predicted_intervals, dtype=torch.long)

num_samples = attention_tensor.shape[0]
num_intervals = attention_tensor.shape[1]

print(f"\nサンプル数: {num_samples}")
print(f"区間数: {num_intervals}")


def compute_ranking_loss_simple(attention, noise_intervals, margin=0.01):
    """
    シンプルな実装：ノイズ区間のアテンション < 正常区間の最小アテンションを保証
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


def compute_noise_interval_prediction_loss(attention, noise_intervals):
    """
    ノイズ区間予測損失：ノイズ区間のアテンションを0に近づける
    """
    B = attention.shape[0]
    noise_interval_loss = torch.tensor(0.0, device=attention.device)
    
    for b in range(B):
        noise_idx = noise_intervals[b].item()
        noise_attn = attention[b, noise_idx]
        target = torch.tensor(0.0, device=attention.device)
        noise_interval_loss = noise_interval_loss + (noise_attn - target) ** 2
    
    return noise_interval_loss / B if B > 0 else torch.tensor(0.0, device=attention.device)


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


def predict_noise_interval(attention):
    """アテンション値からノイズ区間を予測（最小値を持つ区間）"""
    predictions = []
    for i in range(attention.shape[0]):
        pred = torch.argmin(attention[i]).item()
        predictions.append(pred)
    return np.array(predictions)


def calculate_accuracy(predictions, true_intervals):
    """予測精度を計算"""
    correct = np.sum(predictions == true_intervals)
    return correct / len(predictions)


# 初期状態を確認
print("\n" + "=" * 60)
print("【初期状態の確認】")
print("=" * 60)

initial_violations = check_ranking_violations(attention_tensor, true_intervals_tensor)
initial_predictions = predict_noise_interval(attention_tensor)
initial_accuracy = calculate_accuracy(initial_predictions, true_intervals)

print(f"ランキング違反の数: {len(initial_violations)} / {num_samples} ({len(initial_violations)/num_samples*100:.2f}%)")
print(f"ノイズ区間予測精度: {initial_accuracy*100:.2f}%")
print(f"正解数: {np.sum(initial_predictions == true_intervals)} / {num_samples}")

# 学習をシミュレート
print("\n" + "=" * 60)
print("【学習シミュレーション（シンプルなランキング損失）】")
print("=" * 60)

# アテンション値をパラメータとして扱う
attention_param = nn.Parameter(attention_tensor.clone())

# オプティマイザーを設定
optimizer = optim.Adam([attention_param], lr=0.1)

# 損失の重み
lambda_noise_interval = 500.0
lambda_ranking = 300.0
margin = 0.01

num_epochs = 20
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # ノイズ区間予測損失
    noise_interval_loss = compute_noise_interval_prediction_loss(attention_param, true_intervals_tensor)
    
    # ランキング損失
    ranking_loss = compute_ranking_loss_simple(attention_param, true_intervals_tensor, margin)
    
    # 総損失
    total_loss = lambda_noise_interval * noise_interval_loss + lambda_ranking * ranking_loss
    
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
    
    # 予測精度を計算
    predictions = predict_noise_interval(attention_param)
    accuracy = calculate_accuracy(predictions, true_intervals)
    
    print(f"\nエポック {epoch+1}:")
    print(f"  ノイズ区間予測損失: {noise_interval_loss.item():.9f}")
    print(f"  ランキング損失: {ranking_loss.item():.9f}")
    print(f"  総損失: {total_loss.item():.9f}")
    print(f"  勾配ノルム: {grad_norm:.9f}")
    print(f"  ランキング違反: {len(violations)} / {num_samples} ({len(violations)/num_samples*100:.2f}%)")
    print(f"  ノイズ区間予測精度: {accuracy*100:.2f}%")
    print(f"  正解数: {np.sum(predictions == true_intervals)} / {num_samples}")
    
    if len(violations) == 0 and accuracy == 1.0:
        print(f"  ✅ ランキング違反が解消され、予測精度が100%になりました！")
        break
    elif len(violations) == 0:
        print(f"  ✅ ランキング違反が解消されました！")
    elif accuracy == 1.0:
        print(f"  ✅ 予測精度が100%になりました！")

print("\n" + "=" * 60)
print("【結果】")
print("=" * 60)

final_violations = check_ranking_violations(attention_param, true_intervals_tensor)
final_predictions = predict_noise_interval(attention_param)
final_accuracy = calculate_accuracy(final_predictions, true_intervals)

print(f"\n最終的なランキング違反の数: {len(final_violations)} / {num_samples} ({len(final_violations)/num_samples*100:.2f}%)")
print(f"最終的なノイズ区間予測精度: {final_accuracy*100:.2f}%")
print(f"正解数: {np.sum(final_predictions == true_intervals)} / {num_samples}")

if len(final_violations) == 0:
    print("\n✅ ランキング違反が解消されました！")
else:
    print("\n❌ ランキング違反が残っています。")

if final_accuracy == 1.0:
    print("✅ ノイズ区間予測精度が100%になりました！")
elif final_accuracy > initial_accuracy:
    print(f"✅ ノイズ区間予測精度が改善しました！（{initial_accuracy*100:.2f}% → {final_accuracy*100:.2f}%）")
else:
    print(f"❌ ノイズ区間予測精度が改善しませんでした。（{initial_accuracy*100:.2f}% → {final_accuracy*100:.2f}%）")

print("\n" + "=" * 60)
print("【結論】")
print("=" * 60)
if len(final_violations) == 0 and final_accuracy >= initial_accuracy:
    print("✅ シンプルなランキング損失の実装で、ランキング違反が解消され、")
    print("   ノイズ区間予測精度が改善または維持されました。")
    print("   この実装を使用して、train.pyを修正します。")
else:
    print("❌ シンプルなランキング損失の実装でも、期待通りの結果が得られませんでした。")
    print("   別のアプローチを検討する必要があります。")

print("\n" + "=" * 60)

