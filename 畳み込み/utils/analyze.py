"""
ノイズの大きさと検知可能性を分析するスクリプト
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

# データセットの読み込み
print("データセットを読み込み中...")
with open('baseline_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

train_data = dataset['train']['data']
train_labels = dataset['train']['labels']

# Tensorに変換
if not isinstance(train_data, torch.Tensor):
    train_data = torch.FloatTensor(train_data)
if not isinstance(train_labels, torch.Tensor):
    train_labels = torch.LongTensor(train_labels)

# 元のデータを読み込み（ノイズなし）
print("\n元のデータを読み込み中...")
with open('data_lowF_noise.pickle', 'rb') as f:
    original_data = pickle.load(f)

original_psd = original_data['x'][:, 0, :]  # (32000, 3000)
if not isinstance(original_psd, torch.Tensor):
    original_psd = torch.FloatTensor(original_psd)

# 対応するサンプルを取得
sample_indices = []
for i in range(10):
    indices = torch.where(train_labels == i)[0]
    if len(indices) > 0:
        sample_indices.append(indices[0].item())

print(f"\nサンプルインデックス: {sample_indices}")

# ノイズの大きさを分析
print("\n" + "=" * 60)
print("ノイズの大きさの分析")
print("=" * 60)

noise_ratios_before = []
noise_ratios_after = []
noise_diffs_before = []
noise_diffs_after = []

for idx in sample_indices[:10]:  # 最初の10サンプルを分析
    noisy_sample = train_data[idx]
    label = train_labels[idx].item()
    
    # 元のデータを取得（同じインデックス）
    original_sample = original_psd[idx]
    
    # ノイズ区間
    start_idx = label * 300
    end_idx = start_idx + 300
    
    # 前処理前の分析
    noisy_region_before = noisy_sample[start_idx:end_idx]
    original_region_before = original_sample[start_idx:end_idx]
    other_regions_before = torch.cat([noisy_sample[:start_idx], noisy_sample[end_idx:]])
    other_original_before = torch.cat([original_sample[:start_idx], original_sample[end_idx:]])
    
    # ノイズの大きさ（追加されたノイズ）
    added_noise = noisy_region_before - original_region_before
    noise_magnitude = added_noise.abs().mean()
    
    # ノイズ区間と他の区間の比較
    noisy_mean = noisy_region_before.mean()
    other_mean = other_regions_before.mean()
    
    if other_mean != 0:
        ratio = noisy_mean / other_mean
        noise_ratios_before.append(ratio.item())
    
    diff = abs(noisy_mean - other_mean)
    noise_diffs_before.append(diff.item())
    
    # ログ変換後の分析
    noisy_log = torch.log(noisy_sample.clamp(min=1e-30))
    original_log = torch.log(original_sample.clamp(min=1e-30))
    
    noisy_region_after = noisy_log[start_idx:end_idx]
    other_regions_after = torch.cat([noisy_log[:start_idx], noisy_log[end_idx:]])
    
    noisy_mean_after = noisy_region_after.mean()
    other_mean_after = other_regions_after.mean()
    
    if other_mean_after != 0:
        ratio_after = abs(noisy_mean_after / other_mean_after)
        noise_ratios_after.append(ratio_after.item())
    
    diff_after = abs(noisy_mean_after - other_mean_after)
    noise_diffs_after.append(diff_after.item())

print("\n【前処理前（元のスケール）】")
print(f"  ノイズ区間/他区間比の平均: {np.mean(noise_ratios_before):.2f}x")
print(f"  ノイズ区間/他区間比の範囲: [{np.min(noise_ratios_before):.2f}x, {np.max(noise_ratios_before):.2f}x]")
print(f"  ノイズ区間と他区間の差の平均: {np.mean(noise_diffs_before):.6e}")

print("\n【ログ変換後】")
print(f"  ノイズ区間/他区間比の平均: {np.mean(noise_ratios_after):.2f}x")
print(f"  ノイズ区間/他区間比の範囲: [{np.min(noise_ratios_after):.2f}x, {np.max(noise_ratios_after):.2f}x]")
print(f"  ノイズ区間と他区間の差の平均: {np.mean(noise_diffs_after):.4f}")

# 詳細な分析（1サンプル）
print("\n" + "=" * 60)
print("詳細な分析（サンプル1）")
print("=" * 60)

idx = sample_indices[0]
noisy_sample = train_data[idx]
original_sample = original_psd[idx]
label = train_labels[idx].item()

start_idx = label * 300
end_idx = start_idx + 300

# 前処理前
noisy_region = noisy_sample[start_idx:end_idx]
original_region = original_sample[start_idx:end_idx]
other_noisy = torch.cat([noisy_sample[:start_idx], noisy_sample[end_idx:]])
other_original = torch.cat([original_sample[:start_idx], original_sample[end_idx:]])

added_noise = noisy_region - original_region
noise_magnitude = added_noise.abs().mean()

print(f"\n【前処理前】")
print(f"  ラベル: {label} (ノイズ区間: {start_idx}-{end_idx})")
print(f"  元のデータ（ノイズ区間）の平均: {original_region.mean():.6e}")
print(f"  ノイズ付与後（ノイズ区間）の平均: {noisy_region.mean():.6e}")
print(f"  追加されたノイズの平均: {noise_magnitude:.6e}")
print(f"  ノイズの大きさ（元のデータに対する倍率）: {noise_magnitude / original_region.mean().abs():.2f}x")
print(f"  ノイズ区間/他区間比: {noisy_region.mean() / other_noisy.mean():.2f}x")

# ログ変換後
noisy_log = torch.log(noisy_sample.clamp(min=1e-30))
original_log = torch.log(original_sample.clamp(min=1e-30))

noisy_region_log = noisy_log[start_idx:end_idx]
original_region_log = original_log[start_idx:end_idx]
other_noisy_log = torch.cat([noisy_log[:start_idx], noisy_log[end_idx:]])
other_original_log = torch.cat([original_log[:start_idx], original_log[end_idx:]])

print(f"\n【ログ変換後】")
print(f"  元のデータ（ノイズ区間）の平均: {original_region_log.mean():.4f}")
print(f"  ノイズ付与後（ノイズ区間）の平均: {noisy_region_log.mean():.4f}")
print(f"  差: {abs(noisy_region_log.mean() - original_region_log.mean()):.4f}")
print(f"  ノイズ区間/他区間比: {abs(noisy_region_log.mean() / other_noisy_log.mean()):.2f}x")
print(f"  ノイズ区間と他区間の差: {abs(noisy_region_log.mean() - other_noisy_log.mean()):.4f}")

# 可視化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 前処理前：ノイズ区間と他の区間の比較
axes[0, 0].plot(noisy_sample.numpy(), alpha=0.7, label='ノイズ付与後', linewidth=1)
axes[0, 0].plot(original_sample.numpy(), alpha=0.7, label='元のデータ', linewidth=1)
axes[0, 0].axvspan(start_idx, end_idx, alpha=0.2, color='red', label='ノイズ区間')
axes[0, 0].set_title(f'前処理前（クラス {label}）')
axes[0, 0].set_xlabel('ポイント')
axes[0, 0].set_ylabel('値')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# 2. ログ変換後：ノイズ区間と他の区間の比較
axes[0, 1].plot(noisy_log.numpy(), alpha=0.7, label='ノイズ付与後（ログ）', linewidth=1)
axes[0, 1].plot(original_log.numpy(), alpha=0.7, label='元のデータ（ログ）', linewidth=1)
axes[0, 1].axvspan(start_idx, end_idx, alpha=0.2, color='red', label='ノイズ区間')
axes[0, 1].set_title(f'ログ変換後（クラス {label}）')
axes[0, 1].set_xlabel('ポイント')
axes[0, 1].set_ylabel('ログ値')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. ノイズ区間の拡大表示（前処理前）
axes[1, 0].plot(noisy_region.numpy(), alpha=0.7, label='ノイズ付与後', linewidth=2)
axes[1, 0].plot(original_region.numpy(), alpha=0.7, label='元のデータ', linewidth=2)
axes[1, 0].set_title(f'ノイズ区間の拡大（前処理前）')
axes[1, 0].set_xlabel('ポイント（ノイズ区間内）')
axes[1, 0].set_ylabel('値')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale('log')

# 4. ノイズ区間の拡大表示（ログ変換後）
axes[1, 1].plot(noisy_region_log.numpy(), alpha=0.7, label='ノイズ付与後（ログ）', linewidth=2)
axes[1, 1].plot(original_region_log.numpy(), alpha=0.7, label='元のデータ（ログ）', linewidth=2)
axes[1, 1].set_title(f'ノイズ区間の拡大（ログ変換後）')
axes[1, 1].set_xlabel('ポイント（ノイズ区間内）')
axes[1, 1].set_ylabel('ログ値')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('noise_magnitude_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ ノイズの大きさの分析を 'noise_magnitude_analysis.png' に保存しました")

# 検知可能性の評価
print("\n" + "=" * 60)
print("検知可能性の評価")
print("=" * 60)

# ログ変換後の差が十分大きいか
avg_diff_after = np.mean(noise_diffs_after)
avg_std_after = np.mean([torch.cat([
    torch.log(train_data[i].clamp(min=1e-30))[:train_labels[i]*300],
    torch.log(train_data[i].clamp(min=1e-30))[train_labels[i]*300+300:]
]).std().item() for i in sample_indices[:10]])

print(f"\nログ変換後の統計:")
print(f"  ノイズ区間と他区間の平均差: {avg_diff_after:.4f}")
print(f"  他区間の平均標準偏差: {avg_std_after:.4f}")
print(f"  差/標準偏差比: {avg_diff_after / avg_std_after:.2f}")

if avg_diff_after / avg_std_after > 3.0:
    print("\n✓ 検知可能: 差が標準偏差の3倍以上（統計的に有意）")
elif avg_diff_after / avg_std_after > 2.0:
    print("\n⚠ 検知可能だが困難: 差が標準偏差の2-3倍（検知可能だが難しい）")
elif avg_diff_after / avg_std_after > 1.0:
    print("\n⚠ 検知が困難: 差が標準偏差の1-2倍（検知が難しい）")
else:
    print("\n✗ 検知が非常に困難: 差が標準偏差の1倍未満（ほぼ検知不可能）")

plt.show()

print("\n✓ 分析が完了しました")

