"""
データセットが正しく生成されているか確認するスクリプト
- ノイズが実際に付与されているか
- ラベルが正しいか
- データとラベルの対応が正しいか
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

# データセットの読み込み
print("=" * 60)
print("データセットの確認")
print("=" * 60)

with open('baseline_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

print(f"\n設定:")
print(f"  区間数: {dataset['config']['num_intervals']}")
print(f"  ノイズタイプ: {dataset['config']['noise_type']}")
print(f"  ノイズレベル: {dataset['config']['noise_level']}")

train_data = dataset['train']['data']
train_labels = dataset['train']['labels']

print(f"\nデータ形状:")
print(f"  訓練データ: {train_data.shape}")
print(f"  訓練ラベル: {train_labels.shape}")

# ラベルの分布を確認
print(f"\nラベルの分布:")
label_counts = torch.bincount(train_labels)
for i in range(len(label_counts)):
    print(f"  区間 {i}: {label_counts[i].item():,}サンプル ({label_counts[i].item()/len(train_labels)*100:.1f}%)")

# 元のデータを読み込み（比較用）
print("\n元のデータを読み込み中...")
with open('data_lowF_noise.pickle', 'rb') as f:
    original_data = pickle.load(f)

original_psd = original_data['x'][0, 0, :]  # 最初のサンプル

# いくつかのサンプルを確認
NUM_INTERVALS = dataset['config']['num_intervals']
POINTS_PER_INTERVAL = 3000 // NUM_INTERVALS

print(f"\n1区間あたりのポイント数: {POINTS_PER_INTERVAL}")

# 各クラスのサンプルを1つずつ確認
print("\n" + "=" * 60)
print("各クラスのサンプルを確認（ノイズが付与されているか）")
print("=" * 60)

fig, axes = plt.subplots(5, 2, figsize=(16, 20))
axes = axes.flatten()

for class_idx in range(min(10, NUM_INTERVALS)):
    # このクラスのサンプルを探す
    class_indices = (train_labels == class_idx).nonzero(as_tuple=True)[0]
    if len(class_indices) == 0:
        continue
    
    sample_idx = class_indices[0].item()
    noisy_psd = train_data[sample_idx]
    label = train_labels[sample_idx].item()
    
    # ノイズが付与された区間を計算
    start_idx = label * POINTS_PER_INTERVAL
    end_idx = start_idx + POINTS_PER_INTERVAL
    
    # 元のデータと比較（同じサンプルを使用）
    original_sample = original_data['x'][sample_idx % len(original_data['x']), 0, :]
    
    # 差分を計算
    diff = noisy_psd - original_sample
    
    # ノイズが付与された区間での差分
    noise_diff = diff[start_idx:end_idx]
    other_diff = torch.cat([diff[:start_idx], diff[end_idx:]])
    
    # 可視化
    ax = axes[class_idx]
    ax.plot(original_sample.numpy(), label='元データ', alpha=0.7, linewidth=1)
    ax.plot(noisy_psd.numpy(), label='ノイズ付与後', linewidth=1.5)
    ax.axvspan(start_idx, end_idx, alpha=0.2, color='red', label=f'区間{label+1}')
    ax.set_title(f'クラス {label} (区間 {label+1})\nノイズ差分: {noise_diff.abs().mean():.2e} (他: {other_diff.abs().mean():.2e})')
    ax.set_xlabel('周波数ポイント')
    ax.set_ylabel('PSD値')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ノイズが実際に付与されているか確認
    noise_magnitude = noise_diff.abs().mean().item()
    other_magnitude = other_diff.abs().mean().item()
    
    if noise_magnitude > other_magnitude * 1.5:
        print(f"✓ クラス {label}: ノイズが正しく付与されています (ノイズ区間: {noise_magnitude:.2e}, 他: {other_magnitude:.2e})")
    else:
        print(f"⚠ クラス {label}: ノイズが弱い可能性があります (ノイズ区間: {noise_magnitude:.2e}, 他: {other_magnitude:.2e})")

plt.tight_layout()
plt.savefig('dataset_verification.png', dpi=150, bbox_inches='tight')
print(f"\n可視化結果を 'dataset_verification.png' に保存しました")

# 統計情報
print("\n" + "=" * 60)
print("統計情報")
print("=" * 60)

# ノイズが付与された区間での差分の統計
all_noise_diffs = []
all_other_diffs = []

for i in range(min(100, len(train_data))):  # 最初の100サンプルを確認
    label = train_labels[i].item()
    start_idx = label * POINTS_PER_INTERVAL
    end_idx = start_idx + POINTS_PER_INTERVAL
    
    original_sample = original_data['x'][i % len(original_data['x']), 0, :]
    noisy_sample = train_data[i]
    
    diff = noisy_sample - original_sample
    noise_diff = diff[start_idx:end_idx]
    other_diff = torch.cat([diff[:start_idx], diff[end_idx:]])
    
    all_noise_diffs.append(noise_diff.abs().mean().item())
    all_other_diffs.append(other_diff.abs().mean().item())

all_noise_diffs = np.array(all_noise_diffs)
all_other_diffs = np.array(all_other_diffs)

print(f"\nノイズ区間での差分の平均: {all_noise_diffs.mean():.2e}")
print(f"その他の区間での差分の平均: {all_other_diffs.mean():.2e}")
print(f"比率: {all_noise_diffs.mean() / all_other_diffs.mean():.2f}x")

if all_noise_diffs.mean() > all_other_diffs.mean() * 1.5:
    print("✓ ノイズが正しく付与されています")
else:
    print("⚠ ノイズが弱い可能性があります。ノイズレベルを上げることを検討してください。")

print("\nデータセット確認完了！")


