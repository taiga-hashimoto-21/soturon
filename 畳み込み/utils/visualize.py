"""
前処理前後のデータを可視化するスクリプト
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

# 前処理前のデータ
data_before = train_data.clone()

# ログ変換を適用
data_after = torch.log(train_data.clamp(min=1e-30))

# サンプルを取得（各クラスから1つずつ）
samples_before = []
samples_after = []
sample_labels = []
for i in range(10):
    indices = torch.where(train_labels == i)[0]
    if len(indices) > 0:
        samples_before.append(data_before[indices[0]].cpu().numpy())
        samples_after.append(data_after[indices[0]].cpu().numpy())
        sample_labels.append(i)

# 可視化
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
fig.suptitle('前処理前後のデータ比較（各クラスのサンプル）', fontsize=16)

for i in range(10):
    row = i // 5
    col = i % 5
    ax = axes[row, col]
    
    # 前処理前
    ax.plot(samples_before[i], label='前処理前', alpha=0.7, linewidth=1, color='blue')
    
    # 前処理後（スケールを調整して表示）
    # ログ変換後のデータは負の値なので、オフセットを追加して表示
    log_data_scaled = samples_after[i] - samples_after[i].min()  # 最小値を0に
    log_data_scaled = log_data_scaled / log_data_scaled.max() * samples_before[i].max()  # スケールを合わせる
    ax.plot(log_data_scaled, label='前処理後（スケール調整）', alpha=0.7, linewidth=1, color='red')
    
    # ノイズ区間を強調
    label = sample_labels[i]
    start_idx = label * 300
    end_idx = start_idx + 300
    ax.axvspan(start_idx, end_idx, alpha=0.2, color='green', label='ノイズ区間')
    
    ax.set_title(f'クラス {label} (ノイズ区間: {start_idx}-{end_idx})')
    ax.set_xlabel('ポイント')
    ax.set_ylabel('値')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
print("✓ 前処理前後の比較を 'preprocessing_comparison.png' に保存しました")

# 別の図：前処理後のデータを直接表示
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
fig.suptitle('ログ変換後のデータ（各クラスのサンプル）', fontsize=16)

for i in range(10):
    row = i // 5
    col = i % 5
    ax = axes[row, col]
    
    # ログ変換後のデータを直接表示
    ax.plot(samples_after[i], alpha=0.7, linewidth=1, color='purple')
    
    # ノイズ区間を強調
    label = sample_labels[i]
    start_idx = label * 300
    end_idx = start_idx + 300
    ax.axvspan(start_idx, end_idx, alpha=0.2, color='green', label='ノイズ区間')
    
    ax.set_title(f'クラス {label} (ログ変換後)')
    ax.set_xlabel('ポイント')
    ax.set_ylabel('ログ値')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('preprocessing_after.png', dpi=150, bbox_inches='tight')
print("✓ ログ変換後のデータを 'preprocessing_after.png' に保存しました")

# 統計情報を表示
print("\n=== 前処理前の統計 ===")
print(f"平均: {data_before.mean():.6e}")
print(f"標準偏差: {data_before.std():.6e}")
print(f"最小値: {data_before.min():.6e}")
print(f"最大値: {data_before.max():.6e}")

print("\n=== ログ変換後の統計 ===")
print(f"平均: {data_after.mean():.4f}")
print(f"標準偏差: {data_after.std():.4f}")
print(f"最小値: {data_after.min():.4f}")
print(f"最大値: {data_after.max():.4f}")

# ノイズ区間と他の区間の差を確認（ログ変換後）
print("\n=== ノイズ区間と他の区間の差（ログ変換後、サンプル1） ===")
sample = data_after[0]
label = train_labels[0].item()
start_idx = label * 300
end_idx = start_idx + 300

noise_region = sample[start_idx:end_idx]
other_regions = torch.cat([sample[:start_idx], sample[end_idx:]])

print(f"ラベル: {label}")
print(f"ノイズ区間の平均: {noise_region.mean():.4f}")
print(f"ノイズ区間の標準偏差: {noise_region.std():.4f}")
print(f"他の区間の平均: {other_regions.mean():.4f}")
print(f"他の区間の標準偏差: {other_regions.std():.4f}")
print(f"差: {abs(noise_region.mean() - other_regions.mean()):.4f}")
if other_regions.mean() != 0:
    ratio = abs(noise_region.mean() / other_regions.mean())
    print(f"比: {ratio:.2f}x")

# ヒストグラムで分布を比較
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(data_before.flatten().cpu().numpy(), bins=100, alpha=0.7, color='blue', label='前処理前')
axes[0].set_xlabel('値')
axes[0].set_ylabel('頻度')
axes[0].set_title('前処理前のデータ分布')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(data_after.flatten().cpu().numpy(), bins=100, alpha=0.7, color='purple', label='ログ変換後')
axes[1].set_xlabel('ログ値')
axes[1].set_ylabel('頻度')
axes[1].set_title('ログ変換後のデータ分布')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('preprocessing_histogram.png', dpi=150, bbox_inches='tight')
print("✓ データ分布のヒストグラムを 'preprocessing_histogram.png' に保存しました")

plt.show()

print("\n✓ すべての可視化が完了しました")

