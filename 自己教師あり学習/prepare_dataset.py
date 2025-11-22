"""
SSL用の固定データセット準備スクリプト
畳み込みと同じ方法で、事前にデータセットを準備して保存
評価時に同じデータを使えるようにする
"""

import pickle
import torch
import numpy as np
import random
import sys
import os

# noiseモジュールをインポート（プロジェクトルートから）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# ノイズの付与(共通)フォルダをインポート
noise_module_path = os.path.join(project_root, 'ノイズの付与(共通)')
sys.path.insert(0, noise_module_path)
from add_noise import add_noise_to_interval


def process_noise(noise, clip_range=0.5, smoothing_factor=0.1):
    """
    ノイズを処理（tanhでクリッピング + スムージング）
    """
    scaled_noise = noise / clip_range
    processed_noise = torch.tanh(scaled_noise) * clip_range
    smoothed_noise = processed_noise * (1 - smoothing_factor) + noise * smoothing_factor
    return smoothed_noise


def add_structured_noise(psd_data, clip_range=0.5, smoothing_factor=0.1):
    """
    全体的な構造化ノイズを付与（実験データに近づけるため）
    """
    device = psd_data.device if isinstance(psd_data, torch.Tensor) else torch.device('cpu')
    
    if isinstance(psd_data, np.ndarray):
        psd_data = torch.from_numpy(psd_data).to(device)
    
    if psd_data.dim() == 1:
        psd_data = psd_data.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    L = psd_data.shape[-1]
    B = psd_data.shape[0] if psd_data.dim() > 1 else 1
    
    # 位置に応じて分散が変わる（0.2 → 0.3）
    x = torch.linspace(1, L, L, device=device)
    var = 0.2 + 0.1 * x / 1000
    var = torch.clamp(var, max=0.3)
    std = torch.sqrt(var)
    
    std = std.unsqueeze(0).expand(B, L)
    noise = torch.normal(mean=0.0, std=std).to(device)
    
    processed_noise = process_noise(noise, clip_range=clip_range, smoothing_factor=smoothing_factor)
    noisy_psd = psd_data * (1 + processed_noise)
    
    if squeeze_output:
        noisy_psd = noisy_psd.squeeze(0)
    
    return noisy_psd


# パラメータ設定（畳み込みと同じ）
NUM_INTERVALS = 30
NOISE_LEVEL = 0.3
NOISE_TYPE = 'power_supply'  # 'power_supply', 'interference', 'clock_leakage' から選択（USE_RANDOM_NOISE=Falseの場合のみ使用）
NOISE_TYPES = ['power_supply', 'interference', 'clock_leakage']
USE_RANDOM_NOISE = True  # Trueにするとランダムに選択（推奨：学習データの多様性向上）
ADD_STRUCTURED_NOISE = True
STRUCTURED_NOISE_CLIP_RANGE = 0.5
STRUCTURED_NOISE_SMOOTHING_FACTOR = 0.1
MASK_RATIO = 0.15  # SSL用：マスクする区間の割合
SCALE_FACTOR = 2.5e24

# シードを固定（再現性のため）
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("=" * 60)
print("SSL用固定データセットの準備")
print("=" * 60)
print(f"区間数: {NUM_INTERVALS}")
if USE_RANDOM_NOISE:
    print(f"ノイズタイプ: ランダム（power_supply, interference, clock_leakage）")
else:
    print(f"ノイズタイプ: {NOISE_TYPE}")
print(f"ノイズレベル: {NOISE_LEVEL}")
print(f"構造化ノイズ: {ADD_STRUCTURED_NOISE}")
print(f"マスク比率: {MASK_RATIO}")

# データの読み込み
print("\nデータを読み込み中...")
with open('data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)

x = data['x']  # PSDデータ (32000, 1, 3000)
num_samples = x.shape[0]

print(f"サンプル数: {num_samples:,}")

# データセットの準備
print("\nノイズを付与中...")
noisy_data = []
labels = []
masks = []  # SSL用：マスク位置

points_per_interval = 3000 // NUM_INTERVALS

for i in range(num_samples):
    # 元のPSDデータ
    original_psd = x[i, 0, :]
    original_psd_tensor = torch.from_numpy(original_psd) if isinstance(original_psd, np.ndarray) else original_psd
    
    # 1. 構造化ノイズを付与
    if ADD_STRUCTURED_NOISE:
        structured_noisy_psd = add_structured_noise(
            original_psd_tensor,
            clip_range=STRUCTURED_NOISE_CLIP_RANGE,
            smoothing_factor=STRUCTURED_NOISE_SMOOTHING_FACTOR
        )
    else:
        structured_noisy_psd = original_psd_tensor
    
    # 2. ノイズタイプを選択（ランダムまたは固定）
    if USE_RANDOM_NOISE:
        selected_noise_type = random.choice(NOISE_TYPES)
    else:
        selected_noise_type = NOISE_TYPE
    
    # 3. ノイズタイプに応じて、主要な周波数を決定
    # 周波数範囲: 0-15000Hz, 3000ポイント → 1ポイント = 5Hz
    # 30区間、1区間 = 100ポイント = 500Hz
    freq_min = 0.0
    freq_max = 15000.0
    num_points = 3000
    points_per_interval = num_points // NUM_INTERVALS  # 100ポイント
    
    if selected_noise_type == 'power_supply':
        # 電源ノイズ: 2kHzのスイッチングノイズが最も目立つ
        main_freq = 2000.0
    elif selected_noise_type == 'interference':
        # 干渉ノイズ: 3000Hz
        main_freq = 3000.0
    elif selected_noise_type == 'clock_leakage':
        # クロックリークノイズ: 5000Hz
        main_freq = 5000.0
    else:
        # デフォルト（念のため）
        main_freq = 1000.0
    
    # 4. 周波数から区間インデックスを計算
    freq_to_point = main_freq / (freq_max / num_points)
    point_idx = int(freq_to_point)
    noise_interval = min(point_idx // points_per_interval, NUM_INTERVALS - 1)
    
    # 5. 区間ノイズを付与
    noisy_psd_tensor, start_idx, end_idx = add_noise_to_interval(
        structured_noisy_psd,
        noise_interval,
        noise_type=selected_noise_type,
        noise_level=NOISE_LEVEL,
        num_intervals=NUM_INTERVALS
    )
    
    # 4. マスク位置を決定（15%の区間をランダムにマスク、ノイズ区間は除外）
    available_intervals = [j for j in range(NUM_INTERVALS) if j != noise_interval]
    num_masked_intervals = max(1, int(len(available_intervals) * MASK_RATIO))
    num_masked_intervals = min(num_masked_intervals, len(available_intervals))
    
    masked_interval_indices = random.sample(available_intervals, num_masked_intervals)
    
    # マスク位置を計算
    mask_positions = np.zeros(3000, dtype=bool)
    for interval_idx in masked_interval_indices:
        start_idx_mask = interval_idx * points_per_interval
        end_idx_mask = min(start_idx_mask + points_per_interval, 3000)
        mask_positions[start_idx_mask:end_idx_mask] = True
    
    noisy_data.append(noisy_psd_tensor)
    labels.append(noise_interval)
    masks.append(mask_positions)
    
    if (i + 1) % 1000 == 0:
        print(f"  処理済み: {i+1:,} / {num_samples:,}")

# Tensorに変換
noisy_data = torch.stack(noisy_data)  # (32000, 3000)
labels = torch.tensor(labels, dtype=torch.long)  # (32000,)
masks = np.array(masks)  # (32000, 3000)

print(f"\nデータセットの形状:")
print(f"  入力データ: {noisy_data.shape}")
print(f"  ラベル: {labels.shape}")
print(f"  マスク: {masks.shape}")

# データの前処理: スケーリング + ログ変換 + 正規化
print("\nデータの前処理中（スケーリング + ログ変換 + 正規化）...")
noisy_data_scaled = noisy_data * SCALE_FACTOR
noisy_data_scaled = torch.clamp(noisy_data_scaled, min=1e-30)
noisy_data_log = torch.log(noisy_data_scaled)

# 正規化
train_mean = noisy_data_log.mean().item()
train_std = noisy_data_log.std().item()
if train_std < 1e-6:
    train_std = 1.0

noisy_data_norm = (noisy_data_log - train_mean) / (train_std + 1e-8)

print(f"  スケーリング後: 平均={noisy_data_scaled.mean():.6e}, 標準偏差={noisy_data_scaled.std():.6e}")
print(f"  ログ変換後: 平均={noisy_data_log.mean():.6f}, 標準偏差={noisy_data_log.std():.6f}")
print(f"  正規化後: 平均={noisy_data_norm.mean():.6f}, 標準偏差={noisy_data_norm.std():.6f}")

# データの分割（訓練:80%, 検証:10%, テスト:10%）
print("\nデータを分割中...")
indices = list(range(num_samples))
random.shuffle(indices)

train_size = int(num_samples * 0.8)
val_size = int(num_samples * 0.1)
test_size = num_samples - train_size - val_size

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

train_data = noisy_data_norm[train_indices]
train_labels = labels[train_indices]
train_masks = masks[train_indices]

val_data = noisy_data_norm[val_indices]
val_labels = labels[val_indices]
val_masks = masks[val_indices]

test_data = noisy_data_norm[test_indices]
test_labels = labels[test_indices]
test_masks = masks[test_indices]

print(f"訓練データ: {len(train_indices):,}サンプル")
print(f"検証データ: {len(val_indices):,}サンプル")
print(f"テストデータ: {len(test_indices):,}サンプル")

# ラベルの分布を確認
print("\nラベルの分布（訓練データ）:")
label_counts_full = torch.bincount(train_labels)
# NUM_INTERVALSのサイズに合わせて拡張（存在しない区間は0）
label_counts = torch.zeros(NUM_INTERVALS, dtype=torch.long)
for i in range(min(NUM_INTERVALS, len(label_counts_full))):
    label_counts[i] = label_counts_full[i]
for i in range(NUM_INTERVALS):
    print(f"  区間 {i+1:2d}: {label_counts[i].item():4d}サンプル")

# データセットを保存
print("\nデータセットを保存中...")
dataset = {
    'train': {
        'data': train_data,
        'labels': train_labels,
        'masks': train_masks
    },
    'val': {
        'data': val_data,
        'labels': val_labels,
        'masks': val_masks
    },
    'test': {
        'data': test_data,
        'labels': test_labels,
        'masks': test_masks
    },
    'config': {
        'num_intervals': NUM_INTERVALS,
        'noise_type': NOISE_TYPE,
        'use_random_noise': USE_RANDOM_NOISE,
        'noise_types': NOISE_TYPES,
        'noise_level': NOISE_LEVEL,
        'add_structured_noise': ADD_STRUCTURED_NOISE,
        'structured_noise_clip_range': STRUCTURED_NOISE_CLIP_RANGE,
        'structured_noise_smoothing_factor': STRUCTURED_NOISE_SMOOTHING_FACTOR,
        'mask_ratio': MASK_RATIO,
        'scale_factor': SCALE_FACTOR,
        'normalization_mean': train_mean,
        'normalization_std': train_std
    }
}

with open('ssl_dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)

print("データセットを 'ssl_dataset.pickle' に保存しました")
print("\nデータセット準備完了！")

