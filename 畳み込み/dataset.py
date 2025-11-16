"""
10クラス分類用のデータセット準備
32000点すべてにノイズを付与し、ラベルを作成

ノイズタイプ:
- 'frequency_band': 周波数帯域集中ノイズ（特定の周波数帯域に集中的に発生）
- 'localized_spike': 局所スパイクノイズ（一部のポイントに集中的に発生）
- 'amplitude_dependent': 振幅依存ノイズ（信号が大きい領域に集中的に発生）

修正内容:
- 構造化ノイズを追加（実験データに近づけるため）
- 正規化を追加（ログ変換後）
"""

import pickle
import torch
import numpy as np
import random
import sys
import os

# noiseモジュールをインポート（プロジェクトルートから）
# baseline/ から noise/ へのパス: ../noise/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from noise.add_noise import add_noise_to_interval


def process_noise(noise, clip_range=0.5, smoothing_factor=0.1):
    """
    ノイズを処理（tanhでクリッピング + スムージング）
    
    Args:
        noise: ノイズテンソル
        clip_range: クリッピング範囲
        smoothing_factor: スムージング係数
    
    Returns:
        processed_noise: 処理されたノイズ
    """
    scaled_noise = noise / clip_range
    processed_noise = torch.tanh(scaled_noise) * clip_range
    smoothed_noise = processed_noise * (1 - smoothing_factor) + noise * smoothing_factor
    return smoothed_noise


def add_structured_noise(psd_data, clip_range=0.5, smoothing_factor=0.1):
    """
    全体的な構造化ノイズを付与（実験データに近づけるため）
    
    Args:
        psd_data: PSDデータ (L,) または (B, L)
        clip_range: クリッピング範囲
        smoothing_factor: スムージング係数
    
    Returns:
        noisy_psd: ノイズが付与されたPSDデータ
    """
    device = psd_data.device if isinstance(psd_data, torch.Tensor) else torch.device('cpu')
    
    if isinstance(psd_data, np.ndarray):
        psd_data = torch.from_numpy(psd_data).to(device)
    
    # バッチ次元があるかチェック
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
    
    # バッチごとにノイズを生成
    std = std.unsqueeze(0).expand(B, L)  # (B, L)
    noise = torch.normal(mean=0.0, std=std).to(device)
    
    # ノイズを処理
    processed_noise = process_noise(noise, clip_range=clip_range, smoothing_factor=smoothing_factor)
    
    # 乗算的に追加
    noisy_psd = psd_data * (1 + processed_noise)
    
    if squeeze_output:
        noisy_psd = noisy_psd.squeeze(0)
    
    return noisy_psd


# パラメータ設定
NUM_INTERVALS = 30  # SSLと同じ30区間に統一（30クラス分類）
NOISE_LEVEL = 0.3  # ノイズレベル（元の値の30%程度）
NOISE_TYPE = 'frequency_band'  # 'frequency_band', 'localized_spike', 'amplitude_dependent' から選択
ADD_STRUCTURED_NOISE = True  # 構造化ノイズを追加するか
STRUCTURED_NOISE_CLIP_RANGE = 0.5
STRUCTURED_NOISE_SMOOTHING_FACTOR = 0.1

print("=" * 60)
print("データセットの準備")
print("=" * 60)
print(f"区間数: {NUM_INTERVALS}")
print(f"ノイズタイプ: {NOISE_TYPE}")
print(f"ノイズレベル: {NOISE_LEVEL}")
print(f"構造化ノイズ: {ADD_STRUCTURED_NOISE}")
print(f"ノイズ付与範囲: 1~{NUM_INTERVALS}区間（全範囲）")

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

for i in range(num_samples):
    # 元のPSDデータ
    original_psd = x[i, 0, :]
    
    # Tensorに変換
    original_psd_tensor = torch.from_numpy(original_psd) if isinstance(original_psd, np.ndarray) else original_psd
    
    # 1. 構造化ノイズを付与（実験データに近づけるため）
    if ADD_STRUCTURED_NOISE:
        structured_noisy_psd = add_structured_noise(
            original_psd_tensor,
            clip_range=STRUCTURED_NOISE_CLIP_RANGE,
            smoothing_factor=STRUCTURED_NOISE_SMOOTHING_FACTOR
        )
    else:
        structured_noisy_psd = original_psd_tensor
    
    # 2. ランダムに1つの区間を選ぶ（1~30区間すべてから）
    noise_interval = random.randint(0, NUM_INTERVALS - 1)
    
    # 3. 区間ノイズを付与（ノイズ検知の対象）
    noisy_psd_tensor, start_idx, end_idx = add_noise_to_interval(
        structured_noisy_psd,
        noise_interval,
        noise_type=NOISE_TYPE,
        noise_level=NOISE_LEVEL,
        num_intervals=NUM_INTERVALS
    )
    
    noisy_data.append(noisy_psd_tensor)
    labels.append(noise_interval)
    
    if (i + 1) % 1000 == 0:
        print(f"  処理済み: {i+1:,} / {num_samples:,}")

# Tensorに変換
noisy_data = torch.stack(noisy_data)  # (32000, 3000)
labels = torch.tensor(labels, dtype=torch.long)  # (32000,)

print(f"\nデータセットの形状:")
print(f"  入力データ: {noisy_data.shape}")
print(f"  ラベル: {labels.shape}")

# データの前処理: スケーリング + ログ変換 + 正規化
print("\nデータの前処理中（スケーリング + ログ変換 + 正規化）...")
scale_factor = 2.5e24
noisy_data_scaled = noisy_data * scale_factor

# 負の値を1e-30にクリップ
noisy_data_scaled = torch.clamp(noisy_data_scaled, min=1e-30)

# ログ変換
noisy_data_log = torch.log(noisy_data_scaled)

# 正規化（ログ変換後のデータで正規化）
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
val_data = noisy_data_norm[val_indices]
val_labels = labels[val_indices]
test_data = noisy_data_norm[test_indices]
test_labels = labels[test_indices]

print(f"訓練データ: {len(train_indices):,}サンプル")
print(f"検証データ: {len(val_indices):,}サンプル")
print(f"テストデータ: {len(test_indices):,}サンプル")

# ラベルの分布を確認
print("\nラベルの分布（訓練データ）:")
label_counts = torch.bincount(train_labels)
for i in range(NUM_INTERVALS):
    print(f"  区間 {i+1:2d}: {label_counts[i].item():4d}サンプル")

# データセットを保存
print("\nデータセットを保存中...")
dataset = {
    'train': {
        'data': train_data,
        'labels': train_labels
    },
    'val': {
        'data': val_data,
        'labels': val_labels
    },
    'test': {
        'data': test_data,
        'labels': test_labels
    },
    'config': {
        'num_intervals': NUM_INTERVALS,
        'noise_type': NOISE_TYPE,
        'noise_level': NOISE_LEVEL,
        'add_structured_noise': ADD_STRUCTURED_NOISE,
        'structured_noise_clip_range': STRUCTURED_NOISE_CLIP_RANGE,
        'structured_noise_smoothing_factor': STRUCTURED_NOISE_SMOOTHING_FACTOR,
        'scale_factor': scale_factor,
        'normalization_mean': train_mean,
        'normalization_std': train_std
    }
}

with open('baseline_dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)

print("データセットを 'baseline_dataset.pickle' に保存しました")
print("\nデータセット準備完了！")
