"""
30クラス分類用のデータセット準備
32000点すべてにノイズを付与し、ラベルを作成

ノイズタイプ（新しい3種類）:
- 'power_supply': 電源ノイズ（電源周波数50Hzとその高調波、スイッチングノイズ2kHz）
- 'interference': 干渉ノイズ（外部からの電磁干渉、広帯域ノイズ）
- 'clock_leakage': クロックリークノイズ（クロック信号の漏れ、高周波ノイズ）

修正内容:
- 構造化ノイズを追加（実験データに近づけるため）
- 正規化を追加（ログ変換後）
- 新しい3種類のノイズに対応
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
NOISE_TYPE = 'power_supply'  # 'power_supply', 'interference', 'clock_leakage' から選択（USE_RANDOM_NOISE=Falseの場合のみ使用）
# 3種類のノイズをランダムに使用する場合は、以下のように設定:
NOISE_TYPES = ['power_supply', 'interference', 'clock_leakage']
USE_RANDOM_NOISE = True  # Trueにするとランダムに選択（推奨：学習データの多様性向上）
ADD_STRUCTURED_NOISE = True  # 構造化ノイズを追加するか
STRUCTURED_NOISE_CLIP_RANGE = 0.5
STRUCTURED_NOISE_SMOOTHING_FACTOR = 0.1

print("=" * 60)
print("データセットの準備")
print("=" * 60)
print(f"区間数: {NUM_INTERVALS}")
if USE_RANDOM_NOISE:
    print(f"ノイズタイプ: ランダム（power_supply, interference, clock_leakage）")
else:
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
original_data = []  # ノイズ付与前のデータ（復元タスク用）
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
        # 電源ノイズ: 50Hz, 100Hz, 150Hz, 2kHz
        # 2kHzのスイッチングノイズは帯域幅が広く（100Hz）、視覚的に最も目立つ
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
    # 周波数 → ポイントインデックス → 区間インデックス
    freq_to_point = main_freq / (freq_max / num_points)  # main_freq / 5
    point_idx = int(freq_to_point)
    noise_interval = min(point_idx // points_per_interval, NUM_INTERVALS - 1)
    
    # 5. 区間ノイズを付与（ノイズ検知の対象）
    # 注意: interval_idxは無視されるが、ノイズタイプに応じて自動的に周波数領域にノイズが付与される
    noisy_psd_tensor, start_idx, end_idx = add_noise_to_interval(
        structured_noisy_psd,
        noise_interval,  # この値は無視されるが、一応渡す
        noise_type=selected_noise_type,
        noise_level=NOISE_LEVEL,
        num_intervals=NUM_INTERVALS
    )
    
    noisy_data.append(noisy_psd_tensor)
    # 元のデータも保存（構造化ノイズ付与前のデータ）
    original_data.append(structured_noisy_psd)
    labels.append(noise_interval)  # 計算された区間インデックスをラベルとして使用
    
    if (i + 1) % 1000 == 0:
        print(f"  処理済み: {i+1:,} / {num_samples:,}")

# Tensorに変換
noisy_data = torch.stack(noisy_data)  # (32000, 3000)
original_data = torch.stack(original_data)  # (32000, 3000) - ノイズ付与前
labels = torch.tensor(labels, dtype=torch.long)  # (32000,)

print(f"\nデータセットの形状:")
print(f"  ノイズ付きデータ: {noisy_data.shape}")
print(f"  元のデータ（ノイズ付与前）: {original_data.shape}")
print(f"  ラベル: {labels.shape}")

# データの前処理: スケーリング + ログ変換 + 正規化
print("\nデータの前処理中（スケーリング + ログ変換 + 正規化）...")
scale_factor = 2.5e24

# ノイズ付きデータの前処理
noisy_data_scaled = noisy_data * scale_factor
noisy_data_scaled = torch.clamp(noisy_data_scaled, min=1e-30)
noisy_data_log = torch.log(noisy_data_scaled)

# 元のデータの前処理（同じ変換を適用）
original_data_scaled = original_data * scale_factor
original_data_scaled = torch.clamp(original_data_scaled, min=1e-30)
original_data_log = torch.log(original_data_scaled)

# 正規化（ノイズ付きデータの統計量を使用）
train_mean = noisy_data_log.mean().item()
train_std = noisy_data_log.std().item()
if train_std < 1e-6:
    train_std = 1.0

noisy_data_norm = (noisy_data_log - train_mean) / (train_std + 1e-8)
original_data_norm = (original_data_log - train_mean) / (train_std + 1e-8)  # 同じ正規化を適用

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
train_original = original_data_norm[train_indices]  # ノイズ付与前
train_labels = labels[train_indices]
val_data = noisy_data_norm[val_indices]
val_original = original_data_norm[val_indices]  # ノイズ付与前
val_labels = labels[val_indices]
test_data = noisy_data_norm[test_indices]
test_original = original_data_norm[test_indices]  # ノイズ付与前
test_labels = labels[test_indices]

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
        'noisy_data': train_data,  # ノイズ付きデータ
        'original_data': train_original,  # ノイズ付与前のデータ（復元タスク用）
        'labels': train_labels
    },
    'val': {
        'noisy_data': val_data,
        'original_data': val_original,
        'labels': val_labels
    },
    'test': {
        'noisy_data': test_data,
        'original_data': test_original,
        'labels': test_labels
    },
    'config': {
        'num_intervals': NUM_INTERVALS,
        'noise_type': NOISE_TYPE if not USE_RANDOM_NOISE else 'random',
        'noise_types': NOISE_TYPES if USE_RANDOM_NOISE else None,
        'use_random_noise': USE_RANDOM_NOISE,
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
