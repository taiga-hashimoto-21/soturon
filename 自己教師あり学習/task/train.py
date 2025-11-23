"""
タスク4の学習スクリプト
マスク予測 + 正則化項
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# インポートパス（Colabとローカルで異なる）
try:
    # ローカル実行時（プロジェクトルートから）
    from ssl.task.dataset import Task4Dataset
    from ssl.task.model import Task4BERT
except ImportError:
    # Colab実行時（作業ディレクトリがSSLフォルダ）
    from task.dataset import Task4Dataset
    from task.model import Task4BERT


def compute_loss(
    predictions,
    targets,
    mask_positions,
    attention_weights,
    noise_intervals,
    noise_strength=None,  # (B, num_intervals) 各區間のノイズ強度（正規化済み）
    reconstructed_interval=None,
    reconstructed_intervals_info=None,  # (B, 3) 復元範囲の情報 [start_interval, end_interval, predicted_interval]
    original_data=None,
    lambda_reg=0.1,
    lambda_recon=1.0,
    lambda_mask=1.0,  # マスク予測損失の重み（予測精度を上げるために重要）
    lambda_noise_interval=10.0,  # ノイズ区間予測損失の重み（ノイズ区間を特定するために重要）
    num_intervals=30,
    points_per_interval=100,
    margin=0.1,
    debug=False,
    batch_idx=0,
):
    """
    損失関数を計算（マスク予測損失 + ノイズ強度の逆数損失 + 復元損失）
    
    Args:
        predictions: (B, L) 予測されたPSDデータ（マスク予測タスク用）
        targets: (B, L) ノイズが付与されたPSDデータ（マスクなし）
        mask_positions: (B, L) マスク位置
        attention_weights: (B, n_heads, L+1, L+1) アテンションウェイト
        noise_intervals: (B,) ノイズ区間のインデックス（真の値）
        noise_strength: (B, num_intervals) 各區間のノイズ強度（正規化済み、合計=1）
        reconstructed_interval: (B, 100) 復元された区間（Noneの場合は計算しない）
        original_data: (B, L) 元のデータ（ノイズ付与前、Noneの場合は計算しない）
        lambda_reg: 正則化項の重み
        lambda_recon: 復元損失の重み
        lambda_mask: マスク予測損失の重み（予測精度を上げるために重要）
        num_intervals: 区間数
        points_per_interval: 1区間あたりのポイント数
        margin: マージン（noise_strengthがNoneの場合のフォールバック用）
        debug: デバッグ情報を出力するか
        batch_idx: バッチインデックス（デバッグ用）
    
    Returns:
        total_loss: 総損失
        mask_loss: マスク予測損失
        reg_loss: 正則化損失（ノイズ強度の逆数損失）
        recon_loss: 復元損失（Noneの場合は計算されない）
    """
    # 1. マスク予測損失
    if mask_positions.any():
        mask_loss = F.mse_loss(predictions[mask_positions], targets[mask_positions])
    else:
        mask_loss = torch.tensor(0.0, device=predictions.device)
    
    # 2. 正則化損失: アテンションウェイトがノイズ強度の逆数になるように学習
    reg_loss = torch.tensor(0.0, device=predictions.device)
    noise_interval_prediction_loss = None  # ノイズ区間予測損失（初期化）
    recon_accuracy = 0.0  # 復元精度（初期化）
    
    if attention_weights is not None:
        if debug and batch_idx == 0:
            print(f"\n=== デバッグ情報 (バッチ {batch_idx}) ===")
            print(f"attention_weights.shape: {attention_weights.shape}")
            print(f"attention_weights.min(): {attention_weights.min().item():.6f}")
            print(f"attention_weights.max(): {attention_weights.max().item():.6f}")
            print(f"attention_weights.mean(): {attention_weights.mean().item():.6f}")
            print(f"attention_weights.std(): {attention_weights.std().item():.6f}")
            print(f"noise_intervals: {noise_intervals.cpu().numpy()}")
        # CLSトークンから各区間へのアテンションを取得
        B = attention_weights.shape[0]
        L = predictions.shape[1]
        points_per_interval = L // num_intervals
        
        # CLSトークンはインデックス0
        cls_attention_full = attention_weights[:, :, 0, 1:]  # (B, n_heads, L)
        cls_attention_mean = cls_attention_full.mean(dim=1)  # (B, L)
        
        # 区間ごとのアテンションを計算
        interval_attention_list = []
        for i in range(B):
            interval_attn = []
            for j in range(num_intervals):
                start_idx = j * points_per_interval
                end_idx = min(start_idx + points_per_interval, L)
                attn = cls_attention_mean[i, start_idx:end_idx].mean()
                interval_attn.append(attn)
            interval_attention_list.append(torch.stack(interval_attn))
        
        interval_attention = torch.stack(interval_attention_list)  # (B, num_intervals)
        
        if noise_strength is not None:
            # ノイズ強度の逆数を使った損失（野中さんの説明通り）
            # 目標: attention_weight[i] ≈ 1 / normalized_noise_strength[i]
            # ノイズ強度が0の場合は逆数が無限大になるので、小さな値（epsilon）を加える
            epsilon = 1e-6
            noise_strength_safe = noise_strength + epsilon  # (B, num_intervals)
            target_attention = 1.0 / noise_strength_safe  # (B, num_intervals)
            
            # アテンションウェイトを正規化（合計=1になるように）
            # アテンションウェイトは既に合計=1になっているはずだが、念のため正規化
            interval_attention_normalized = F.normalize(interval_attention, p=1, dim=1)  # (B, num_intervals)
            
            # 目標値も正規化（合計=1になるように）
            target_attention_normalized = F.normalize(target_attention, p=1, dim=1)  # (B, num_intervals)
            
            # 重み付きMSE損失で学習（ノイズ強度が高い区間ほど大きな重み）
            # ノイズ強度が高い区間の誤差を重視する
            # 重み = ノイズ強度 + 小さな定数（ノイズ強度が0の区間にも最小限の重みを付ける）
            weight_base = noise_strength + 0.01  # 0.01は最小重み
            weights = weight_base / weight_base.sum(dim=1, keepdim=True)  # 正規化（合計=1）
            
            # 重み付きMSE損失を計算
            diff = interval_attention_normalized - target_attention_normalized  # (B, num_intervals)
            weighted_squared_error = (diff ** 2) * weights  # (B, num_intervals)
            base_loss = weighted_squared_error.mean()
            
            # ノイズ区間の誤差に追加のペナルティ（ノイズ区間が目標より大きい場合）
            # ノイズ区間のアテンションが目標より大きい場合、追加ペナルティを課す
            noise_penalty = torch.tensor(0.0, device=predictions.device)
            for i in range(B):
                noise_idx = noise_intervals[i].item()
                actual_noise_attn = interval_attention_normalized[i, noise_idx]
                target_noise_attn = target_attention_normalized[i, noise_idx]
                
                # ノイズ区間のアテンションが目標より大きい場合、追加ペナルティ
                if actual_noise_attn > target_noise_attn:
                    noise_penalty = noise_penalty + (actual_noise_attn - target_noise_attn) ** 2
            
            noise_penalty = noise_penalty / B if B > 0 else torch.tensor(0.0, device=predictions.device)  # バッチ平均
            
            # 総損失 = ベース損失 + ノイズ区間ペナルティ（100倍の重み）
            reg_loss = lambda_reg * (base_loss + 100.0 * noise_penalty)
            
            # ノイズ区間予測損失: 実際のノイズ区間のアテンションを小さくするように学習
            # 1. 実際のノイズ区間を正解として使用（最重要の改善）
            target_noise_interval = noise_intervals  # (B,) 実際のノイズ区間を直接使用
            
            # 2. ノイズ区間のアテンションを取得
            noise_interval_attention = interval_attention_normalized[range(B), target_noise_interval]  # (B,)
            
            # 3. 目標アテンションは0（ノイズ区間のアテンションを最小化）
            target_noise_interval_attention = torch.zeros_like(noise_interval_attention)  # (B,)
            
            # 4. MSE損失で学習（クロスエントロピー損失の代わりにMSE損失を使用）
            noise_interval_prediction_loss = F.mse_loss(
                noise_interval_attention,
                target_noise_interval_attention
            )
            
            if debug and batch_idx == 0:
                print(f"\n=== ノイズ強度の逆数損失 (バッチ {batch_idx}) ===")
                print(f"interval_attention.shape: {interval_attention.shape}")
                print(f"noise_strength.shape: {noise_strength.shape}")
                print(f"target_attention.shape: {target_attention.shape}")
                print(f"\n最初のサンプル (i=0):")
                print(f"  ノイズ強度: {noise_strength[0].detach().cpu().numpy()}")
                print(f"  目標アテンション (1/ノイズ強度): {target_attention_normalized[0].detach().cpu().numpy()}")
                print(f"  実際のアテンション: {interval_attention_normalized[0].detach().cpu().numpy()}")
                print(f"  重み付きMSE損失（ベース）: {base_loss.item():.6f}")
                print(f"  ノイズ区間ペナルティ: {noise_penalty.item():.6f}")
                print(f"  総正則化損失: {reg_loss.item():.6f} (ベース: {base_loss.item():.6f}, ペナルティ: {100.0 * noise_penalty.item():.6f})")
                print(f"  lambda_reg: {lambda_reg}")
                print(f"  lambda_mask: {lambda_mask}")
                print(f"  マスク予測損失（重み付き前）: {mask_loss.item():.6f}")
                print(f"  マスク予測損失（重み付き後）: {(lambda_mask * mask_loss).item():.6f}")
                print(f"\n=== ノイズ区間予測損失 (バッチ {batch_idx}) ===")
                print(f"  実際のノイズ区間（noise_intervals）: {noise_intervals[0].item()}")
                print(f"  目標ノイズ区間（target_noise_interval = noise_intervals）: {target_noise_interval[0].item()}")
                print(f"  予測ノイズ区間（interval_attention_normalized.argmin）: {interval_attention_normalized.argmin(dim=1)[0].item()}")
                print(f"  ノイズ区間の予測アテンション: {noise_interval_attention[0].item():.6f}")
                print(f"  ノイズ区間の目標アテンション: {target_noise_interval_attention[0].item():.6f}")
                print(f"  ノイズ区間予測損失（MSE）: {noise_interval_prediction_loss.item():.6f}")
                print(f"  lambda_noise_interval: {lambda_noise_interval}")
                print(f"  ノイズ区間予測損失（重み付き後）: {(lambda_noise_interval * noise_interval_prediction_loss).item():.6f}")
                print("=" * 60)
        else:
            # フォールバック: ノイズ強度が提供されない場合は旧実装を使用（後方互換性）
            # アテンションウェイトをスケーリング（値が小さいため）
            attention_scale = 100.0
            interval_attention = interval_attention * attention_scale
            
            # 旧実装: ノイズ区間のアテンション < 正常区間のアテンション になるように学習
            fallback_losses = []
            for i in range(B):
                noise_idx = noise_intervals[i].item()
                noise_attn = interval_attention[i, noise_idx]
                normal_indices = [j for j in range(num_intervals) if j != noise_idx]
                normal_attn_list = interval_attention[i, normal_indices]
                
                for normal_attn in normal_attn_list:
                    attn_linear = normal_attn - noise_attn
                    max_term = torch.clamp(margin - attn_linear, min=0.0)
                    loss_term = 2.0 * max_term  # coeff = 2
                    fallback_losses.append(loss_term)
            
            if len(fallback_losses) > 0:
                avg_fallback_loss = torch.stack(fallback_losses).mean()
                reg_loss = lambda_reg * avg_fallback_loss
            else:
                reg_loss = torch.tensor(0.0, device=predictions.device)
            
            # フォールバックケースでもノイズ区間予測損失を計算（実際のノイズ区間を正解として使用）
            interval_attention_normalized_fallback = F.normalize(interval_attention, p=1, dim=1)
            # ノイズ区間のアテンションを取得
            noise_interval_attention_fallback = interval_attention_normalized_fallback[range(B), noise_intervals]  # (B,)
            # 目標アテンションは0（ノイズ区間のアテンションを最小化）
            target_noise_interval_attention_fallback = torch.zeros_like(noise_interval_attention_fallback)  # (B,)
            # MSE損失で学習
            noise_interval_prediction_loss = F.mse_loss(
                noise_interval_attention_fallback,
                target_noise_interval_attention_fallback
            )
    
    # 3. 復元損失（ノイズ検知が正しい場合のみ計算 - アプローチ2）
    # 予測区間を中心に左右2区間ずつ（合計5区間）を復元
    recon_loss = None
    # recon_accuracyは既に86行目で初期化されているが、復元損失の計算ブロック内で再定義される可能性があるため、
    # ここでも確実に初期化する（Pythonのローカル変数の扱いを考慮）
    if reconstructed_interval is not None and original_data is not None and reconstructed_intervals_info is not None:
        B = predictions.shape[0]
        L = predictions.shape[1]
        
        # アテンションウェイトからノイズ区間を予測
        if attention_weights is not None:
            # CLSトークンから各区間へのアテンションを取得
            cls_attention_full = attention_weights[:, :, 0, 1:]  # (B, n_heads, L)
            cls_attention_mean = cls_attention_full.mean(dim=1)  # (B, L)
            
            # 区間ごとのアテンションを計算
            interval_attention_list = []
            for i in range(B):
                interval_attn = []
                for j in range(num_intervals):
                    start_idx = j * points_per_interval
                    end_idx = min(start_idx + points_per_interval, L)
                    attn = cls_attention_mean[i, start_idx:end_idx].mean()
                    interval_attn.append(attn)
                interval_attention_list.append(torch.stack(interval_attn))
            
            interval_attention = torch.stack(interval_attention_list)  # (B, num_intervals)
            
            # 最もアテンションが低い区間をノイズ区間として予測
            predicted_noise_intervals = interval_attention.argmin(dim=1)  # (B,)
            
            # 復元損失を計算（予測が正しい場合のみ）
            reconstruction_losses = []
            reconstruction_accuracies = []  # 復元精度を記録
            data_std = 0.82  # 正規化後の標準偏差（約0.82、eval.pyと同じ値）
            
            for i in range(B):
                pred_interval = predicted_noise_intervals[i].item()
                true_interval = noise_intervals[i].item()
                
                # 予測が正しい場合のみ復元損失を計算（アプローチ2）
                if pred_interval == true_interval:
                    # 復元範囲の情報を取得
                    start_interval = reconstructed_intervals_info[i, 0].item()
                    end_interval = reconstructed_intervals_info[i, 1].item()
                    
                    # 復元されたデータを取得（パディングを除く）
                    num_intervals_reconstructed = end_interval - start_interval + 1
                    reconstructed_length = num_intervals_reconstructed * points_per_interval
                    pred_reconstructed = reconstructed_interval[i, :reconstructed_length]  # (reconstructed_length,)
                    
                    # 正解データも同じ範囲を取得
                    true_start_idx = start_interval * points_per_interval
                    true_end_idx = min((end_interval + 1) * points_per_interval, L)
                    true_original = original_data[i, true_start_idx:true_end_idx]  # (reconstructed_length,)
                    
                    # 長さが一致することを確認
                    min_length = min(pred_reconstructed.shape[0], true_original.shape[0])
                    pred_reconstructed = pred_reconstructed[:min_length]
                    true_original = true_original[:min_length]
                    
                    # MSE損失
                    interval_loss = F.mse_loss(pred_reconstructed, true_original)
                    reconstruction_losses.append(interval_loss)
                    
                    # 復元精度（%）を計算（eval.pyと同じ方法）
                    mse_loss = interval_loss.item()
                    rmse = np.sqrt(mse_loss)
                    relative_rmse = rmse / data_std
                    reconstruction_accuracy = max(0.0, (1.0 - relative_rmse) * 100.0)
                    reconstruction_accuracies.append(reconstruction_accuracy)
            
            if len(reconstruction_losses) > 0:
                recon_loss = lambda_recon * torch.stack(reconstruction_losses).mean()
                recon_accuracy = np.mean(reconstruction_accuracies) if len(reconstruction_accuracies) > 0 else 0.0
            else:
                recon_loss = torch.tensor(0.0, device=predictions.device)
                recon_accuracy = 0.0
            
            if debug and batch_idx == 0:
                print(f"\n復元損失の統計（5区間復元）:")
                print(f"  予測が正しいサンプル数: {len(reconstruction_losses)} / {B}")
                if len(reconstruction_losses) > 0:
                    print(f"  平均復元損失: {torch.stack(reconstruction_losses).mean().item():.6f}")
                    print(f"  lambda_recon: {lambda_recon}")
                    print(f"  recon_loss (lambda_recon * avg): {recon_loss.item():.6f}")
                    if len(reconstruction_losses) > 0:
                        sample_idx = 0
                        start_interval = reconstructed_intervals_info[sample_idx, 0].item()
                        end_interval = reconstructed_intervals_info[sample_idx, 1].item()
                        print(f"  サンプル0の復元範囲: 区間{start_interval}〜{end_interval}（合計{end_interval-start_interval+1}区間）")
        else:
            # attention_weights is Noneの場合、復元損失は計算しない
            # recon_accuracyは既に0.0で初期化されている
            recon_accuracy = 0.0  # 確実に定義する
    else:
        # reconstructed_interval is Noneの場合、復元損失は計算しない
        # recon_accuracyは既に0.0で初期化されている
        recon_accuracy = 0.0  # 確実に定義する
    
    # 4. 総損失（マスク予測損失に重みを付ける）
    total_loss = lambda_mask * mask_loss + reg_loss
    if recon_loss is not None:
        total_loss = total_loss + recon_loss
    if noise_interval_prediction_loss is not None:
        total_loss = total_loss + lambda_noise_interval * noise_interval_prediction_loss
    
    return total_loss, mask_loss, reg_loss, recon_loss, recon_accuracy


def train_task4(
    pickle_path,
    batch_size=8,
    num_epochs=100,
    lr=1e-3,
    val_ratio=0.2,
    device="cuda" if torch.cuda.is_available() else "cpu",
    out_dir="task4_output",
    resume=True,
    lambda_reg=0.1,
    lambda_recon=10.0,  # 復元損失の重み（復元損失が最も重要なので大きく設定）
    lambda_mask=20.0,  # マスク予測損失の重み（予測精度を上げるために重要）
    lambda_noise_interval=10.0,  # ノイズ区間予測損失の重み（ノイズ区間を特定するために重要、30.0 → 10.0に調整）
    d_model=128,  # 埋め込み次元（64 → 128に増加）
    n_heads=4,  # アテンションヘッド数（2 → 4に増加）
    num_layers=3,  # Transformerレイヤー数（2 → 3に増加）
    dim_feedforward=256,  # Feedforward層の次元（128 → 256に増加）
    num_intervals=30,
    noise_type='power_supply',
    use_random_noise=True,
    noise_level=0.3,  # ベースラインと合わせる（prepare_baseline_dataset.pyを確認）
    margin=0.1,
):
    """
    タスク4を学習
    
    Args:
        pickle_path: data_lowF_noise.pickleのパス
        batch_size: バッチサイズ
        num_epochs: エポック数
        lr: 学習率
        val_ratio: 検証データの割合
        device: デバイス
        out_dir: 出力ディレクトリ
        resume: チェックポイントから再開するか
        lambda_reg: 正則化項の重み
        num_intervals: 区間数
        noise_type: ノイズタイプ（'power_supply', 'interference', 'clock_leakage'、use_random_noise=Falseの場合のみ使用）
        use_random_noise: 3種類のノイズをランダムに使用するか（デフォルト: True）
        noise_level: ノイズレベル（デフォルト: 0.3）
        margin: ランキング損失のマージン（デフォルト: 0.1）
    """
    print(f"Using device: {device}")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "checkpoint.pth")
    
    # データセット読み込み
    try:
        full_dataset = Task4Dataset(
            pickle_path=pickle_path,
            num_intervals=num_intervals,
            noise_type=noise_type,
            use_random_noise=use_random_noise,
            noise_level=noise_level,
            add_structured_noise_flag=True,  # 全体的な構造化ノイズを付与（実験データに近づけるため）
        )
        seq_len = full_dataset.seq_len
        print(f"Dataset loaded. Sequence length: {seq_len}, Total samples: {len(full_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None, None
    
    # Train / Val 分割
    n_total = len(full_dataset)
    if n_total < 2:
        train_dataset = full_dataset
        val_dataset = None
        print("Warning: dataset size < 2, skip validation.")
    else:
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
        print(f"Train samples: {n_train}, Val samples: {n_val}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None
    
    # モデル・最適化
    model = Task4BERT(
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    ).to(device)
    
    print(f"Model created: d_model={d_model}, n_heads={n_heads}, num_layers={num_layers}, dim_feedforward={dim_feedforward}")
    
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler(enabled=(device == "cuda"))
    
    # 再開用の初期値
    start_epoch = 0
    train_losses = []
    train_mask_losses = []
    train_reg_losses = []
    train_recon_losses = []
    train_recon_accuracies = []  # 復元精度を記録
    train_noise_interval_losses = []  # ノイズ区間予測損失を記録
    train_noise_interval_accuracies = []  # ノイズ区間予測精度を記録
    val_losses = []
    val_recon_losses = []
    val_recon_accuracies = []  # 復元精度を記録
    val_noise_interval_losses = []  # ノイズ区間予測損失を記録
    val_noise_interval_accuracies = []  # ノイズ区間予測精度を記録
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    
    # チェックポイントから再開
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        train_losses = checkpoint["train_losses"]
        train_mask_losses = checkpoint.get("train_mask_losses", [])
        train_reg_losses = checkpoint.get("train_reg_losses", [])
        train_recon_losses = checkpoint.get("train_recon_losses", [])
        train_noise_interval_losses = checkpoint.get("train_noise_interval_losses", [])
        train_noise_interval_accuracies = checkpoint.get("train_noise_interval_accuracies", [])
        val_losses = checkpoint["val_losses"]
        val_recon_losses = checkpoint.get("val_recon_losses", [])
        val_noise_interval_losses = checkpoint.get("val_noise_interval_losses", [])
        val_noise_interval_accuracies = checkpoint.get("val_noise_interval_accuracies", [])
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_train_loss = checkpoint.get("best_train_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")
    
    # 学習ループ
    for epoch in range(start_epoch, num_epochs):
        # Train
        model.train()
        running_train_loss = 0.0
        running_mask_loss = 0.0
        running_reg_loss = 0.0
        running_recon_loss = 0.0
        running_recon_accuracy = 0.0
        num_recon_samples = 0  # 復元損失が計算されたサンプル数
        running_noise_interval_loss = 0.0  # ノイズ区間予測損失
        running_noise_interval_correct = 0  # ノイズ区間予測が正しいサンプル数
        num_noise_interval_samples = 0  # ノイズ区間予測が計算されたサンプル数
        
        for batch_idx, batch in enumerate(train_loader):
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            m = batch["mask"].to(device)
            noise_intervals = batch["noise_interval"].to(device)
            noise_strength = batch.get("noise_strength", None)  # 各區間のノイズ強度（正規化済み）
            if noise_strength is not None:
                noise_strength = noise_strength.to(device)
            original_data = batch.get("original", None)  # 元のデータ（ノイズ付与前）
            if original_data is not None:
                original_data = original_data.to(device)
            
            # 学習開始時にノイズ強度を確認（最初のバッチのみ）
            if epoch == start_epoch and batch_idx == 0:
                print("\n" + "=" * 60)
                print("【学習開始時のノイズ強度確認】")
                print("=" * 60)
                if noise_strength is not None:
                    noise_strength_np = noise_strength.cpu().numpy()  # (batch_size, num_intervals)
                    print(f"ノイズ強度の形状: {noise_strength_np.shape}")
                    print(f"バッチサイズ: {noise_strength_np.shape[0]}")
                    print(f"区間数: {noise_strength_np.shape[1]}")
                    
                    # 各サンプルのノイズ強度を確認
                    for sample_idx in range(min(3, noise_strength_np.shape[0])):  # 最初の3サンプルを確認
                        sample_noise_strength = noise_strength_np[sample_idx]
                        sample_noise_interval = noise_intervals[sample_idx].item()
                        
                        print(f"\n【サンプル {sample_idx}】")
                        print(f"  ノイズが付与された区間: {sample_noise_interval}")
                        print(f"  ノイズ強度の統計:")
                        print(f"    最小値: {sample_noise_strength.min():.6f}")
                        print(f"    最大値: {sample_noise_strength.max():.6f}")
                        print(f"    平均値: {sample_noise_strength.mean():.6f}")
                        print(f"    標準偏差: {sample_noise_strength.std():.6f}")
                        print(f"    合計: {sample_noise_strength.sum():.6f}")
                        
                        # ノイズ区間と正常区間の比較
                        noise_interval_strength = sample_noise_strength[sample_noise_interval]
                        normal_intervals = [i for i in range(num_intervals) if i != sample_noise_interval]
                        normal_interval_strength = sample_noise_strength[normal_intervals].mean()
                        
                        print(f"  ノイズ区間（区間{sample_noise_interval}）の強度: {noise_interval_strength:.6f}")
                        print(f"  正常区間の平均強度: {normal_interval_strength:.6f}")
                        print(f"  差（ノイズ区間 - 正常区間）: {noise_interval_strength - normal_interval_strength:.6f}")
                        if normal_interval_strength > 0:
                            print(f"  比率（ノイズ区間 / 正常区間）: {noise_interval_strength / normal_interval_strength:.2f}倍")
                        
                        # 問題の確認
                        if sample_noise_strength.std() < 0.0001:
                            print(f"  ⚠️ 警告: ノイズ強度の標準偏差が非常に小さいです（{sample_noise_strength.std():.6f}）")
                            print("     全ての区間で同じ値になっている可能性があります。")
                            print("     学習を中断することを推奨します。")
                        elif noise_interval_strength <= normal_interval_strength:
                            print(f"  ⚠️ 警告: ノイズ区間の強度が正常区間以下です。")
                            print("     ノイズ強度の計算に問題がある可能性があります。")
                            print("     学習を中断することを推奨します。")
                        else:
                            print(f"  ✅ ノイズ強度は正しく計算されています。")
                    
                    print("\n" + "=" * 60)
                    print("【学習パラメータ確認】")
                    print(f"  lambda_reg: {lambda_reg}")
                    print(f"  lambda_recon: {lambda_recon}")
                    print(f"  lambda_mask: {lambda_mask}")
                    print(f"  lambda_noise_interval: {lambda_noise_interval}")
                    print(f"  lr: {lr}")
                    print("=" * 60)
                else:
                    print("⚠️ 警告: noise_strengthがNoneです。ノイズ強度の計算が行われていません。")
                    print("=" * 60)
            
            optimizer.zero_grad()
            
            with autocast(enabled=(device == "cuda")):
                # 復元機能を使う場合はreturn_attention=Trueで復元結果も取得
                if original_data is not None:
                    pred, cls_out, attention_weights, reconstructed_interval, reconstructed_intervals_info = model(
                        x, m, return_attention=True, num_intervals=num_intervals
                    )
                else:
                    pred, cls_out, attention_weights, _, _ = model(x, m, return_attention=True)
                    reconstructed_interval = None
                    reconstructed_intervals_info = None
                
                # 最初のバッチのみデバッグ情報を出力
                debug_flag = (epoch == start_epoch and batch_idx == 0)
                
                total_loss, mask_loss, reg_loss, recon_loss, recon_accuracy = compute_loss(
                    pred, y, m, attention_weights, noise_intervals,
                    noise_strength=noise_strength,  # 各區間のノイズ強度
                    reconstructed_interval=reconstructed_interval,
                    reconstructed_intervals_info=reconstructed_intervals_info,  # 復元範囲の情報
                    original_data=original_data,
                    lambda_reg=lambda_reg,
                    lambda_recon=lambda_recon,  # 復元損失の重み（パラメータから取得）
                    lambda_mask=lambda_mask,  # マスク予測損失の重み（パラメータから取得）
                    lambda_noise_interval=lambda_noise_interval,  # ノイズ区間予測損失の重み（パラメータから取得）
                    num_intervals=num_intervals,
                    points_per_interval=seq_len // num_intervals,
                    margin=margin,
                    debug=debug_flag, batch_idx=batch_idx
                )
            
            scaler.scale(total_loss).backward()
            # 勾配クリッピングを追加（勾配の爆発を防ぐ）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_train_loss += total_loss.item() * x.size(0)
            running_mask_loss += mask_loss.item() * x.size(0)
            running_reg_loss += reg_loss.item() * x.size(0)
            if recon_loss is not None and recon_loss.item() > 0:
                running_recon_loss += recon_loss.item() * x.size(0)
                # 復元精度は、復元損失が計算されたサンプル数で平均を取る
                if recon_accuracy > 0:
                    running_recon_accuracy += recon_accuracy * x.size(0)
                    num_recon_samples += x.size(0)
            
            # ノイズ区間予測の精度と損失を計算
            if attention_weights is not None and noise_strength is not None:
                with torch.no_grad():
                    B = attention_weights.shape[0]
                    L = x.shape[1]
                    points_per_interval = L // num_intervals
                    
                    # CLSトークンから各区間へのアテンションを取得
                    cls_attention_full = attention_weights[:, :, 0, 1:]  # (B, n_heads, L)
                    cls_attention_mean = cls_attention_full.mean(dim=1)  # (B, L)
                    
                    # 区間ごとのアテンションを計算
                    interval_attention_list = []
                    for i in range(B):
                        interval_attn = []
                        for j in range(num_intervals):
                            start_idx = j * points_per_interval
                            end_idx = min(start_idx + points_per_interval, L)
                            attn = cls_attention_mean[i, start_idx:end_idx].mean()
                            interval_attn.append(attn)
                        interval_attention_list.append(torch.stack(interval_attn))
                    
                    interval_attention = torch.stack(interval_attention_list)  # (B, num_intervals)
                    interval_attention_normalized = F.normalize(interval_attention, p=1, dim=1)  # (B, num_intervals)
                    
                    # 予測ノイズ区間
                    predicted_noise_intervals = interval_attention_normalized.argmin(dim=1)  # (B,)
                    
                    # 予測精度を計算
                    correct_predictions = (predicted_noise_intervals == noise_intervals).sum().item()
                    running_noise_interval_correct += correct_predictions
                    num_noise_interval_samples += B
                    
                    # ノイズ区間予測損失を計算（MSE損失）
                    noise_interval_attention = interval_attention_normalized[range(B), noise_intervals]  # (B,)
                    target_noise_interval_attention = torch.zeros_like(noise_interval_attention)  # (B,)
                    noise_interval_prediction_loss = F.mse_loss(
                        noise_interval_attention,
                        target_noise_interval_attention,
                        reduction='sum'
                    )
                    running_noise_interval_loss += noise_interval_prediction_loss.item()
        
        train_loss = running_train_loss / len(train_loader.dataset)
        train_mask_loss = running_mask_loss / len(train_loader.dataset)
        train_reg_loss = running_reg_loss / len(train_loader.dataset)
        train_recon_loss = running_recon_loss / len(train_loader.dataset) if running_recon_loss > 0 else 0.0
        train_recon_accuracy = running_recon_accuracy / num_recon_samples if num_recon_samples > 0 else 0.0
        train_noise_interval_loss = running_noise_interval_loss / num_noise_interval_samples if num_noise_interval_samples > 0 else 0.0
        train_noise_interval_accuracy = running_noise_interval_correct / num_noise_interval_samples if num_noise_interval_samples > 0 else 0.0
        
        train_losses.append(train_loss)
        train_mask_losses.append(train_mask_loss)
        train_reg_losses.append(train_reg_loss)
        train_recon_losses.append(train_recon_loss)
        train_recon_accuracies.append(train_recon_accuracy)
        train_noise_interval_losses.append(train_noise_interval_loss)
        train_noise_interval_accuracies.append(train_noise_interval_accuracy)
        
        # Validation
        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0
            running_val_recon_loss = 0.0
            running_val_recon_accuracy = 0.0
            num_val_recon_samples = 0
            running_val_noise_interval_loss = 0.0
            running_val_noise_interval_correct = 0
            num_val_noise_interval_samples = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["input"].to(device)
                    y = batch["target"].to(device)
                    m = batch["mask"].to(device)
                    noise_intervals = batch["noise_interval"].to(device)
                    
                    original_data = batch.get("original", None)
                    if original_data is not None:
                        original_data = original_data.to(device)
                    
                    noise_strength = batch.get("noise_strength", None)  # 各區間のノイズ強度（正規化済み）
                    if noise_strength is not None:
                        noise_strength = noise_strength.to(device)
                    
                    with autocast(enabled=(device == "cuda")):
                        if original_data is not None:
                            pred, cls_out, attention_weights, reconstructed_interval, reconstructed_intervals_info = model(
                                x, m, return_attention=True, num_intervals=num_intervals
                            )
                        else:
                            pred, cls_out, attention_weights, _, _ = model(x, m, return_attention=True)
                            reconstructed_interval = None
                            reconstructed_intervals_info = None
                        
                        total_loss, _, _, recon_loss, recon_accuracy = compute_loss(
                            pred, y, m, attention_weights, noise_intervals,
                            noise_strength=noise_strength,  # 各區間のノイズ強度
                            reconstructed_interval=reconstructed_interval,
                            reconstructed_intervals_info=reconstructed_intervals_info,  # 復元範囲の情報
                            original_data=original_data,
                            lambda_reg=lambda_reg,
                            lambda_recon=lambda_recon,  # 復元損失の重み（パラメータから取得）
                            lambda_mask=lambda_mask,  # マスク予測損失の重み（パラメータから取得）
                            lambda_noise_interval=lambda_noise_interval,  # ノイズ区間予測損失の重み（パラメータから取得）
                            num_intervals=num_intervals,
                            points_per_interval=seq_len // num_intervals,
                            margin=margin
                        )
                    
                    running_val_loss += total_loss.item() * x.size(0)
                    if recon_loss is not None and recon_loss.item() > 0:
                        running_val_recon_loss += recon_loss.item() * x.size(0)
                        if recon_accuracy > 0:
                            running_val_recon_accuracy += recon_accuracy * x.size(0)
                            num_val_recon_samples += x.size(0)
                    
                    # ノイズ区間予測の精度と損失を計算
                    if attention_weights is not None and noise_strength is not None:
                        B = attention_weights.shape[0]
                        L = x.shape[1]
                        points_per_interval = L // num_intervals
                        
                        # CLSトークンから各区間へのアテンションを取得
                        cls_attention_full = attention_weights[:, :, 0, 1:]  # (B, n_heads, L)
                        cls_attention_mean = cls_attention_full.mean(dim=1)  # (B, L)
                        
                        # 区間ごとのアテンションを計算
                        interval_attention_list = []
                        for i in range(B):
                            interval_attn = []
                            for j in range(num_intervals):
                                start_idx = j * points_per_interval
                                end_idx = min(start_idx + points_per_interval, L)
                                attn = cls_attention_mean[i, start_idx:end_idx].mean()
                                interval_attn.append(attn)
                            interval_attention_list.append(torch.stack(interval_attn))
                        
                        interval_attention = torch.stack(interval_attention_list)  # (B, num_intervals)
                        interval_attention_normalized = F.normalize(interval_attention, p=1, dim=1)  # (B, num_intervals)
                        
                        # 予測ノイズ区間
                        predicted_noise_intervals = interval_attention_normalized.argmin(dim=1)  # (B,)
                        
                        # 予測精度を計算
                        correct_predictions = (predicted_noise_intervals == noise_intervals).sum().item()
                        running_val_noise_interval_correct += correct_predictions
                        num_val_noise_interval_samples += B
                        
                        # ノイズ区間予測損失を計算（MSE損失）
                        noise_interval_attention = interval_attention_normalized[range(B), noise_intervals]  # (B,)
                        target_noise_interval_attention = torch.zeros_like(noise_interval_attention)  # (B,)
                        noise_interval_prediction_loss = F.mse_loss(
                            noise_interval_attention,
                            target_noise_interval_attention,
                            reduction='sum'
                        )
                        running_val_noise_interval_loss += noise_interval_prediction_loss.item()
            
            val_loss = running_val_loss / len(val_loader.dataset)
            val_recon_loss = running_val_recon_loss / len(val_loader.dataset) if running_val_recon_loss > 0 else 0.0
            val_recon_accuracy = running_val_recon_accuracy / num_val_recon_samples if num_val_recon_samples > 0 else 0.0
            val_noise_interval_loss = running_val_noise_interval_loss / num_val_noise_interval_samples if num_val_noise_interval_samples > 0 else 0.0
            val_noise_interval_accuracy = running_val_noise_interval_correct / num_val_noise_interval_samples if num_val_noise_interval_samples > 0 else 0.0
            val_losses.append(val_loss)
            val_recon_losses.append(val_recon_loss)
            val_recon_accuracies.append(val_recon_accuracy)
            val_noise_interval_losses.append(val_noise_interval_loss)
            val_noise_interval_accuracies.append(val_noise_interval_accuracy)
        else:
            val_loss = None
            val_recon_loss = None
            val_noise_interval_loss = None
            val_noise_interval_accuracy = None
            val_noise_interval_loss = None
            val_noise_interval_accuracy = None
        
        scheduler.step()
        
        # ログ出力
        # マスク予測損失、ノイズ区間予測精度、復元損失を表示
        mask_pred_str = f"{train_mask_loss:.6f}"
        noise_interval_pred_str = f"{train_noise_interval_accuracy:.4f}" if train_noise_interval_accuracy > 0 else "0.0000"
        recon_loss_str = f"{train_recon_loss:.6f}" if train_recon_loss > 0 else "0.000000"
        
        print(
            f"エポック {epoch+1}/{num_epochs} "
            f"マスク予測:{mask_pred_str},"
            f"ノイズ区間予測:{noise_interval_pred_str},"
            f"復元損失:{recon_loss_str}"
        )
        
        # best model 保存
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best_train_model.pth"))
        
        if (val_loss is not None) and (val_loss < best_val_loss):
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best_val_model.pth"))
            print(f"  >> New best val model saved (Val Loss: {val_loss:.6f})")
        
        # チェックポイント保存
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "train_losses": train_losses,
                "train_mask_losses": train_mask_losses,
                "train_reg_losses": train_reg_losses,
                "train_recon_losses": train_recon_losses,
                "train_noise_interval_losses": train_noise_interval_losses,
                "train_noise_interval_accuracies": train_noise_interval_accuracies,
                "val_losses": val_losses,
                "val_recon_losses": val_recon_losses,
                "val_noise_interval_losses": val_noise_interval_losses,
                "val_noise_interval_accuracies": val_noise_interval_accuracies,
                "best_val_loss": best_val_loss,
                "best_train_loss": best_train_loss,
            },
            checkpoint_path,
        )
    
    print("Training complete.")
    
    # 最終モデル保存
    final_model_path = os.path.join(out_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # 学習終了後にアテンション情報を保存（デバッグ用）
    print("\n保存用にアテンション情報を計算中...")
    model.eval()
    attention_data = {
        "predicted_attention": [],  # 予測アテンション（interval_attention_normalized）
        "target_attention": [],  # 目標アテンション（target_attention_normalized）
        "true_noise_intervals": [],  # 実際のノイズ区間（noise_intervals）
        "predicted_noise_intervals": [],  # 予測ノイズ区間（interval_attention_normalized.argmin）
        "noise_strength": [],  # ノイズ強度（noise_strength）
    }
    
    # 検証データセットからサンプルを取得（最大100サンプル）
    if val_loader is not None:
        with torch.no_grad():
            num_samples_saved = 0
            max_samples = 100
            
            for batch in val_loader:
                if num_samples_saved >= max_samples:
                    break
                
                x = batch["input"].to(device)
                y = batch["target"].to(device)
                m = batch["mask"].to(device)
                noise_intervals = batch["noise_interval"].to(device)
                
                noise_strength = batch.get("noise_strength", None)
                if noise_strength is not None:
                    noise_strength = noise_strength.to(device)
                
                # アテンションウェイトを取得
                _, _, attention_weights, _, _ = model(x, m, return_attention=True)
                
                if attention_weights is not None and noise_strength is not None:
                    B = attention_weights.shape[0]
                    L = x.shape[1]
                    points_per_interval = L // num_intervals
                    
                    # CLSトークンから各区間へのアテンションを取得
                    cls_attention_full = attention_weights[:, :, 0, 1:]  # (B, n_heads, L)
                    cls_attention_mean = cls_attention_full.mean(dim=1)  # (B, L)
                    
                    # 区間ごとのアテンションを計算
                    interval_attention_list = []
                    for i in range(B):
                        interval_attn = []
                        for j in range(num_intervals):
                            start_idx = j * points_per_interval
                            end_idx = min(start_idx + points_per_interval, L)
                            attn = cls_attention_mean[i, start_idx:end_idx].mean()
                            interval_attn.append(attn)
                        interval_attention_list.append(torch.stack(interval_attn))
                    
                    interval_attention = torch.stack(interval_attention_list)  # (B, num_intervals)
                    
                    # ノイズ強度の逆数を使った目標アテンションを計算
                    epsilon = 1e-6
                    noise_strength_safe = noise_strength + epsilon
                    target_attention = 1.0 / noise_strength_safe  # (B, num_intervals)
                    
                    # 正規化
                    interval_attention_normalized = F.normalize(interval_attention, p=1, dim=1)  # (B, num_intervals)
                    target_attention_normalized = F.normalize(target_attention, p=1, dim=1)  # (B, num_intervals)
                    
                    # 予測ノイズ区間
                    predicted_noise_intervals = interval_attention_normalized.argmin(dim=1)  # (B,)
                    
                    # CPUに移動して保存
                    attention_data["predicted_attention"].append(interval_attention_normalized.cpu().numpy())
                    attention_data["target_attention"].append(target_attention_normalized.cpu().numpy())
                    attention_data["true_noise_intervals"].append(noise_intervals.cpu().numpy())
                    attention_data["predicted_noise_intervals"].append(predicted_noise_intervals.cpu().numpy())
                    attention_data["noise_strength"].append(noise_strength.cpu().numpy())
                    
                    num_samples_saved += B
    
    # リストをnumpy配列に変換
    if len(attention_data["predicted_attention"]) > 0:
        attention_data["predicted_attention"] = np.concatenate(attention_data["predicted_attention"], axis=0)
        attention_data["target_attention"] = np.concatenate(attention_data["target_attention"], axis=0)
        attention_data["true_noise_intervals"] = np.concatenate(attention_data["true_noise_intervals"], axis=0)
        attention_data["predicted_noise_intervals"] = np.concatenate(attention_data["predicted_noise_intervals"], axis=0)
        attention_data["noise_strength"] = np.concatenate(attention_data["noise_strength"], axis=0)
        
        # pickleで保存
        attention_save_path = os.path.join(out_dir, "attention_data.pkl")
        with open(attention_save_path, "wb") as f:
            pickle.dump(attention_data, f)
        print(f"アテンション情報を保存しました: {attention_save_path}")
        print(f"  サンプル数: {attention_data['predicted_attention'].shape[0]}")
        print(f"  区間数: {attention_data['predicted_attention'].shape[1]}")
    else:
        print("⚠️ アテンション情報を保存できませんでした（データがありません）")
    
    # Lossカーブ描画
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    if len(val_losses) > 0:
        plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_mask_losses, label="Mask Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Mask Prediction Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_reg_losses, label="Regularization Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Regularization Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_curve_path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Loss curve saved to {loss_curve_path}")
    
    # ノイズ区間予測の精度と損失のグラフを描画
    if len(train_noise_interval_losses) > 0:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        epochs = range(1, len(train_noise_interval_losses) + 1)
        plt.plot(epochs, train_noise_interval_losses, label="Train Noise Interval Loss", color="blue")
        if len(val_noise_interval_losses) > 0:
            plt.plot(epochs, val_noise_interval_losses, label="Val Noise Interval Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Noise Interval Prediction Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_noise_interval_accuracies, label="Train Noise Interval Accuracy", color="blue")
        if len(val_noise_interval_accuracies) > 0:
            plt.plot(epochs, val_noise_interval_accuracies, label="Val Noise Interval Accuracy", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Noise Interval Prediction Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        noise_interval_curve_path = os.path.join(out_dir, "noise_interval_curve.png")
        plt.savefig(noise_interval_curve_path)
        plt.close()
        print(f"Noise interval curve saved to {noise_interval_curve_path}")
    
    # Loss履歴をpickle保存
    with open(os.path.join(out_dir, "train_losses.pkl"), "wb") as f:
        pickle.dump(train_losses, f)
    with open(os.path.join(out_dir, "val_losses.pkl"), "wb") as f:
        pickle.dump(val_losses, f)
    with open(os.path.join(out_dir, "train_recon_losses.pkl"), "wb") as f:
        pickle.dump(train_recon_losses, f)
    with open(os.path.join(out_dir, "val_recon_losses.pkl"), "wb") as f:
        pickle.dump(val_recon_losses, f)
    with open(os.path.join(out_dir, "train_recon_accuracies.pkl"), "wb") as f:
        pickle.dump(train_recon_accuracies, f)
    with open(os.path.join(out_dir, "val_recon_accuracies.pkl"), "wb") as f:
        pickle.dump(val_recon_accuracies, f)
    with open(os.path.join(out_dir, "train_noise_interval_losses.pkl"), "wb") as f:
        pickle.dump(train_noise_interval_losses, f)
    with open(os.path.join(out_dir, "val_noise_interval_losses.pkl"), "wb") as f:
        pickle.dump(val_noise_interval_losses, f)
    with open(os.path.join(out_dir, "train_noise_interval_accuracies.pkl"), "wb") as f:
        pickle.dump(train_noise_interval_accuracies, f)
    with open(os.path.join(out_dir, "val_noise_interval_accuracies.pkl"), "wb") as f:
        pickle.dump(val_noise_interval_accuracies, f)
    
    return model, train_losses, val_losses, best_val_loss, train_recon_losses, val_recon_losses, train_recon_accuracies, val_recon_accuracies


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model, train_losses, val_losses, best_val_loss, train_recon_losses, val_recon_losses = train_task4(
        pickle_path="data_lowF_noise.pickle",
        batch_size=16,  # 8 → 16に増やして速度向上
        num_epochs=10,  # 一旦10エポックに設定
        lr=1e-3,  # 1e-4 → 1e-3に変更（正則化項の学習を促進）
        val_ratio=0.2,
        device=device,
        out_dir="task4_output",
        resume=True,
        lambda_reg=2.0,  # 0.1 → 2.0に変更（正則化項の重みを大きく）
        num_intervals=30,
        noise_type='power_supply',
        use_random_noise=True,
        noise_level=0.3,
        margin=0.01,  # 0.1 → 0.01に変更（学習が進みやすくする）
    )

