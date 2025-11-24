"""
ノイズ検出 + 復元タスク用の損失関数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_noise_detection_reconstruction_loss(
    predicted_mask,
    reconstructed_interval,
    reconstructed_intervals_info,
    true_mask,
    original_psd,
    noisy_psd,
    true_noise_intervals,
    num_intervals=30,
    mask_loss_weight=0.5,
    reconstruction_loss_weight=1.0,
    points_per_interval=100
):
    """
    ノイズ検出 + 復元タスクの損失関数（アプローチ2: マスク予測が正しい場合のみ復元損失を計算）
    
    Args:
        predicted_mask: 予測されたノイズマスク (batch_size, 30) - 各区間のノイズ強度
        reconstructed_interval: 復元された区間 (batch_size, max_length) - 予測区間±2区間（最大500ポイント）
        reconstructed_intervals_info: 復元範囲の情報 (batch_size, 3) - [start_interval, end_interval, predicted_interval]
        true_mask: 真のノイズマスク (batch_size, 30) - 各区間のノイズ有無（0 or 1）
        original_psd: 元のPSDデータ（ノイズ付与前） (batch_size, 3000)
        noisy_psd: ノイズ付きPSDデータ (batch_size, 3000)
        true_noise_intervals: 真のノイズ区間のインデックス (batch_size,) - 区間番号
        num_intervals: 区間数（デフォルト: 30）
        mask_loss_weight: マスク予測損失の重み（デフォルト: 0.5）
        reconstruction_loss_weight: 復元損失の重み（デフォルト: 1.0）
        points_per_interval: 1区間あたりのポイント数（デフォルト: 100）
    
    Returns:
        total_loss: 総損失
        mask_loss: マスク予測損失
        reconstruction_loss: 復元損失（マスク予測が正しい場合のみ）
        loss_dict: 詳細な損失情報の辞書
    """
    batch_size = predicted_mask.size(0)
    device = predicted_mask.device
    
    # 1. マスク予測損失（BCE Loss）
    # true_maskは各区間にノイズがあるかどうか（0 or 1）
    mask_loss = F.binary_cross_entropy(predicted_mask, true_mask.float(), reduction='mean')
    
    # 2. 復元損失（マスク予測が正しい場合のみ計算）- アプローチ2
    reconstruction_losses = []
    relative_losses = []
    smoothness_losses = []
    num_correct_masks = 0  # マスク予測が正しいサンプル数
    
    for i in range(batch_size):
        true_noise_interval = true_noise_intervals[i].item()
        
        # マスク予測が正しいかチェック（最も確信度の高い区間を予測区間とする）
        predicted_interval = predicted_mask[i].argmax().item()
        mask_correct = (predicted_interval == true_noise_interval)
        
        if mask_correct:
            num_correct_masks += 1
            
            # マスク予測が正しい場合のみ復元損失を計算
            # 予測区間±2区間（合計5区間）の復元精度を評価
            # 復元範囲の情報を取得
            start_interval = reconstructed_intervals_info[i, 0].item()
            end_interval = reconstructed_intervals_info[i, 1].item()
            
            # 復元されたデータを取得（パディングを除く）
            num_intervals_reconstructed = end_interval - start_interval + 1
            reconstructed_length = num_intervals_reconstructed * points_per_interval
            pred_reconstructed = reconstructed_interval[i, :reconstructed_length]  # (reconstructed_length,)
            
            # 正解データも同じ範囲を取得
            true_start_idx = start_interval * points_per_interval
            true_end_idx = min((end_interval + 1) * points_per_interval, original_psd.size(1))
            true_original = original_psd[i, true_start_idx:true_end_idx]  # (reconstructed_length,)
            
            # 長さが一致することを確認
            min_length = min(pred_reconstructed.shape[0], true_original.shape[0])
            pred_reconstructed = pred_reconstructed[:min_length]
            true_original = true_original[:min_length]
            
            # (a) MSE損失（基本的な復元精度）
            mse_loss = F.mse_loss(pred_reconstructed, true_original)
            reconstruction_losses.append(mse_loss)
            
            # 復元精度（%）を計算（eval.pyと同じ方法）
            import numpy as np
            data_std = 0.82  # 正規化後の標準偏差（約0.82）
            rmse = np.sqrt(mse_loss.item())
            relative_rmse = rmse / data_std
            reconstruction_accuracy = max(0.0, (1.0 - relative_rmse) * 100.0)
            
            # (b) 相対誤差（大きな値と小さな値の両方で精度を評価）
            relative_error = torch.abs(pred_reconstructed - true_original) / (
                torch.abs(true_original) + 1e-8
            )
            relative_loss = relative_error.mean()
            relative_losses.append(relative_loss)
            
            # (c) 平滑性損失（周辺区間との連続性を保つ）
            # 復元範囲の前後の区間との境界での連続性
            if start_interval > 0:
                prev_end = start_interval * points_per_interval - 1
                if prev_end >= 0:
                    boundary_loss_prev = F.mse_loss(
                        pred_reconstructed[0:1],
                        noisy_psd[i, prev_end:prev_end+1]  # ノイズ付きデータの前の区間
                    )
                else:
                    boundary_loss_prev = torch.tensor(0.0, device=original_psd.device)
            else:
                boundary_loss_prev = torch.tensor(0.0, device=original_psd.device)
            
            if end_interval < num_intervals - 1:
                next_start = (end_interval + 1) * points_per_interval
                if next_start < original_psd.size(1):
                    boundary_loss_next = F.mse_loss(
                        pred_reconstructed[-1:],
                        noisy_psd[i, next_start:next_start+1]  # ノイズ付きデータの次の区間
                    )
                else:
                    boundary_loss_next = torch.tensor(0.0, device=original_psd.device)
            else:
                boundary_loss_next = torch.tensor(0.0, device=original_psd.device)
            
            smoothness_loss = (boundary_loss_prev + boundary_loss_next) / 2
            smoothness_losses.append(smoothness_loss)
    
    # マスク予測が正しいサンプルのみで復元損失を計算
    if num_correct_masks > 0:
        reconstruction_loss_mse = torch.stack(reconstruction_losses).mean()
        reconstruction_loss_relative = torch.stack(relative_losses).mean()
        reconstruction_loss_smoothness = torch.stack(smoothness_losses).mean()
        
        # 総復元損失（MSEを重視、相対誤差と平滑性も考慮）
        reconstruction_loss = (
            1.0 * reconstruction_loss_mse +
            0.3 * reconstruction_loss_relative +
            0.1 * reconstruction_loss_smoothness
        )
        
        # 復元精度を計算（MSE損失から、eval.pyと同じ方法）
        import numpy as np
        data_std = 0.82  # 正規化後の標準偏差（約0.82）
        rmse = np.sqrt(reconstruction_loss_mse.item())
        relative_rmse = rmse / data_std
        reconstruction_accuracy = max(0.0, (1.0 - relative_rmse) * 100.0)
    else:
        # マスク予測が全て間違っている場合、復元損失は0（マスク予測損失のみで学習）
        reconstruction_loss = torch.tensor(0.0, device=predicted_mask.device)
        reconstruction_loss_mse = torch.tensor(0.0, device=predicted_mask.device)
        reconstruction_loss_relative = torch.tensor(0.0, device=predicted_mask.device)
        reconstruction_loss_smoothness = torch.tensor(0.0, device=predicted_mask.device)
        reconstruction_accuracy = 0.0
    
    # 3. 総損失
    total_loss = (
        mask_loss_weight * mask_loss +
        reconstruction_loss_weight * reconstruction_loss
    )
    
    # 詳細な損失情報
    loss_dict = {
        'total_loss': total_loss.item(),
        'mask_loss': mask_loss.item(),
        'reconstruction_loss': reconstruction_loss.item(),
        'reconstruction_loss_mse': reconstruction_loss_mse.item(),
        'reconstruction_loss_relative': reconstruction_loss_relative.item(),
        'reconstruction_loss_smoothness': reconstruction_loss_smoothness.item(),
        'num_correct_masks': num_correct_masks,
        'mask_accuracy': num_correct_masks / batch_size,  # マスク予測の精度
        'reconstruction_accuracy': reconstruction_accuracy if num_correct_masks > 0 else 0.0,  # 復元精度を追加
        'mask_loss_weight': mask_loss_weight,
        'reconstruction_loss_weight': reconstruction_loss_weight,
    }
    
    return total_loss, mask_loss, reconstruction_loss, loss_dict


def create_true_mask_from_intervals(true_noise_intervals, num_intervals=30, batch_size=None):
    """
    真のノイズ区間のインデックスからマスクを作成
    
    Args:
        true_noise_intervals: 真のノイズ区間のインデックス (batch_size,) または (batch_size, num_noise_intervals)
        num_intervals: 区間数（デフォルト: 30）
        batch_size: バッチサイズ（Noneの場合はtrue_noise_intervalsから推論）
    
    Returns:
        true_mask: 真のノイズマスク (batch_size, 30) - 各区間のノイズ有無（0 or 1）
    """
    if true_noise_intervals.dim() == 1:
        # (batch_size,) の場合：1つの区間にノイズ
        batch_size = true_noise_intervals.size(0)
        true_mask = torch.zeros(batch_size, num_intervals, device=true_noise_intervals.device)
        true_mask.scatter_(1, true_noise_intervals.unsqueeze(1), 1.0)
    else:
        # (batch_size, num_noise_intervals) の場合：複数の区間にノイズ
        batch_size = true_noise_intervals.size(0)
        true_mask = torch.zeros(batch_size, num_intervals, device=true_noise_intervals.device)
        for i in range(batch_size):
            noise_intervals = true_noise_intervals[i]
            # -1でパディングされている場合は除外
            valid_intervals = noise_intervals[noise_intervals >= 0]
            true_mask[i, valid_intervals] = 1.0
    
    return true_mask

