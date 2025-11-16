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
    lambda_reg=0.1,
    num_intervals=30,
    margin=0.1,
):
    """
    損失関数を計算（ランキング損失を使用）
    
    Args:
        predictions: (B, L) 予測されたPSDデータ
        targets: (B, L) ノイズが付与されたPSDデータ（マスクなし）
        mask_positions: (B, L) マスク位置
        attention_weights: (B, n_heads, L+1, L+1) アテンションウェイト
        noise_intervals: (B,) ノイズ区間のインデックス
        lambda_reg: 正則化項の重み
        num_intervals: 区間数
        margin: ランキング損失のマージン
    
    Returns:
        total_loss: 総損失
        mask_loss: マスク予測損失
        reg_loss: ランキング損失
    """
    # 1. マスク予測損失
    if mask_positions.any():
        mask_loss = F.mse_loss(predictions[mask_positions], targets[mask_positions])
    else:
        mask_loss = torch.tensor(0.0, device=predictions.device)
    
    # 2. ランキング損失: ノイズ区間のアテンション < 正常区間のアテンション
    reg_loss = torch.tensor(0.0, device=predictions.device)
    
    if attention_weights is not None:
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
        
        # ランキング損失を計算（数式(3)と(4)に基づく）
        ranking_losses = []
        for i in range(B):
            noise_idx = noise_intervals[i].item()
            noise_attn = interval_attention[i, noise_idx]  # ノイズ区間のアテンション a_ij
            
            # 正常区間のアテンションを取得（ノイズ区間以外）
            normal_indices = [j for j in range(num_intervals) if j != noise_idx]
            normal_attn_list = interval_attention[i, normal_indices]
            
            # 数式(4): Y_dot_ij = { -1, if Y_hat_i = Y_hat_j ; +1, if Y_hat_i ≠ Y_hat_j }
            # ノイズ検知の場合:
            # - ノイズ区間: Y_hat = 1
            # - 正常区間: Y_hat = 0
            # 
            # ノイズ区間(i) vs ノイズ区間(j): Y_dot_ij = -1 (同じラベル)
            # ノイズ区間(i) vs 正常区間(k): Y_dot_ik = +1 (異なるラベル)
            
            for normal_attn in normal_attn_list:
                # 数式(3): L_rank = (1 - Y_dot_ij * Y_dot_ik) * max(m, Y_dot_ij * a_ij + Y_dot_ik * a_ik)
                # 
                # ノイズ区間(i) vs ノイズ区間(j) vs 正常区間(k)の場合:
                # Y_dot_ij = -1 (ノイズ区間同士は同じラベル)
                # Y_dot_ik = +1 (ノイズ区間と正常区間は異なるラベル)
                # 
                # Y_dot_ij * Y_dot_ik = (-1) * (+1) = -1
                # (1 - Y_dot_ij * Y_dot_ik) = (1 - (-1)) = 2
                # 
                # Y_dot_ij * a_ij + Y_dot_ik * a_ik = (-1) * noise_attn + (+1) * normal_attn
                # = normal_attn - noise_attn
                # 
                # ノイズ区間のアテンション < 正常区間のアテンション になるように学習
                # → normal_attn - noise_attn > 0 になるように
                # → max(margin, normal_attn - noise_attn) を損失とする
                
                # 係数項: (1 - Y_dot_ij * Y_dot_ik)
                Y_dot_ij = -1  # ノイズ区間同士（同じラベル）
                Y_dot_ik = +1  # ノイズ区間と正常区間（異なるラベル）
                coeff = 1 - Y_dot_ij * Y_dot_ik  # = 1 - (-1) * (+1) = 2
                
                # アテンションの線形結合: Y_dot_ij * a_ij + Y_dot_ik * a_ik
                attn_linear = Y_dot_ij * noise_attn + Y_dot_ik * normal_attn
                # = (-1) * noise_attn + (+1) * normal_attn
                # = normal_attn - noise_attn
                
                # max(margin, Y_dot_ij * a_ij + Y_dot_ik * a_ik)
                max_term = torch.max(torch.tensor(margin, device=attn_linear.device), attn_linear)
                
                # 数式(3): (1 - Y_dot_ij * Y_dot_ik) * max(m, Y_dot_ij * a_ij + Y_dot_ik * a_ik)
                loss_term = coeff * max_term  # = 2 * max(margin, normal_attn - noise_attn)
                ranking_losses.append(loss_term)
        
        if len(ranking_losses) > 0:
            reg_loss = lambda_reg * torch.stack(ranking_losses).mean()
    
    # 3. 総損失
    total_loss = mask_loss + reg_loss
    
    return total_loss, mask_loss, reg_loss


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
    num_intervals=30,
    noise_type='frequency_band',
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
        noise_type: ノイズタイプ（'frequency_band', 'localized_spike', 'amplitude_dependent'）
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
        d_model=64,
        n_heads=2,
        num_layers=2,
        dim_feedforward=128,
    ).to(device)
    
    print("Model created: d_model=64, n_heads=2, num_layers=2")
    
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler(enabled=(device == "cuda"))
    
    # 再開用の初期値
    start_epoch = 0
    train_losses = []
    train_mask_losses = []
    train_reg_losses = []
    val_losses = []
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
        val_losses = checkpoint["val_losses"]
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
        
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            m = batch["mask"].to(device)
            noise_intervals = batch["noise_interval"].to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=(device == "cuda")):
                pred, cls_out, attention_weights = model(x, m, return_attention=True)
                
                total_loss, mask_loss, reg_loss = compute_loss(
                    pred, y, m, attention_weights, noise_intervals,
                    lambda_reg=lambda_reg, num_intervals=num_intervals, margin=margin
                )
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_train_loss += total_loss.item() * x.size(0)
            running_mask_loss += mask_loss.item() * x.size(0)
            running_reg_loss += reg_loss.item() * x.size(0)
        
        train_loss = running_train_loss / len(train_loader.dataset)
        train_mask_loss = running_mask_loss / len(train_loader.dataset)
        train_reg_loss = running_reg_loss / len(train_loader.dataset)
        
        train_losses.append(train_loss)
        train_mask_losses.append(train_mask_loss)
        train_reg_losses.append(train_reg_loss)
        
        # Validation
        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["input"].to(device)
                    y = batch["target"].to(device)
                    m = batch["mask"].to(device)
                    noise_intervals = batch["noise_interval"].to(device)
                    
                    with autocast(enabled=(device == "cuda")):
                        pred, cls_out, attention_weights = model(x, m, return_attention=True)
                        total_loss, _, _ = compute_loss(
                            pred, y, m, attention_weights, noise_intervals,
                            lambda_reg=lambda_reg, num_intervals=num_intervals, margin=margin
                        )
                    
                    running_val_loss += total_loss.item() * x.size(0)
            
            val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
        else:
            val_loss = None
        
        scheduler.step()
        
        # ログ出力
        if val_loss is not None:
            print(
                f"Epoch {epoch+1}/{num_epochs} "
                f"Train Loss: {train_loss:.6f} "
                f"(Mask: {train_mask_loss:.6f}, Reg: {train_reg_loss:.6f}) "
                f"Val Loss: {val_loss:.6f} "
                f"LR: {scheduler.get_last_lr()[0]:.6e}"
            )
        else:
            print(
                f"Epoch {epoch+1}/{num_epochs} "
                f"Train Loss: {train_loss:.6f} "
                f"(Mask: {train_mask_loss:.6f}, Reg: {train_reg_loss:.6f}) "
                f"(no validation) "
                f"LR: {scheduler.get_last_lr()[0]:.6e}"
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
                "val_losses": val_losses,
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
    
    # Loss履歴をpickle保存
    with open(os.path.join(out_dir, "train_losses.pkl"), "wb") as f:
        pickle.dump(train_losses, f)
    with open(os.path.join(out_dir, "val_losses.pkl"), "wb") as f:
        pickle.dump(val_losses, f)
    
    return model, train_losses, val_losses, best_val_loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model, train_losses, val_losses, best_val_loss = train_task4(
        pickle_path="data_lowF_noise.pickle",
        batch_size=16,  # 8 → 16に増やして速度向上
        num_epochs=10,  # 一旦10エポックに設定
        lr=1e-4,
        val_ratio=0.2,
        device=device,
        out_dir="task4_output",
        resume=True,
        lambda_reg=0.1,
        num_intervals=30,
        noise_type='frequency_band',
        noise_level=0.3,
        margin=0.1,
    )

