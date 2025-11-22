"""
復元データの可視化スクリプト
元のデータ、ノイズ付きデータ、復元データを比較
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# プロジェクトパスを追加
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

task_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'task')
sys.path.insert(0, task_path)
from model import Task4BERT
from dataset import Task4Dataset
from torch.utils.data import DataLoader

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def visualize_reconstruction(
    model_path='task4_output/best_val_model.pth',
    pickle_path='data_lowF_noise.pickle',
    num_samples=1,
    device='cpu',
    output_path='reconstruction.png'
):
    """
    復元データを可視化
    
    Args:
        model_path: 学習済みモデルのパス
        pickle_path: データのパス
        num_samples: 可視化するサンプル数
        device: デバイス
        output_path: 出力画像のパス
    """
    # デバイス設定
    if device == 'cpu' and torch.cuda.is_available():
        device = 'cuda'
    device = torch.device(device)
    
    # データセット読み込み
    dataset = Task4Dataset(
        pickle_path=pickle_path,
        num_intervals=30,
        noise_type='power_supply',
        use_random_noise=True,
        noise_level=0.3,
        add_structured_noise_flag=True,
    )
    
    # データローダー作成
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # モデル読み込み
    seq_len = dataset.seq_len
    model = Task4BERT(
        seq_len=seq_len,
        d_model=64,
        n_heads=2,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    points_per_interval = seq_len // 30  # 100ポイント
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_samples:
                break
            
            x = batch['input'].to(device)
            m = batch['mask'].to(device)
            noise_interval = batch['noise_interval'].to(device)
            original_data = batch.get('original', None)
            
            if original_data is None:
                print(f"サンプル {idx+1}: 元のデータがありません（固定データセット）")
                continue
            
            original_data = original_data.to(device)
            
            # モデルの出力（5つの値を返す）
            _, _, attention_weights, reconstructed_interval, reconstructed_intervals_info = model(
                x, m, return_attention=True, num_intervals=30
            )
            
            if reconstructed_interval is None:
                print(f"サンプル {idx+1}: 復元データがありません")
                continue
            
            # アテンションウェイトからノイズ区間を予測
            if attention_weights is not None:
                B = attention_weights.shape[0]
                L = x.shape[1]
                cls_attention_full = attention_weights[:, :, 0, 1:]  # (B, n_heads, L)
                cls_attention_mean = cls_attention_full.mean(dim=1)  # (B, L)
                
                cls_attention_intervals = []
                for i in range(30):
                    start_idx = i * points_per_interval
                    end_idx = min(start_idx + points_per_interval, L)
                    interval_attn = cls_attention_mean[:, start_idx:end_idx].mean(dim=1)
                    cls_attention_intervals.append(interval_attn)
                
                cls_attention = torch.stack(cls_attention_intervals, dim=1)  # (B, 30)
                predicted_noise_interval = cls_attention.argmin(dim=1)[0].item()
            else:
                predicted_noise_interval = noise_interval[0].item()
            
            # 復元範囲の情報を取得
            start_interval = reconstructed_intervals_info[0, 0].item()
            end_interval = reconstructed_intervals_info[0, 1].item()
            
            # 復元されたデータを取得（パディングを除く）
            num_intervals_reconstructed = end_interval - start_interval + 1
            reconstructed_length = num_intervals_reconstructed * points_per_interval
            pred_reconstructed = reconstructed_interval[0, :reconstructed_length].cpu().numpy()
            
            # 正解データも同じ範囲を取得
            true_start_idx = start_interval * points_per_interval
            true_end_idx = min((end_interval + 1) * points_per_interval, original_data.size(1))
            true_original = original_data[0, true_start_idx:true_end_idx].cpu().numpy()
            
            # ノイズ付きデータも同じ範囲を取得
            noisy_data = x[0, true_start_idx:true_end_idx].cpu().numpy()
            
            # 可視化
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            # ポイント番号（0から500まで）
            points = np.arange(len(true_original))
            
            ax.plot(points, true_original, label='元のデータ', color='green', linestyle='--', linewidth=2, alpha=0.8)
            ax.plot(points, noisy_data, label='ノイズ付きデータ', color='red', linewidth=2, alpha=0.8)
            ax.plot(points, pred_reconstructed, label='復元データ', color='blue', linewidth=2, alpha=0.8)
            
            # 予測した区間を背景に表示
            pred_start_idx = (predicted_noise_interval - start_interval) * points_per_interval
            pred_end_idx = pred_start_idx + points_per_interval
            if pred_start_idx >= 0 and pred_end_idx <= len(points):
                ax.axvspan(pred_start_idx, pred_end_idx, alpha=0.2, color='yellow', label='予測した区間')
            
            ax.set_xlabel('ポイント番号', fontsize=12, fontweight='bold')
            ax.set_ylabel('値', fontsize=12, fontweight='bold')
            ax.set_title('予測した区間を真ん中として横2つずつの区間も復元', fontsize=14, fontweight='bold', 
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2))
            ax.legend(fontsize=11, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # 横軸の範囲を0から500に設定
            ax.set_xlim(0, 500)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"復元データの可視化を '{output_path}' に保存しました")
            plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='task4_output/best_val_model.pth')
    parser.add_argument('--pickle_path', type=str, default='data_lowF_noise.pickle')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_path', type=str, default='reconstruction.png')
    
    args = parser.parse_args()
    
    visualize_reconstruction(
        model_path=args.model_path,
        pickle_path=args.pickle_path,
        num_samples=args.num_samples,
        device=args.device,
        output_path=args.output_path
    )

