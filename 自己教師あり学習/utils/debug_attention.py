"""
学習時のアテンションウェイトを確認するデバッグスクリプト
既存の学習済みモデルを使って、アテンションウェイトを確認
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import sys
import os

# プロジェクトパスを追加
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)

# 直接インポート（task/__init__.pyを通さない）
task_path = os.path.join(os.path.dirname(__file__), 'task')
sys.path.insert(0, task_path)
from model import Task4BERT
from dataset import Task4Dataset
from torch.utils.data import DataLoader

def debug_attention_weights(
    model_path='task4_output/best_val_model.pth',
    pickle_path='data_lowF_noise.pickle',
    num_samples=10,
    device='cpu'
):
    """
    学習済みモデルを使ってアテンションウェイトを確認
    
    Args:
        model_path: 学習済みモデルのパス
        pickle_path: データのパス
        num_samples: 確認するサンプル数
        device: デバイス
    """
    print("=" * 60)
    print("学習時のアテンションウェイトを確認")
    print("=" * 60)
    
    # デバイス設定
    if device == 'cpu' and torch.cuda.is_available():
        device = 'cuda'
    device = torch.device(device)
    print(f"Using device: {device}")
    
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
    print(f"Model loaded from: {model_path}")
    
    # アテンションウェイトを確認
    all_attention_stats = []
    all_interval_attention = []
    all_noise_intervals = []
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_samples:
                break
            
            x = batch['input'].to(device)
            m = batch['mask'].to(device)
            noise_interval = batch['noise_interval'].to(device)
            
            # アテンションウェイトを取得
            _, _, attention_weights = model(x, m, return_attention=True)
            
            if attention_weights is not None:
                # アテンションウェイトの統計情報
                attn_stats = {
                    'shape': attention_weights.shape,
                    'min': attention_weights.min().item(),
                    'max': attention_weights.max().item(),
                    'mean': attention_weights.mean().item(),
                    'std': attention_weights.std().item(),
                }
                
                # CLSトークンから系列へのアテンション
                cls_to_seq = attention_weights[0, :, 0, 1:].mean().item()
                attn_stats['cls_to_seq'] = cls_to_seq
                
                # 区間別アテンションを取得
                cls_attention = model.get_cls_attention_to_intervals(
                    attention_weights, num_intervals=30
                )
                
                if cls_attention is not None:
                    cls_attention_np = cls_attention[0].cpu().numpy()
                    noise_idx = noise_interval[0].item()
                    
                    # ノイズ区間と正常区間のアテンション
                    noise_attn = cls_attention_np[noise_idx]
                    normal_indices = [i for i in range(30) if i != noise_idx]
                    normal_attn = cls_attention_np[normal_indices].mean()
                    
                    attn_stats['noise_interval'] = noise_idx
                    attn_stats['noise_attention'] = noise_attn
                    attn_stats['normal_attention'] = normal_attn
                    attn_stats['attention_diff'] = normal_attn - noise_attn
                    
                    all_interval_attention.append(cls_attention_np)
                    all_noise_intervals.append(noise_idx)
                
                all_attention_stats.append(attn_stats)
                
                # 最初のサンプルの詳細情報を表示
                if idx == 0:
                    print(f"\nサンプル {idx+1} の詳細情報:")
                    print(f"  アテンションウェイトの形状: {attn_stats['shape']}")
                    print(f"  最小値: {attn_stats['min']:.6f}")
                    print(f"  最大値: {attn_stats['max']:.6f}")
                    print(f"  平均値: {attn_stats['mean']:.6f}")
                    print(f"  標準偏差: {attn_stats['std']:.6f}")
                    print(f"  CLSトークンから系列へのアテンション: {attn_stats['cls_to_seq']:.6f}")
                    if 'noise_interval' in attn_stats:
                        print(f"  ノイズ区間: {attn_stats['noise_interval']}")
                        print(f"  ノイズ区間のアテンション: {attn_stats['noise_attention']:.6f}")
                        print(f"  正常区間のアテンション: {attn_stats['normal_attention']:.6f}")
                        print(f"  アテンションウェイトの差: {attn_stats['attention_diff']:.6f}")
    
    # 統計情報を集計
    if len(all_attention_stats) > 0:
        print(f"\n全{num_samples}サンプルの統計情報:")
        
        cls_to_seq_list = [s['cls_to_seq'] for s in all_attention_stats]
        print(f"  CLSトークンから系列へのアテンション:")
        print(f"    平均: {np.mean(cls_to_seq_list):.6f}")
        print(f"    最小: {np.min(cls_to_seq_list):.6f}")
        print(f"    最大: {np.max(cls_to_seq_list):.6f}")
        
        if len(all_interval_attention) > 0:
            noise_attn_list = []
            normal_attn_list = []
            diff_list = []
            
            for i, (interval_attn, noise_idx) in enumerate(zip(all_interval_attention, all_noise_intervals)):
                noise_attn = interval_attn[noise_idx]
                normal_indices = [j for j in range(30) if j != noise_idx]
                normal_attn = interval_attn[normal_indices].mean()
                
                noise_attn_list.append(noise_attn)
                normal_attn_list.append(normal_attn)
                diff_list.append(normal_attn - noise_attn)
            
            print(f"\n  区間別アテンションウェイト:")
            print(f"    ノイズ区間の平均: {np.mean(noise_attn_list):.6f} (最小: {np.min(noise_attn_list):.6f}, 最大: {np.max(noise_attn_list):.6f})")
            print(f"    正常区間の平均: {np.mean(normal_attn_list):.6f} (最小: {np.min(normal_attn_list):.6f}, 最大: {np.max(normal_attn_list):.6f})")
            print(f"    アテンションウェイトの差: {np.mean(diff_list):.6f} (最小: {np.min(diff_list):.6f}, 最大: {np.max(diff_list):.6f})")
            
            # 問題点の確認
            print(f"\n問題点の確認:")
            if np.mean(cls_to_seq_list) < 1e-6:
                print(f"  ⚠️ CLSトークンから系列へのアテンションがほぼ0（平均: {np.mean(cls_to_seq_list):.6f}）")
            else:
                print(f"  ✓ CLSトークンから系列へのアテンションは正常（平均: {np.mean(cls_to_seq_list):.6f}）")
            
            if np.std(noise_attn_list) < 1e-6:
                print(f"  ⚠️ ノイズ区間のアテンションが全て同じ（標準偏差: {np.std(noise_attn_list):.6f}）")
            else:
                print(f"  ✓ ノイズ区間のアテンションにばらつきがある（標準偏差: {np.std(noise_attn_list):.6f}）")
            
            if np.std(normal_attn_list) < 1e-6:
                print(f"  ⚠️ 正常区間のアテンションが全て同じ（標準偏差: {np.std(normal_attn_list):.6f}）")
            else:
                print(f"  ✓ 正常区間のアテンションにばらつきがある（標準偏差: {np.std(normal_attn_list):.6f}）")
            
            if np.mean(diff_list) < 1e-4:
                print(f"  ⚠️ アテンションウェイトの差が小さい（平均: {np.mean(diff_list):.6f}）")
            else:
                print(f"  ✓ アテンションウェイトの差がある（平均: {np.mean(diff_list):.6f}）")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='task4_output/best_val_model.pth')
    parser.add_argument('--pickle_path', type=str, default='data_lowF_noise.pickle')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    debug_attention_weights(
        model_path=args.model_path,
        pickle_path=args.pickle_path,
        num_samples=args.num_samples,
        device=args.device
    )

