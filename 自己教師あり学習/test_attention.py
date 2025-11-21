"""
アテンションウェイトが正しく取得できるかテストするスクリプト
学習前に実行して、アテンションウェイトの取得を確認
"""

import torch
import sys
import os

# パス設定（Colab対応、dataset.pyと同じ方法）
try:
    # ローカル実行時
    current_file = __file__
    current_dir = os.path.dirname(os.path.abspath(current_file))
except NameError:
    # Colab実行時
    current_dir = os.getcwd()
    # Colabの場合、作業ディレクトリが自己教師あり学習フォルダになっている想定
    if not os.path.exists(os.path.join(current_dir, 'task')):
        # プロジェクトルートから実行されている場合
        current_dir = os.path.join(current_dir, '自己教師あり学習')
    current_file = os.path.join(current_dir, 'test_attention.py')

# プロジェクトルートを取得（dataset.pyと同じ方法）
# dataset.pyは task/dataset.py にあるので、3階層上がる
# test_attention.pyは 自己教師あり学習/test_attention.py にあるので、1階層上がる
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# ノイズモジュールのパスを追加（dataset.pyと同じ方法）
# 重要: dataset.pyがインポートされる前に、ノイズモジュールのパスを設定する必要がある
noise_module_path = os.path.join(project_root, 'ノイズの付与(共通)')
sys.path.insert(0, noise_module_path)

# 自己教師あり学習フォルダをパスに追加
sys.path.insert(0, current_dir)

# taskパッケージをインポート（これによりtaskパッケージが認識される）
import task

# Task4Datasetをインポート（dataset.py内でノイズモジュールがインポートされる）
# dataset.pyがインポートされる時点で、ノイズモジュールのパスは既に設定されている
from task.dataset import Task4Dataset
from task.model import Task4BERT
from torch.utils.data import DataLoader

def test_attention_weights(
    pickle_path='data_lowF_noise.pickle',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    num_samples=3
):
    """
    アテンションウェイトが正しく取得できるかテスト
    
    Args:
        pickle_path: データのパス
        device: デバイス
        num_samples: テストするサンプル数
    """
    print("=" * 60)
    print("アテンションウェイト取得テスト")
    print("=" * 60)
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
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # モデル作成（学習前の初期状態）
    seq_len = dataset.seq_len
    model = Task4BERT(
        seq_len=seq_len,
        d_model=64,
        n_heads=2,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)
    
    model.eval()
    print(f"Model created: seq_len={seq_len}, d_model=64, n_heads=2, num_layers=2")
    
    # アテンションウェイトを確認
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
            
            x = batch['input'].to(device)
            m = batch['mask'].to(device)
            noise_intervals = batch['noise_interval'].to(device)
            
            print(f"\n--- バッチ {batch_idx} ---")
            print(f"入力形状: {x.shape}")
            print(f"マスク形状: {m.shape}")
            print(f"ノイズ区間: {noise_intervals.cpu().numpy()}")
            
            # アテンションウェイトを取得
            try:
                pred, cls_out, attention_weights = model(x, m, return_attention=True)
                
                if attention_weights is None:
                    print("  ❌ アテンションウェイトが None です！")
                    continue
                
                print(f"\nアテンションウェイト:")
                print(f"  形状: {attention_weights.shape}")
                print(f"  最小値: {attention_weights.min().item():.6f}")
                print(f"  最大値: {attention_weights.max().item():.6f}")
                print(f"  平均値: {attention_weights.mean().item():.6f}")
                print(f"  標準偏差: {attention_weights.std().item():.6f}")
                
                # CLSトークンから系列へのアテンションを確認
                B = attention_weights.shape[0]
                L = x.shape[1]
                num_intervals = 30
                points_per_interval = L // num_intervals
                
                # CLSトークンはインデックス0
                cls_attention_full = attention_weights[:, :, 0, 1:]  # (B, n_heads, L)
                cls_attention_mean = cls_attention_full.mean(dim=1)  # (B, L)
                
                print(f"\nCLSトークンから系列へのアテンション:")
                print(f"  cls_attention_mean.shape: {cls_attention_mean.shape}")
                print(f"  最小値: {cls_attention_mean.min().item():.6f}")
                print(f"  最大値: {cls_attention_mean.max().item():.6f}")
                print(f"  平均値: {cls_attention_mean.mean().item():.6f}")
                print(f"  標準偏差: {cls_attention_mean.std().item():.6f}")
                
                # 区間ごとのアテンションを計算
                for i in range(B):
                    noise_idx = noise_intervals[i].item()
                    interval_attn = []
                    for j in range(num_intervals):
                        start_idx = j * points_per_interval
                        end_idx = min(start_idx + points_per_interval, L)
                        attn = cls_attention_mean[i, start_idx:end_idx].mean()
                        interval_attn.append(attn.item())
                    
                    interval_attn = torch.tensor(interval_attn)
                    noise_attn = interval_attn[noise_idx].item()
                    normal_indices = [j for j in range(num_intervals) if j != noise_idx]
                    normal_attn = interval_attn[normal_indices].mean().item()
                    
                    print(f"\nサンプル {i} の区間アテンション:")
                    print(f"  ノイズ区間インデックス: {noise_idx}")
                    print(f"  ノイズ区間のアテンション: {noise_attn:.6f}")
                    print(f"  正常区間のアテンション平均: {normal_attn:.6f}")
                    print(f"  正常区間 - ノイズ区間: {normal_attn - noise_attn:.6f}")
                    print(f"  全区間のアテンション: {interval_attn.numpy()}")
                    
                    # ランキング損失の計算をシミュレート
                    margin = 0.1
                    attn_diff = normal_attn - noise_attn
                    max_term = max(margin, attn_diff)
                    coeff = 2
                    loss_term = coeff * max_term
                    
                    print(f"\nランキング損失の計算:")
                    print(f"  attn_diff (normal - noise): {attn_diff:.6f}")
                    print(f"  max_term (max(margin={margin}, attn_diff)): {max_term:.6f}")
                    print(f"  loss_term (coeff={coeff} * max_term): {loss_term:.6f}")
                    
                    if abs(attn_diff) < 1e-6:
                        print(f"  ⚠️ 警告: アテンションの差がほぼ0です！")
                    if max_term == margin and attn_diff < margin:
                        print(f"  ⚠️ 警告: max_termが常にmarginになっています（attn_diff < margin）")
                
            except Exception as e:
                print(f"  ❌ エラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_path', type=str, default='data_lowF_noise.pickle')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_samples', type=int, default=3)
    
    args = parser.parse_args()
    
    test_attention_weights(
        pickle_path=args.pickle_path,
        device=args.device,
        num_samples=args.num_samples
    )

