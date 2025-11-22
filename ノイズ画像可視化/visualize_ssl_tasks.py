"""
自己教師あり学習モデルのタスク説明用画像生成
- 主タスク: マスク予測（ノイズ以外の区間にマスクを付与）
- 副タスク: ノイズ検知（ノイズ区間の検出）
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
noise_folder = os.path.join(project_root, 'ノイズの付与(共通)')
sys.path.insert(0, project_root)
sys.path.insert(0, noise_folder)

# ノイズモジュールをインポート
import importlib.util
power_supply_path = os.path.join(noise_folder, 'power_supply_noise.py')
spec = importlib.util.spec_from_file_location("power_supply_noise", power_supply_path)
power_supply_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(power_supply_module)
add_power_supply_noise = power_supply_module.add_power_supply_noise

import torch

# 日本語フォントの設定
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

def generate_psd_data(num_points=3000, freq_max=15000.0):
    """ベースとなるPSDデータを生成（滑らかな形状、周波数0で最大値）"""
    frequencies = np.linspace(0, freq_max, num_points)
    
    # シンプルに：周波数0で最大値、周波数が高くなるほど小さくなる
    # 1/f特性のような形状
    psd = 1.0 / (1.0 + frequencies / 1000.0)
    
    # 周波数0で確実に最大値に設定
    psd[0] = psd.max()
    
    # 滑らかにするために移動平均を適用（周波数0は保護）
    window_size = 50
    psd_smooth = np.convolve(psd, np.ones(window_size)/window_size, mode='same')
    
    # 移動平均後も周波数0を最大値に設定
    psd_smooth[0] = psd_smooth.max()
    
    # 正規化（0.9にスケール）
    psd_smooth = psd_smooth / psd_smooth.max() * 0.9
    
    # 最終確認：周波数0が最大値（0.9）であることを保証
    psd_smooth[0] = 0.9
    
    return psd_smooth, frequencies

def add_mask_to_intervals(psd_data, masked_intervals, num_intervals=30):
    """指定された区間にマスクを付与（0で置き換え）"""
    masked_data = psd_data.copy()
    points_per_interval = len(psd_data) // num_intervals
    
    for interval_idx in masked_intervals:
        start_idx = interval_idx * points_per_interval
        end_idx = min(start_idx + points_per_interval, len(psd_data))
        masked_data[start_idx:end_idx] = 0.0
    
    return masked_data

def visualize_ssl_tasks():
    """自己教師あり学習モデルのタスク説明画像を生成"""
    
    # パラメータ設定
    num_points = 3000
    num_intervals = 30
    points_per_interval = num_points // num_intervals
    freq_max = 15000.0
    
    # ベースPSDデータを生成（周波数0で最大値、周波数が高くなるほど小さくなる）
    original_psd, frequencies = generate_psd_data(num_points, freq_max)
    
    # ノイズ区間を決定（2kHz付近、区間4）
    freq_per_interval = freq_max / num_intervals  # 500Hz per interval
    noise_freq = 2000.0  # 2kHz
    noise_interval = int(noise_freq / freq_per_interval)  # 約4区間目
    
    original_psd_tensor = torch.tensor(original_psd, dtype=torch.float32)
    
    # ノイズ区間の周波数範囲を計算
    noise_start_freq = (noise_interval * points_per_interval / num_points) * freq_max
    noise_end_freq = ((noise_interval + 1) * points_per_interval / num_points) * freq_max
    noise_start_idx = noise_interval * points_per_interval
    noise_end_idx = min((noise_interval + 1) * points_per_interval, num_points)
    
    # ノイズを特定の区間内にのみ適用（区間内に収まるように）
    noisy_psd = original_psd.copy()
    noise_frequencies = frequencies[noise_start_idx:noise_end_idx]
    
    # 電源ノイズの特性を保ちつつ、区間内に収まるようにノイズを生成
    base_amplitude = original_psd.mean() * 0.3
    
    # 2kHz付近にピークを持つノイズ（区間内に収まる）
    noise_center_freq = (noise_start_freq + noise_end_freq) / 2
    sigma_freq = (noise_end_freq - noise_start_freq) / 4  # 区間幅の1/4
    noise_peak = base_amplitude * np.exp(-0.5 * ((noise_frequencies - noise_center_freq) / sigma_freq) ** 2)
    
    # ノイズ区間内にのみノイズを追加
    noisy_psd[noise_start_idx:noise_end_idx] += noise_peak
    
    # ノイズ追加後も周波数0が最大値であることを保証
    if noisy_psd[0] < noisy_psd.max():
        noisy_psd[0] = noisy_psd.max() * 1.1
    
    # マスクを付与する区間（ノイズ区間以外から4区間を選択、隣り合わせにしない）
    available_intervals = [i for i in range(num_intervals) if i != noise_interval]
    
    # 隣り合わせにならないように選択
    masked_intervals = []
    used_intervals = set()
    
    # ランダムに選択するが、隣り合わせを避ける
    np.random.seed(42)  # 再現性のため
    while len(masked_intervals) < 4 and len(available_intervals) > 0:
        candidate = np.random.choice(available_intervals)
        # 隣り合わせでないことを確認
        if candidate not in used_intervals and (candidate - 1) not in used_intervals and (candidate + 1) not in used_intervals:
            masked_intervals.append(candidate)
            used_intervals.add(candidate)
            available_intervals.remove(candidate)
        elif len(available_intervals) == 1:
            # 最後の1つは強制的に追加
            masked_intervals.append(available_intervals[0])
            break
    
    masked_intervals = sorted(masked_intervals)  # ソートして見やすく
    
    # マスク付きデータを作成
    masked_noisy_psd = add_mask_to_intervals(noisy_psd, masked_intervals, num_intervals)
    
    # グラフ描画前に最終確認：周波数0が最大値であることを保証
    original_psd[0] = max(original_psd[0], original_psd.max() * 1.05)
    noisy_psd[0] = max(noisy_psd[0], noisy_psd.max() * 1.05)
    
    # 図を作成（上下2つのサブプロット、より洗練されたデザイン）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.patch.set_facecolor('white')
    
    # ノイズ区間の周波数範囲を計算
    noise_start_freq = (noise_interval * points_per_interval / num_points) * freq_max
    noise_end_freq = ((noise_interval + 1) * points_per_interval / num_points) * freq_max
    
    # ===== 上: 主タスク - マスク予測 =====
    # 背景にマスク区間を表示
    for interval_idx in masked_intervals:
        start_freq = (interval_idx * points_per_interval / num_points) * freq_max
        end_freq = ((interval_idx + 1) * points_per_interval / num_points) * freq_max
        ax1.axvspan(start_freq, end_freq, alpha=0.2, color='green', zorder=0)
    
    # ノイズ区間を背景に表示
    ax1.axvspan(noise_start_freq, noise_end_freq, alpha=0.2, color='red', zorder=0)
    
    # PSDデータをプロット（3000点すべてをプロット）
    ax1.plot(frequencies, noisy_psd, label='Noisy PSD', color='#2E86AB', linestyle='--', linewidth=2.5, alpha=0.9, zorder=2)
    ax1.plot(frequencies, masked_noisy_psd, label='Masked PSD', color='#A23B72', linewidth=3, zorder=3)
    
    # 左端と右端を0.7%ずつ切る
    ax1.set_xlim(freq_max * 0.007, freq_max * 0.993)
    y_min, y_max = ax1.get_ylim()
    ax1.set_ylim(y_min, y_max * 1.1)
    ax1.set_ylabel('パワースペクトル密度', fontsize=16, fontweight='bold')
    ax1.set_title('主タスク: マスク予測', fontsize=20, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=13, framealpha=0.9, edgecolor='gray', frameon=True)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(labelsize=12)
    
    # ===== 下: 副タスク - ノイズ検知 =====
    # ノイズ区間を背景に表示
    ax2.axvspan(noise_start_freq, noise_end_freq, alpha=0.2, color='red', zorder=0)
    
    # PSDデータをプロット（3000点すべてをプロット）
    ax2.plot(frequencies, original_psd, label='Original PSD', color='#F18F01', linewidth=3, zorder=2)
    ax2.plot(frequencies, noisy_psd, label='Noisy PSD', color='#C73E1D', linewidth=2.5, alpha=0.9, zorder=3)
    
    # 左端と右端を0.7%ずつ切る
    ax2.set_xlim(freq_max * 0.007, freq_max * 0.993)
    y_min, y_max = ax2.get_ylim()
    ax2.set_ylim(y_min, y_max * 1.1)
    ax2.set_xlabel('周波数 (Hz)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('パワースペクトル密度', fontsize=16, fontweight='bold')
    ax2.set_title('副タスク: ノイズ予測・復元', fontsize=20, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=13, framealpha=0.9, edgecolor='gray', frameon=True)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(labelsize=12)
    
    # 主タスクと副タスクの間隔を広げる（tight_layoutを使わずに手動調整）
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, hspace=0.4)
    
    # 保存
    output_path = os.path.join(os.path.dirname(__file__), 'ssl_tasks_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"画像を保存しました: {output_path}")
    
    plt.close()

if __name__ == '__main__':
    visualize_ssl_tasks()

