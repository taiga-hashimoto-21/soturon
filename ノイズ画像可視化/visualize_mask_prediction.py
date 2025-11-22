"""
マスク予測タスクの説明用画像生成
- 4つの緑のマスク区間を表示
- 「マスクをした箇所の復元」という説明文を追加
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

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

def visualize_mask_prediction():
    """マスク予測タスクの説明画像を生成"""
    
    # パラメータ設定
    num_points = 3000
    num_intervals = 30
    points_per_interval = num_points // num_intervals
    freq_max = 15000.0
    
    # ベースPSDデータを生成
    original_psd, frequencies = generate_psd_data(num_points, freq_max)
    
    # ノイズ区間を決定（2kHz付近、区間4）
    freq_per_interval = freq_max / num_intervals  # 500Hz per interval
    noise_freq = 2000.0  # 2kHz
    noise_interval = int(noise_freq / freq_per_interval)  # 約4区間目
    
    # ノイズを特定の区間内にのみ適用
    noisy_psd = original_psd.copy()
    noise_start_idx = noise_interval * points_per_interval
    noise_end_idx = min((noise_interval + 1) * points_per_interval, num_points)
    noise_frequencies = frequencies[noise_start_idx:noise_end_idx]
    
    # 電源ノイズの特性を保ちつつ、区間内に収まるようにノイズを生成
    base_amplitude = original_psd.mean() * 0.3
    noise_start_freq = (noise_interval * points_per_interval / num_points) * freq_max
    noise_end_freq = ((noise_interval + 1) * points_per_interval / num_points) * freq_max
    noise_center_freq = (noise_start_freq + noise_end_freq) / 2
    sigma_freq = (noise_end_freq - noise_start_freq) / 4
    noise_peak = base_amplitude * np.exp(-0.5 * ((noise_frequencies - noise_center_freq) / sigma_freq) ** 2)
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
    masked_psd = noisy_psd.copy()
    for interval_idx in masked_intervals:
        start_idx = interval_idx * points_per_interval
        end_idx = min(start_idx + points_per_interval, num_points)
        masked_psd[start_idx:end_idx] = 0.0
    
    # グラフ描画前に最終確認：周波数0が最大値であることを保証
    original_psd[0] = max(original_psd[0], original_psd.max() * 1.05)
    noisy_psd[0] = max(noisy_psd[0], noisy_psd.max() * 1.05)
    
    # 図を作成
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    fig.patch.set_facecolor('white')
    
    # ノイズ区間の周波数範囲を計算
    noise_start_freq = (noise_interval * points_per_interval / num_points) * freq_max
    noise_end_freq = ((noise_interval + 1) * points_per_interval / num_points) * freq_max
    
    # 背景にマスク区間を表示（緑色）
    for interval_idx in masked_intervals:
        start_freq = (interval_idx * points_per_interval / num_points) * freq_max
        end_freq = ((interval_idx + 1) * points_per_interval / num_points) * freq_max
        ax.axvspan(start_freq, end_freq, alpha=0.2, color='green', zorder=0)
    
    # ノイズ区間を背景に表示（赤色）
    ax.axvspan(noise_start_freq, noise_end_freq, alpha=0.2, color='red', zorder=0)
    
    # PSDデータをプロット
    ax.plot(frequencies, noisy_psd, label='Noisy PSD', color='#2E86AB', linestyle='--', linewidth=2.5, alpha=0.9, zorder=2)
    ax.plot(frequencies, masked_psd, label='Masked PSD', color='#A23B72', linewidth=3, zorder=3)
    
    # 左端と右端を0.7%ずつ切る
    ax.set_xlim(freq_max * 0.007, freq_max * 0.993)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.1)
    ax.set_xlabel('周波数 (Hz)', fontsize=16, fontweight='bold')
    ax.set_ylabel('パワースペクトル密度', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=13, framealpha=0.9, edgecolor='gray', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    
    # 4つの緑のマスク区間を指して「マスクをした箇所の復元」という説明文を追加
    # マスク区間の中央位置を計算
    mask_centers = []
    for interval_idx in masked_intervals:
        start_freq = (interval_idx * points_per_interval / num_points) * freq_max
        end_freq = ((interval_idx + 1) * points_per_interval / num_points) * freq_max
        center_freq = (start_freq + end_freq) / 2
        mask_centers.append(center_freq)
    
    # 説明文の位置（マスク区間の中央あたり）
    text_x = np.mean(mask_centers)
    text_y = ax.get_ylim()[1] * 0.85
    
    # 説明文を追加
    ax.text(text_x, text_y, 'マスクをした箇所の復元', 
            fontsize=30, fontweight='bold', color='green',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='green', linewidth=2),
            zorder=10)
    
    # 矢印でマスク区間を指す
    for center_freq in mask_centers:
        ax.annotate('', xy=(center_freq, text_y * 0.7), xytext=(center_freq, text_y * 0.95),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2.5), zorder=9)
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(project_root, 'mask_prediction.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"画像を保存しました: {output_path}")
    
    plt.close()

if __name__ == '__main__':
    visualize_mask_prediction()

