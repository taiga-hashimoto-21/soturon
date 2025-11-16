"""
タスク4用のデータセット
マスク予測 + 正則化項のためのデータセット
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys
import os

# noiseモジュールをインポート（プロジェクトルートから）
# ssl/task/ から noise/ へのパス: ../../noise/
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


class Task4Dataset(Dataset):
    """
    タスク4用のデータセット
    
    - PSDデータを30区間に分割
    - ランダムに15%の区間をマスク
    - ノイズ区間の情報を返す（正則化項用）
    """
    
    def __init__(
        self,
        pickle_path: str,
        num_intervals: int = 30,
        mask_ratio: float = 0.15,
        noise_type: str = 'frequency_band',
        noise_level: float = 0.3,
        add_structured_noise_flag: bool = True,
        structured_noise_clip_range: float = 0.5,
        structured_noise_smoothing_factor: float = 0.1,
    ):
        """
        Args:
            pickle_path: data_lowF_noise.pickleのパス
            num_intervals: 区間数（デフォルト: 30）
            mask_ratio: マスクする区間の割合（デフォルト: 0.15 = 15%）
            noise_type: ノイズタイプ（'frequency_band', 'localized_spike', 'amplitude_dependent'）
            noise_level: ノイズレベル（デフォルト: 0.3 = 30%）
            add_structured_noise_flag: 全体的な構造化ノイズを付与するか（デフォルト: True）
            structured_noise_clip_range: 構造化ノイズのクリッピング範囲（デフォルト: 0.5）
            structured_noise_smoothing_factor: 構造化ノイズのスムージング係数（デフォルト: 0.1）
        """
        super().__init__()
        
        self.pickle_path = Path(pickle_path)
        self.num_intervals = num_intervals
        self.mask_ratio = mask_ratio
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.add_structured_noise_flag = add_structured_noise_flag
        self.structured_noise_clip_range = structured_noise_clip_range
        self.structured_noise_smoothing_factor = structured_noise_smoothing_factor
        
        # データを読み込み
        self.data = self._load_pickle(self.pickle_path)
        
        # データの形状を確認・調整
        if isinstance(self.data, dict):
            self.x = np.asarray(self.data['x'], dtype=np.float32)
        else:
            self.x = np.asarray(self.data, dtype=np.float32)
        
        # (N, 1, L) → (N, L)
        if self.x.ndim == 3 and self.x.shape[1] == 1:
            self.x = self.x.squeeze(1)
        
        assert self.x.ndim == 2, f"data shape must be (N, L), got {self.x.shape}"
        
        self.seq_len = self.x.shape[1]
        self.points_per_interval = self.seq_len // self.num_intervals
        
        # baselineと同じ方法: 全データセットで統一した正規化のための統計を事前計算
        # 注意: 動的にノイズを付与するため、代表的なサンプルで統計を計算
        print("全データセットの正規化統計を計算中（baselineと同じ方法）...")
        self.scale_factor = 2.5e24
        self.normalization_mean, self.normalization_std = self._compute_global_normalization_stats()
        
        print(f"Dataset loaded:")
        print(f"  Samples: {len(self.x)}")
        print(f"  Sequence length: {self.seq_len}")
        print(f"  Intervals: {self.num_intervals}")
        print(f"  Points per interval: {self.points_per_interval}")
        print(f"  Noise type: {self.noise_type}")
        print(f"  Noise level: {self.noise_level}")
        print(f"  Add structured noise: {self.add_structured_noise_flag}")
        print(f"  Normalization mean: {self.normalization_mean:.6f}")
        print(f"  Normalization std: {self.normalization_std:.6f}")
    
    def _load_pickle(self, path: Path):
        """pickleファイルを読み込む"""
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def _compute_global_normalization_stats(self, num_samples_for_stats=1000):
        """
        全データセットで統一した正規化のための統計を計算（baselineと同じ方法）
        
        Args:
            num_samples_for_stats: 統計計算に使用するサンプル数（デフォルト: 1000）
        
        Returns:
            mean: 正規化用の平均値
            std: 正規化用の標準偏差
        """
        # 代表的なサンプルを選択（ランダムに選択）
        np.random.seed(42)  # 再現性のため
        sample_indices = np.random.choice(len(self.x), size=min(num_samples_for_stats, len(self.x)), replace=False)
        
        # 各サンプルに対してノイズを付与し、ログ変換後の値を収集
        log_values_list = []
        
        for idx in sample_indices:
            # 元のPSDデータ
            base_psd = self.x[idx].astype(np.float32)
            base_psd = np.nan_to_num(base_psd, nan=0.0, posinf=0.0, neginf=0.0)
            base_psd_tensor = torch.from_numpy(base_psd)
            
            # 構造化ノイズを付与（baselineと同じ）
            if self.add_structured_noise_flag:
                structured_noisy_psd = add_structured_noise(
                    base_psd_tensor,
                    clip_range=self.structured_noise_clip_range,
                    smoothing_factor=self.structured_noise_smoothing_factor
                )
            else:
                structured_noisy_psd = base_psd_tensor
            
            # 区間ノイズを付与（ランダムに1区間）
            noise_interval = np.random.randint(0, self.num_intervals)
            noisy_psd_tensor, _, _ = add_noise_to_interval(
                structured_noisy_psd,
                noise_interval,
                noise_type=self.noise_type,
                noise_level=self.noise_level,
                num_intervals=self.num_intervals
            )
            
            # スケーリング + ログ変換（baselineと同じ）
            noisy_psd = noisy_psd_tensor.numpy()
            noisy_psd_scaled = noisy_psd * self.scale_factor
            noisy_psd_scaled = np.maximum(noisy_psd_scaled, 1e-30)  # 負の値を1e-30にクリップ
            noisy_psd_log = np.log(noisy_psd_scaled)
            
            log_values_list.append(noisy_psd_log)
        
        # 全サンプルのログ変換後の値を結合して統計を計算
        all_log_values = np.concatenate(log_values_list, axis=0)
        mean = float(all_log_values.mean())
        std = float(all_log_values.std())
        
        if std < 1e-6:
            std = 1.0
        
        return mean, std
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        """
        データを取得
        
        Returns:
            dict: {
                'input': ノイズ付与 + マスクされたPSDデータ (seq_len,)
                'target': ノイズ付与されたPSDデータ（マスクなし） (seq_len,)
                'mask': マスク位置 (seq_len,) [True/False]
                'noise_interval': ノイズを付与した区間のインデックス (int)
            }
        """
        # 元のPSDデータ（ノイズなし）
        base_psd = self.x[idx].astype(np.float32)
        
        # NaN / inf 対策
        base_psd = np.nan_to_num(base_psd, nan=0.0, posinf=0.0, neginf=0.0)
        
        L = base_psd.shape[0]
        
        # numpy配列をtorchテンソルに変換
        base_psd_tensor = torch.from_numpy(base_psd)
        
        # 1. 全体的な構造化ノイズを付与（実験データに近づけるため）
        if self.add_structured_noise_flag:
            structured_noisy_psd = add_structured_noise(
                base_psd_tensor,
                clip_range=self.structured_noise_clip_range,
                smoothing_factor=self.structured_noise_smoothing_factor
            )
        else:
            structured_noisy_psd = base_psd_tensor
        
        # 2. ランダムに1つの区間を選ぶ（ノイズを付与する区間）
        noise_interval = np.random.randint(0, self.num_intervals)
        
        # 3. 区間ノイズを付与（ノイズ検知の対象）
        noisy_psd_tensor, start_idx, end_idx = add_noise_to_interval(
            structured_noisy_psd,
            noise_interval,
            noise_type=self.noise_type,
            noise_level=self.noise_level,
            num_intervals=self.num_intervals
        )
        
        # numpy配列に戻す
        noisy_psd = noisy_psd_tensor.numpy()
        
        # マスク位置を決定（15%の区間をランダムにマスク、ノイズ区間は除外）
        # ノイズ区間以外の区間から選択
        available_intervals = [i for i in range(self.num_intervals) if i != noise_interval]
        num_masked_intervals = max(1, int(len(available_intervals) * self.mask_ratio))
        num_masked_intervals = min(num_masked_intervals, len(available_intervals))
        
        masked_interval_indices = np.random.choice(
            available_intervals,
            size=num_masked_intervals,
            replace=False
        )
        
        # マスク位置を計算
        mask_positions = np.zeros(L, dtype=bool)
        for interval_idx in masked_interval_indices:
            start_idx_mask = interval_idx * self.points_per_interval
            end_idx_mask = min(start_idx_mask + self.points_per_interval, L)
            mask_positions[start_idx_mask:end_idx_mask] = True
        
        # マスクされたデータを作成（ノイズが付与されたデータにマスクを適用）
        masked_noisy_psd = noisy_psd.copy()
        masked_noisy_psd[mask_positions] = 0.0  # マスクされた部分を0で置き換え
        
        # ベースラインと同じ前処理: スケーリング + ログ変換
        noisy_psd_scaled = noisy_psd * self.scale_factor
        masked_noisy_psd_scaled = masked_noisy_psd * self.scale_factor
        
        # ベースラインと同じ方法: clampで負の値を防ぐ
        # 負の値や0を1e-30にクリップしてからログ変換
        noisy_psd_scaled = np.maximum(noisy_psd_scaled, 1e-30)
        masked_noisy_psd_scaled = np.maximum(masked_noisy_psd_scaled, 1e-30)
        
        # ログ変換前の最終チェック: 負の値や0を1e-30にクリップ
        noisy_psd_scaled = np.maximum(noisy_psd_scaled, 1e-30)
        masked_noisy_psd_scaled = np.maximum(masked_noisy_psd_scaled, 1e-30)
        
        # ログ変換（マスクされた部分は0のまま）
        noisy_psd_log = np.log(noisy_psd_scaled)
        masked_noisy_psd_log = np.log(masked_noisy_psd_scaled)
        # マスクされた部分は0のまま
        masked_noisy_psd_log[mask_positions] = 0.0
        
        # 正規化（baselineと同じ方法: 全データセットで統一した平均・標準偏差を使用）
        noisy_psd_norm = (noisy_psd_log - self.normalization_mean) / (self.normalization_std + 1e-8)
        masked_noisy_psd_norm = (masked_noisy_psd_log - self.normalization_mean) / (self.normalization_std + 1e-8)
        
        return {
            'input': torch.tensor(masked_noisy_psd_norm, dtype=torch.float32),
            'target': torch.tensor(noisy_psd_norm, dtype=torch.float32),
            'mask': torch.tensor(mask_positions, dtype=torch.bool),
            'noise_interval': torch.tensor(noise_interval, dtype=torch.long),
        }

