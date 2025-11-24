"""
タスク4用のBERTモデル
マスク予測 + アテンションウェイト取得
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Task4BERT(nn.Module):
    """
    タスク4用のBERTモデル
    
    - CLSトークンを使用
    - マスク予測タスク
    - アテンションウェイトを取得可能
    """
    
    def __init__(
        self,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args:
            seq_len: 入力系列の長さ（3000）
            d_model: 埋め込み次元
            n_heads: アテンションヘッド数
            num_layers: Transformerレイヤー数
            dim_feedforward: Feedforward層の次元
            dropout: Dropout率
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 連続値 -> 埋め込み
        self.value_proj = nn.Linear(1, d_model)
        
        # 位置埋め込み（[CLS] 用に +1）
        self.pos_embed = nn.Embedding(seq_len + 1, d_model)
        
        # [CLS] トークン & [MASK] トークン
        self.cls_token = nn.Parameter(torch.randn(d_model))
        self.mask_token = nn.Parameter(torch.randn(d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # (seq_len, batch, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 出力ヘッド（各位置 → スカラー）
        self.output_head = nn.Linear(d_model, 1)
        
        # 復元ヘッド（ノイズ検知した区間のみを復元）
        # 区間の特徴（d_model次元）から100ポイントを復元
        points_per_interval = seq_len // 30  # 30区間、1区間=100ポイント
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, points_per_interval)  # 100ポイントを復元
        )
        
        # アテンションウェイトを保存するためのリスト
        self.attention_weights_list = []
        
        # アテンションフック用のハンドル
        self.hooks = []
    
    def forward(self, x, mask_positions, return_attention=False, num_intervals=30, true_noise_intervals=None):
        """
        Args:
            x: (B, L) マスクされたPSDデータ
            mask_positions: (B, L) マスク位置 [True/False]
            return_attention: アテンションウェイトを返すかどうか
            num_intervals: 区間数（デフォルト: 30）
            true_noise_intervals: (B,) 真のノイズ区間（予測が外れた場合の復元用、Noneの場合は予測区間を使用）
        
        Returns:
            out: (B, L) 予測されたPSDデータ（マスク予測タスク用）
            cls_out: (B, d_model) CLSトークンの表現
            attention_weights: (B, n_heads, L+1, L+1) アテンションウェイト（return_attention=Trueの場合）
            reconstructed_interval: (B, max_length) 復元された区間（予測区間±1区間、最大300ポイント、return_attention=Trueの場合のみ）
            reconstructed_intervals_info: (B, 3) 復元範囲の情報 [start_interval, end_interval, predicted_interval]（return_attention=Trueの場合のみ）
        """
        B, L = x.shape
        assert L == self.seq_len, f"seq_len mismatch: expected {self.seq_len}, got {L}"
        
        # アテンションウェイトをリセット
        if return_attention:
            self.attention_weights_list = []
            self._register_attention_hooks()
        
        # (B, L, 1) → (B, L, d_model)
        x_embed = self.value_proj(x.unsqueeze(-1))
        
        # マスクトークンで置き換え
        mask_token = self.mask_token.view(1, 1, -1)
        x_embed = torch.where(
            mask_positions.unsqueeze(-1), 
            mask_token.expand_as(x_embed), 
            x_embed
        )
        
        # 位置埋め込み
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        pos_embed = self.pos_embed(pos_ids)
        x_embed = x_embed + pos_embed
        
        # CLS トークン
        cls_token = self.cls_token.view(1, 1, -1).expand(B, 1, self.d_model)
        x_with_cls = torch.cat([cls_token, x_embed], dim=1)  # (B, L+1, d_model)
        
        # Transformer (S, B, E)
        x_with_cls = x_with_cls.transpose(0, 1)  # (L+1, B, d_model)
        
        # アテンションウェイトを取得するために、最後のレイヤーで手動計算
        attention_weights = None
        if return_attention:
            # エンコーダーを通しながら、最後のレイヤーまで処理
            x_temp = x_with_cls
            for i, layer in enumerate(self.encoder.layers[:-1]):
                x_temp = layer(x_temp)
            
            # 最後のレイヤーを通す
            last_layer = self.encoder.layers[-1]
            encoded = last_layer(x_temp)
            
            # 最後のレイヤーを通した後の表現でアテンションを計算（修正）
            # 注意: 実際のアテンションはレイヤーを通す前の入力で計算されるが、
            # ここでは最後のレイヤーの入力（x_temp）で計算する
            attention_weights = self._compute_attention_weights(
                last_layer, x_temp, B, L
            )
        else:
            # アテンション不要の場合は通常通り
            encoded = self.encoder(x_with_cls)
        
        # 系列部分取り出し
        encoded_seq = encoded[1:].transpose(0, 1)  # (B, L, d_model)
        
        # 出力（回帰）
        out = self.output_head(encoded_seq).squeeze(-1)  # (B, L)
        
        # CLS も返す
        cls_out = encoded[0].transpose(0, 1)  # (B, d_model)
        
        # 復元機能: アテンションウェイトからノイズ区間を検知して復元
        # 予測区間を中心に左右2区間ずつ（合計5区間）を復元
        reconstructed_interval = None
        if return_attention and attention_weights is not None:
            # CLSトークンから各区間へのアテンションを取得
            cls_attention = self.get_cls_attention_to_intervals(attention_weights, num_intervals)  # (B, num_intervals)
            
            # 最もアテンションが低い区間をノイズ区間として検知
            predicted_noise_intervals = cls_attention.argmin(dim=1)  # (B,) - 予測されたノイズ区間
            
            # 予測した区間を中心に左右1区間ずつ（合計3区間）を復元
            # true_noise_intervalsが指定されている場合は、予測が外れたサンプルでも真のノイズ区間を復元
            points_per_interval = self.seq_len // num_intervals
            window_size = 1  # 左右に1区間ずつ
            reconstructed_intervals_list = []
            
            for i in range(B):
                # 真のノイズ区間が指定されている場合はそれを使用、そうでなければ予測区間を使用
                if true_noise_intervals is not None:
                    interval_to_reconstruct = true_noise_intervals[i].item()
                else:
                    interval_to_reconstruct = predicted_noise_intervals[i].item()
                
                # 復元する区間の範囲を決定（区間±1区間）
                start_interval = max(0, interval_to_reconstruct - window_size)
                end_interval = min(num_intervals - 1, interval_to_reconstruct + window_size)
                num_intervals_to_reconstruct = end_interval - start_interval + 1
                
                # 3区間分の特徴を取得して復元
                reconstructed_parts = []
                for interval_idx in range(start_interval, end_interval + 1):
                    # 各区間の開始・終了インデックス
                    start_idx = interval_idx * points_per_interval
                    end_idx = min(start_idx + points_per_interval, self.seq_len)
                    
                    # 区間のエンコードされた特徴を取得
                    interval_features = encoded_seq[i, start_idx:end_idx, :]  # (points_per_interval, d_model)
                    # 区間の特徴を平均化
                    interval_feature = interval_features.mean(dim=0)  # (d_model,)
                    
                    # 復元ヘッドで復元
                    reconstructed_interval_single = self.reconstruction_head(interval_feature.unsqueeze(0))  # (1, 100)
                    reconstructed_parts.append(reconstructed_interval_single.squeeze(0))  # (100,)
                
                # 3区間分を結合（合計300ポイント）
                reconstructed_combined = torch.cat(reconstructed_parts, dim=0)  # (num_intervals_to_reconstruct * 100,)
                reconstructed_intervals_list.append(reconstructed_combined)
            
            # バッチごとに異なる長さになる可能性があるため、最大長に合わせてパディング
            max_length = max(r.shape[0] for r in reconstructed_intervals_list)
            padded_reconstructed = []
            reconstructed_intervals_info_list = []
            
            for i, r in enumerate(reconstructed_intervals_list):
                predicted_interval = predicted_noise_intervals[i].item()
                # 真のノイズ区間が指定されている場合はそれを使用、そうでなければ予測区間を使用
                if true_noise_intervals is not None:
                    interval_to_reconstruct = true_noise_intervals[i].item()
                else:
                    interval_to_reconstruct = predicted_interval
                
                start_interval = max(0, interval_to_reconstruct - window_size)
                end_interval = min(num_intervals - 1, interval_to_reconstruct + window_size)
                
                if r.shape[0] < max_length:
                    # パディング（最後に0を追加）
                    padding = torch.zeros(max_length - r.shape[0], device=r.device, dtype=r.dtype)
                    r = torch.cat([r, padding], dim=0)
                padded_reconstructed.append(r)
                
                # 復元範囲の情報を保存
                reconstructed_intervals_info_list.append(torch.tensor([
                    start_interval, end_interval, predicted_interval
                ], device=r.device, dtype=torch.long))
            
            reconstructed_interval = torch.stack(padded_reconstructed, dim=0)  # (B, max_length)
            reconstructed_intervals_info = torch.stack(reconstructed_intervals_info_list, dim=0)  # (B, 3)
        
        if return_attention:
            return out, cls_out, attention_weights, reconstructed_interval, reconstructed_intervals_info
        else:
            return out, cls_out
    
    def _register_attention_hooks(self):
        """アテンションウェイトを取得するためのフックを登録"""
        # 既存のフックを削除
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_weights_list = []
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # MultiheadAttentionのforward内でアテンションを計算
                # input: (query, key, value, ...)
                # output: (attn_output, attn_output_weights) または attn_output
                if isinstance(output, tuple) and len(output) >= 2:
                    # attn_output_weightsが返される場合
                    attn_weights = output[1]  # (B, num_heads, L+1, L+1) または (L+1, B, num_heads, L+1)
                    if attn_weights is not None:
                        self.attention_weights_list.append(attn_weights.detach())
                else:
                    # アテンションウェイトが返されない場合、手動で計算
                    # これは簡易版で、正確な計算にはforwardをオーバーライドが必要
                    pass
            return hook
        
        # 各レイヤーのMultiheadAttentionにフックを登録
        for i, layer in enumerate(self.encoder.layers):
            hook = layer.self_attn.register_forward_hook(get_attention_hook(f'layer_{i}'))
            self.hooks.append(hook)
    
    def _compute_attention_weights(self, layer, x, B, L):
        """
        レイヤーでアテンションウェイトを計算
        
        Args:
            layer: TransformerEncoderLayer
            x: (L+1, B, d_model) 入力
            B: バッチサイズ
            L: 系列長
        
        Returns:
            attention_weights: (B, n_heads, L+1, L+1) アテンションウェイト
        """
        # MultiheadAttentionの内部計算を再現
        self_attn = layer.self_attn
        seq_len, batch_size, d_model = x.shape
        
        # xを (B, L+1, d_model) に変換
        x_reshaped = x.transpose(0, 1)  # (B, L+1, d_model)
        
        # MultiheadAttentionの内部実装に合わせて計算
        # in_proj_weight: (3*embed_dim, embed_dim)
        # in_proj_bias: (3*embed_dim,)
        embed_dim = self_attn.embed_dim
        
        # x_reshaped: (B, L+1, d_model) -> (B*L+1, d_model)にreshapeしてから計算
        B, seq_len, d_model = x_reshaped.shape
        x_flat = x_reshaped.reshape(-1, d_model)  # (B*(L+1), d_model)
        
        # 全結合層でq, k, vを一度に計算
        # qkv: (B*(L+1), 3*embed_dim)
        qkv = F.linear(x_flat, self_attn.in_proj_weight, self_attn.in_proj_bias)
        
        # q, k, vに分割してからreshape
        q = qkv[:, :embed_dim].reshape(B, seq_len, embed_dim)  # (B, L+1, embed_dim)
        k = qkv[:, embed_dim:2*embed_dim].reshape(B, seq_len, embed_dim)  # (B, L+1, embed_dim)
        v = qkv[:, 2*embed_dim:].reshape(B, seq_len, embed_dim)  # (B, L+1, embed_dim)
        
        # ヘッドに分割
        q = q.reshape(batch_size, seq_len, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        
        # スケーリング
        scale = math.sqrt(d_model // self.n_heads)
        
        # アテンションスコアを計算
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, n_heads, L+1, L+1)
        
        # ソフトマックス
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Dropoutを適用（学習時のみ、評価時は適用しない）
        # 注意: 評価時はmodel.eval()でself.training=Falseになる
        if self.training and layer.self_attn.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=layer.self_attn.dropout, training=self.training)
        
        return attn_weights
    
    def get_cls_attention_to_intervals(self, attention_weights, num_intervals=30):
        """
        CLSトークンから各区間へのアテンションウェイトを取得
        
        Args:
            attention_weights: (B, n_heads, L+1, L+1) アテンションウェイト
            num_intervals: 区間数
        
        Returns:
            cls_attention: (B, num_intervals) CLSトークンから各区間へのアテンション
        """
        if attention_weights is None:
            return None
        
        B = attention_weights.shape[0]
        L = self.seq_len
        points_per_interval = L // num_intervals
        
        # CLSトークンはインデックス0
        # 各区間はインデックス1から
        cls_attention_full = attention_weights[:, :, 0, 1:]  # (B, n_heads, L)
        
        # ヘッドごとの平均
        cls_attention_mean = cls_attention_full.mean(dim=1)  # (B, L)
        
        # 区間ごとに平均化
        cls_attention_intervals = []
        for i in range(num_intervals):
            start_idx = i * points_per_interval
            end_idx = min(start_idx + points_per_interval, L)
            interval_attn = cls_attention_mean[:, start_idx:end_idx].mean(dim=1)
            cls_attention_intervals.append(interval_attn)
        
        cls_attention = torch.stack(cls_attention_intervals, dim=1)  # (B, num_intervals)
        
        return cls_attention

