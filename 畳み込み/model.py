"""
ResNet風の1D CNNモデル（残差接続付き + GeMプーリング + SiLU）
10クラス分類（どの区間にノイズがあるか）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """
    Generalized Mean (GeM) プーリング層
    学習可能なパラメータpを持つプーリングで、より柔軟な特徴抽出が可能
    """
    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # 学習可能なパラメータp
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        # GeMプーリングを計算
        # x: (batch_size, channels, length)
        x_pooled = F.avg_pool1d(x.clamp(min=eps).pow(p), kernel_size=self.kernel_size, stride=self.kernel_size)
        return x_pooled.pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'


class ResidualBlock1D(nn.Module):
    """
    ResNet風の残差ブロック（1D用）
    SiLU活性化関数とGeMプーリングを追加
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None, use_gem=False):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU()  # SiLU活性化関数（ReLUより滑らか）
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.use_gem = use_gem
        if use_gem:
            self.gem = GeM(kernel_size=2)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_gem:
            out = self.gem(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # 残差接続
        out = self.silu(out)
        
        return out


class ResNet1D(nn.Module):
    """
    改善されたResNet風の1D CNNモデル
    - 残差接続で深いネットワークを学習可能に
    - GeMプーリングで柔軟な特徴抽出
    - SiLU活性化関数で学習の安定化
    - データの前処理（スケールとログ変換）を追加
    """
    
    def __init__(self, num_classes=30, dropout_rate=0.3, use_preprocessing=False):
        super(ResNet1D, self).__init__()
        
        self.use_preprocessing = use_preprocessing
        if use_preprocessing:
            # データの前処理用パラメータ（参考コードから）
            # ただし、すでに正規化されているので、スケールファクターは調整可能
            self.scale_factor = 1e20  # より小さな値に調整
        
        # 初期畳み込み層
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.silu = nn.SiLU()  # SiLU活性化関数
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 残差ブロック層1（GeMプーリングなし）
        self.layer1 = self._make_layer(64, 64, 2, stride=1, use_gem=False)
        
        # 残差ブロック層2（GeMプーリングあり）
        self.layer2 = self._make_layer(64, 128, 2, stride=2, use_gem=True)
        
        # 残差ブロック層3（GeMプーリングあり）
        self.layer3 = self._make_layer(128, 256, 2, stride=2, use_gem=True)
        
        # 残差ブロック層4（GeMプーリングなし）
        self.layer4 = self._make_layer(256, 512, 2, stride=2, use_gem=False)
        
        # 区間ごとの特徴を抽出（30区間）- GeMプーリングを使用
        self.gem_pool = GeM(kernel_size=8, p=3)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(30)
        
        # 全結合層
        self.fc1 = nn.Linear(512 * 30, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, use_gem=False):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride, downsample=downsample, use_gem=use_gem))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, use_gem=use_gem))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 入力: (batch_size, 3000)
        
        # データの前処理（オプション）
        if self.use_preprocessing:
            # スケールとログ変換（非常に小さい値を扱いやすくする）
            x = x * self.scale_factor
            x = torch.log(x.clamp(min=1e-30))  # 負の値を防ぐ
        
        x = x.unsqueeze(1)  # (batch_size, 1, 3000)
        
        # 初期畳み込み
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)
        x = self.maxpool(x)  # (batch_size, 64, 750)
        
        # 残差ブロック
        x = self.layer1(x)  # (batch_size, 64, 750)
        x = self.layer2(x)  # (batch_size, 128, 375)
        x = self.layer3(x)  # (batch_size, 256, 187)
        x = self.layer4(x)  # (batch_size, 512, 93)
        
        # GeMプーリングで特徴を強調
        x = self.gem_pool(x)  # (batch_size, 512, 約11)
        
        # 区間ごとの特徴を抽出（30区間）
        x = self.adaptive_pool(x)  # (batch_size, 512, 30)
        
        # フラット化
        x = x.view(x.size(0), -1)  # (batch_size, 512 * 30)
        
        # 全結合層
        x = self.fc1(x)
        x = self.silu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.silu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)  # (batch_size, 30)
        
        return x


class ImprovedCNN(nn.Module):
    """
    改善された1D CNNモデル
    - アテンション機構で局所的なノイズを強調
    - マルチスケール特徴抽出
    - ドロップアウトで過学習を防止
    """
    
    def __init__(self, num_classes=30, dropout_rate=0.5):
        super(ImprovedCNN, self).__init__()
        
        # 1D畳み込み層1
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=7,
            padding=3
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.attention1 = AttentionModule(32)
        
        # 1D畳み込み層2
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.attention2 = AttentionModule(64)
        
        # 1D畳み込み層3
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.attention3 = AttentionModule(128)
        
        # 追加の畳み込み層4（より深いネットワーク）
        self.conv4 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1
        )
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        
        # マルチスケール特徴抽出用のパス
        # 区間ごとの特徴を保持するため、Global Average Poolingの前に区間ごとにプーリング
        # 30区間 × 100ポイント = 3000ポイント
        # より細かく区間を捉えるため、Adaptive Poolingを使用
        self.adaptive_pool = nn.AdaptiveAvgPool1d(30)  # 30区間の特徴を抽出
        
        # 全結合層1
        self.fc1 = nn.Linear(256 * 30, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu_fc1 = nn.ReLU()
        
        # 全結合層2
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.relu_fc2 = nn.ReLU()
        
        # 出力層
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # 入力: (batch_size, 3000)
        # チャンネル次元を追加: (batch_size, 1, 3000)
        x = x.unsqueeze(1)
        
        # 畳み込み層1 + アテンション
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)  # (batch_size, 32, 1500)
        x = self.attention1(x)
        
        # 畳み込み層2 + アテンション
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)  # (batch_size, 64, 750)
        x = self.attention2(x)
        
        # 畳み込み層3 + アテンション
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)  # (batch_size, 128, 375)
        x = self.attention3(x)
        
        # 畳み込み層4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)  # (batch_size, 256, 187)
        
        # 区間ごとの特徴を抽出（30区間）
        x = self.adaptive_pool(x)  # (batch_size, 256, 30)
        
        # フラット化
        x = x.view(x.size(0), -1)  # (batch_size, 256 * 30)
        
        # 全結合層1
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu_fc1(x)
        
        # 全結合層2
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu_fc2(x)
        
        # 出力層
        x = self.fc3(x)  # (batch_size, num_classes)
        
        return x


class MultiScaleResNet1D(nn.Module):
    """
    マルチスケールResNet1Dモデル
    - 3つの異なるスケール（Low, Middle, High）で並列処理
    - 各スケールに専用のResNet1Dを使用
    - 特徴を結合して最終分類
    """
    
    def __init__(self, num_classes=30, dropout_rate=0.3, use_preprocessing=True):
        super(MultiScaleResNet1D, self).__init__()
        
        self.use_preprocessing = use_preprocessing
        if use_preprocessing:
            self.scale_factor = 1e20
        
        # Lowスケール用（0-300ポイント）- 細かい特徴
        self.low_net = self._make_small_resnet(64)
        
        # Middleスケール用（300-1500ポイント）- 中程度の特徴
        self.middle_net = self._make_medium_resnet(128)
        
        # Highスケール用（1500-3000ポイント）- 広範囲の特徴
        self.high_net = self._make_large_resnet(256)
        
        # 特徴を結合して分類
        total_features = 64 + 128 + 256
        self.fc1 = nn.Linear(total_features * 30, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
        self.silu = nn.SiLU()
        
    def _make_small_resnet(self, out_channels):
        """小さなResNet（細かい特徴用）"""
        layers = []
        layers.append(nn.Conv1d(1, 32, kernel_size=5, padding=2))
        layers.append(nn.BatchNorm1d(32))
        layers.append(nn.SiLU())
        layers.append(nn.MaxPool1d(2))
        
        # 残差ブロック
        layers.append(self._make_residual_block(32, 32))
        layers.append(self._make_residual_block(32, 64))
        
        layers.append(GeM(kernel_size=4))
        layers.append(nn.AdaptiveAvgPool1d(30))
        
        return nn.Sequential(*layers)
    
    def _make_medium_resnet(self, out_channels):
        """中程度のResNet（中程度の特徴用）"""
        layers = []
        layers.append(nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3))
        layers.append(nn.BatchNorm1d(64))
        layers.append(nn.SiLU())
        layers.append(nn.MaxPool1d(2))
        
        # 残差ブロック
        layers.append(self._make_residual_block(64, 64))
        layers.append(self._make_residual_block(64, 128))
        
        layers.append(GeM(kernel_size=4))
        layers.append(nn.AdaptiveAvgPool1d(30))
        
        return nn.Sequential(*layers)
    
    def _make_large_resnet(self, out_channels):
        """大きなResNet（広範囲の特徴用）"""
        layers = []
        layers.append(nn.Conv1d(1, 128, kernel_size=15, stride=4, padding=7))
        layers.append(nn.BatchNorm1d(128))
        layers.append(nn.SiLU())
        layers.append(nn.MaxPool1d(2))
        
        # 残差ブロック
        layers.append(self._make_residual_block(128, 128))
        layers.append(self._make_residual_block(128, 256))
        
        layers.append(GeM(kernel_size=4))
        layers.append(nn.AdaptiveAvgPool1d(30))
        
        return nn.Sequential(*layers)
    
    def _make_residual_block(self, in_channels, out_channels):
        """簡易的な残差ブロック（残差接続付き）"""
        class ResidualBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm1d(out_ch)
                self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm1d(out_ch)
                self.silu = nn.SiLU()
                
                # 残差接続用のダウンサンプル
                if in_ch != out_ch:
                    self.downsample = nn.Sequential(
                        nn.Conv1d(in_ch, out_ch, kernel_size=1),
                        nn.BatchNorm1d(out_ch)
                    )
                else:
                    self.downsample = None
            
            def forward(self, x):
                identity = x
                
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.silu(out)
                
                out = self.conv2(out)
                out = self.bn2(out)
                
                if self.downsample is not None:
                    identity = self.downsample(x)
                
                out += identity
                out = self.silu(out)
                
                return out
        
        return ResidualBlock(in_channels, out_channels)
    
    def forward(self, x):
        # 入力: (batch_size, 3000)
        
        # データの前処理
        if self.use_preprocessing:
            x = x * self.scale_factor
            x = torch.log(x.clamp(min=1e-30))
        
        # 3つのスケールに分割
        low = x[:, :300]      # 0-300ポイント
        middle = x[:, 300:1500]  # 300-1500ポイント
        high = x[:, 1500:]    # 1500-3000ポイント
        
        # 各スケールを処理
        low = low.unsqueeze(1)    # (batch_size, 1, 300)
        middle = middle.unsqueeze(1)  # (batch_size, 1, 1200)
        high = high.unsqueeze(1)  # (batch_size, 1, 1500)
        
        low_feat = self.low_net(low)      # (batch_size, 64, 10)
        middle_feat = self.middle_net(middle)  # (batch_size, 128, 10)
        high_feat = self.high_net(high)   # (batch_size, 256, 10)
        
        # 特徴を結合
        low_feat = low_feat.view(low_feat.size(0), -1)  # (batch_size, 64*10)
        middle_feat = middle_feat.view(middle_feat.size(0), -1)  # (batch_size, 128*10)
        high_feat = high_feat.view(high_feat.size(0), -1)  # (batch_size, 256*10)
        
        combined = torch.cat([low_feat, middle_feat, high_feat], dim=1)  # (batch_size, 4480)
        
        # 全結合層
        x = self.fc1(combined)
        x = self.silu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.silu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)  # (batch_size, 30)
        
        return x


class SimpleResNet1D(nn.Module):
    """
    シンプルなResNet1Dモデル（確実に学習できるように）
    - 前処理なし（既に正規化済み）
    - シンプルな構造
    - 確実に勾配が流れる
    """
    
    def __init__(self, num_classes=30, dropout_rate=0.3):
        super(SimpleResNet1D, self).__init__()
        
        # 初期畳み込み層
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 残差ブロック層1
        self.layer1 = self._make_layer(64, 64, 2)
        
        # 残差ブロック層2
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        
        # 区間ごとの特徴を抽出（30区間）
        self.adaptive_pool = nn.AdaptiveAvgPool1d(30)
        
        # 全結合層
        self.fc1 = nn.Linear(128 * 30, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride, downsample=downsample, use_gem=False))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, use_gem=False))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 入力: (batch_size, 3000)
        # 注意: データセット準備時に既にスケーリング + ログ変換 + 正規化が完了しているため、
        # モデル内での前処理は不要
        
        x = x.unsqueeze(1)  # (batch_size, 1, 3000)
        
        # 初期畳み込み
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (batch_size, 64, 750)
        
        # 残差ブロック
        x = self.layer1(x)  # (batch_size, 64, 750)
        x = self.layer2(x)  # (batch_size, 128, 375)
        
        # 区間ごとの特徴を抽出
        x = self.adaptive_pool(x)  # (batch_size, 128, 10)
        
        # フラット化
        x = x.view(x.size(0), -1)  # (batch_size, 128 * 10)
        
        # 全結合層
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)  # (batch_size, 30)
        
        return x


# 後方互換性のため、SimpleCNNも残す（既存コードとの互換性）
class SimpleCNN(nn.Module):
    """
    簡単な1D CNNモデル（後方互換性のため）
    シンプルなResNet1Dを使用（確実に学習できるように）
    前処理なしで確実に動作する
    """
    
    def __init__(self, num_classes=30):
        super(SimpleCNN, self).__init__()
        
        # SimpleResNet1Dを使用（シンプルで確実に学習できる、前処理なし）
        self.model = SimpleResNet1D(num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


class NoiseDetectionAndReconstructionModel(nn.Module):
    """
    ノイズ検出 + 復元モデル（改善版）
    - マスク予測: どの区間にノイズがあるか（30区間のノイズ強度）
    - 復元: 周辺区間の情報を使ってノイズ区間を復元（3000ポイント）
    """
    
    def __init__(self, num_intervals=30, points_per_interval=100, dropout_rate=0.3):
        super(NoiseDetectionAndReconstructionModel, self).__init__()
        
        self.num_intervals = num_intervals
        self.points_per_interval = points_per_interval
        
        # Encoder部分（既存のSimpleResNet1Dの特徴抽出部分を再利用）
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 残差ブロック層
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        
        # 区間ごとの特徴を抽出（30区間）
        self.adaptive_pool = nn.AdaptiveAvgPool1d(num_intervals)
        
        # 共有特徴量（Encoderの出力）
        self.shared_features_dim = 128 * num_intervals
        self.interval_feature_dim = 128  # 各区間の特徴次元
        
        # Mask Head: ノイズマスク予測（30区間のノイズ強度）
        self.mask_fc1 = nn.Linear(self.shared_features_dim, 256)
        self.mask_dropout1 = nn.Dropout(dropout_rate)
        self.mask_fc2 = nn.Linear(256, 128)
        self.mask_dropout2 = nn.Dropout(dropout_rate)
        self.mask_fc3 = nn.Linear(128, num_intervals)  # 各区間のノイズ強度
        self.mask_sigmoid = nn.Sigmoid()  # 0-1に正規化
        
        # アテンション機構: 周辺区間から情報を集約
        self.attention = nn.MultiheadAttention(
            embed_dim=self.interval_feature_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=False
        )
        
        # 区間レベルの特徴抽出（各区間の100点から特徴を抽出）
        self.interval_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # (batch_size, 64, 1)
        )
        
        # 復元用デコーダー（区間ごとに復元）
        self.interval_decoder = nn.Sequential(
            nn.Linear(self.interval_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, points_per_interval)  # 100点を復元
        )
        
        # 全体の復元（区間を結合）
        self.final_reconstruction = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride, downsample=downsample, use_gem=False))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, use_gem=False))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: ノイズ付きPSDデータ (batch_size, 3000)
        
        Returns:
            mask: ノイズマスク (batch_size, 30) - 各区間のノイズ強度
            reconstructed_interval: 復元された区間 (batch_size, 100) - 予測した区間のみ
        """
        batch_size = x.size(0)
        
        # ===== 1. Encoder部分: 全体の特徴抽出 =====
        x_conv = x.unsqueeze(1)  # (batch_size, 1, 3000)
        
        x_conv = self.conv1(x_conv)
        x_conv = self.bn1(x_conv)
        x_conv = self.relu(x_conv)
        x_conv = self.maxpool(x_conv)  # (batch_size, 64, 750)
        
        x_conv = self.layer1(x_conv)  # (batch_size, 64, 750)
        x_conv = self.layer2(x_conv)  # (batch_size, 128, 375)
        
        # 区間ごとの特徴を抽出
        interval_features_conv = self.adaptive_pool(x_conv)  # (batch_size, 128, 30)
        
        # フラット化（マスク予測用）
        shared_features = interval_features_conv.view(batch_size, -1)  # (batch_size, 128 * 30)
        
        # ===== 2. Mask Head: ノイズマスク予測 =====
        mask = self.mask_fc1(shared_features)
        mask = self.relu(mask)
        mask = self.mask_dropout1(mask)
        
        mask = self.mask_fc2(mask)
        mask = self.relu(mask)
        mask = self.mask_dropout2(mask)
        
        mask = self.mask_fc3(mask)  # (batch_size, 30)
        mask = self.mask_sigmoid(mask)  # 0-1に正規化
        
        # ===== 3. 区間ごとの特徴抽出（100点から） =====
        # 30区間に分割
        intervals = x.view(batch_size, self.num_intervals, self.points_per_interval)  # (batch_size, 30, 100)
        
        # 各区間の特徴を抽出
        interval_features_list = []
        for i in range(self.num_intervals):
            interval = intervals[:, i, :].unsqueeze(1)  # (batch_size, 1, 100)
            interval_feat = self.interval_encoder(interval)  # (batch_size, 64, 1)
            interval_feat = interval_feat.squeeze(-1)  # (batch_size, 64)
            interval_features_list.append(interval_feat)
        
        interval_features = torch.stack(interval_features_list, dim=1)  # (batch_size, 30, 64)
        
        # 128次元に拡張（conv特徴と結合）
        interval_features_conv_permuted = interval_features_conv.permute(0, 2, 1)  # (batch_size, 30, 128)
        interval_features = torch.cat([interval_features, interval_features_conv_permuted], dim=2)  # (batch_size, 30, 192)
        
        # 128次元に投影
        interval_features = interval_features[:, :, :self.interval_feature_dim]  # (batch_size, 30, 128)
        
        # ===== 4. アテンション機構: 周辺区間から情報を集約 =====
        # ノイズ区間をマスクして、周辺区間から情報を集約
        # (seq_len, batch_size, embed_dim) に変換（MultiheadAttention用）
        interval_features_transposed = interval_features.permute(1, 0, 2)  # (30, batch_size, 128)
        
        # アテンション: 各区間が他の区間から情報を集約
        attended_features, attention_weights = self.attention(
            interval_features_transposed,
            interval_features_transposed,
            interval_features_transposed
        )  # (30, batch_size, 128)
        
        attended_features = attended_features.permute(1, 0, 2)  # (batch_size, 30, 128)
        
        # ノイズ区間の特徴を周辺区間の情報で置き換え
        mask_expanded = mask.unsqueeze(-1)  # (batch_size, 30, 1)
        # ノイズ区間: アテンション後の特徴を使用
        # 正常区間: 元の特徴を使用（ノイズがないのでそのまま）
        reconstructed_interval_features = (
            mask_expanded * attended_features + 
            (1 - mask_expanded) * interval_features
        )
        
        # ===== 5. 予測した区間だけを復元（効率化） =====
        # 最も確信度の高い区間（予測区間）だけを復元
        predicted_intervals = mask.argmax(dim=1)  # (batch_size,) - 予測された区間
        
        # 予測した区間の特徴を取得して復元
        reconstructed_intervals_list = []
        for i in range(batch_size):
            predicted_interval = predicted_intervals[i].item()
            interval_feat = reconstructed_interval_features[i, predicted_interval, :]  # (128,)
            reconstructed_interval = self.interval_decoder(interval_feat.unsqueeze(0))  # (1, 100)
            reconstructed_intervals_list.append(reconstructed_interval.squeeze(0))  # (100,)
        
        # バッチごとに予測した区間の復元結果を結合
        reconstructed_intervals = torch.stack(reconstructed_intervals_list, dim=0)  # (batch_size, 100)
        
        # 最終的な平滑化（オプション）
        reconstructed_intervals = reconstructed_intervals.unsqueeze(1)  # (batch_size, 1, 100)
        reconstructed_intervals = self.final_reconstruction(reconstructed_intervals)  # (batch_size, 1, 100)
        reconstructed_intervals = reconstructed_intervals.squeeze(1)  # (batch_size, 100)
        
        return mask, reconstructed_intervals  # (batch_size, 30), (batch_size, 100)

