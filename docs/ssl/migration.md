# 自己教師あり学習への移行方針

## 結論：既存のevalとノイズの乗せ方を合わせるべき

**理由**：
1. **比較可能性**: 同じデータ、同じノイズパターンで比較することで、手法の違いだけを評価できる
2. **再現性**: 既存の実験設定を維持することで、再現性が保証される
3. **発表の説得力**: 同じ条件で比較することで、「自己教師あり学習の方が優れている」という主張が明確になる

---

## 既存の実装の確認

### 1. ノイズの付与方法
- **3種類のノイズパターン**:
  1. **周波数帯域集中ノイズ** (`frequency_band_noise.py`)
  2. **局所スパイクノイズ** (`localized_spike_noise.py`)
  3. **振幅依存ノイズ** (`amplitude_dependent_noise.py`)

- **データセット準備** (`prepare_baseline_dataset.py`):
  - 32,000件のデータを30区間（または10区間）に分割
  - ランダムに1つの区間にノイズを付与
  - ラベル: ノイズが付与された区間番号（0-29または0-9）

### 2. 評価指標 (`evaluation_metrics.md`)
- **ベースライン手法**: 30クラス分類のAccuracy, F1-score
- **自己教師あり学習**: アテンションウェイトベースの評価
  - 閾値ベースのAccuracy
  - ROC-AUC
  - F1-score（閾値最適化後）

---

## 自己教師あり学習での実装方針

### ✅ 維持すべき点

#### 1. **ノイズの付与方法**
- **同じノイズ生成関数を使用**: `noise/add_noise.py` の関数をそのまま使用
- **同じパラメータ**: `NOISE_LEVEL`, `NUM_INTERVALS` など
- **同じデータ分割**: 訓練:80%, 検証:10%, テスト:10%

#### 2. **評価指標**
- **閾値ベースのAccuracy**: ベースラインと直接比較可能
- **F1-score**: ベースラインと比較
- **ROC-AUC**: 自己教師あり学習の連続値の特性を活かした評価

#### 3. **データセット**
- **同じデータ**: `data_lowF_noise.pickle` を使用
- **同じ区間分割**: 30区間（または10区間）に分割

### 🔄 変更すべき点

#### 1. **ラベルの使用**
- **ベースライン**: ラベル（区間番号）を使って教師あり学習
- **自己教師あり学習**: **ラベルは使わない**（自己教師あり学習のため）
  - ただし、**評価時のみ**ラベルを使用して精度を計算

#### 2. **モデルアーキテクチャ**
- **ベースライン**: CNN（ResNet風の1D CNN）
- **自己教師あり学習**: Transformerベースのモデル
  - アテンション機構を使用
  - アテンションウェイトを取得可能にする

#### 3. **損失関数**
- **ベースライン**: CrossEntropyLoss（分類タスク）
- **自己教師あり学習**: 
  - 自己教師あり学習タスクの損失（例: マスク予測、コントラスト学習など）
  - **正則化項**: ノイズ区間のアテンションウェイトを下げる項

---

## 実装の流れ

### ステップ1: データセットの準備（既存コードを流用）

```python
# prepare_baseline_dataset.py を参考に
# ただし、ラベルは評価時のみ使用
from noise.add_noise import add_noise_to_interval

# ノイズを付与（ラベルは保存するが、学習時は使わない）
noisy_data, labels = prepare_dataset_with_noise(...)
```

### ステップ2: 自己教師あり学習タスクの設計

**推奨タスク**:
1. **マスク予測タスク**: PSDデータの一部をマスクして、マスクされた部分を予測
2. **コントラスト学習**: 同じデータから異なる変換を施したペアを作り、埋め込み表現を学習
3. **次文予測**: データの連続性を学習

### ステップ3: Transformerモデルの実装

```python
class TransformerNoiseDetector(nn.Module):
    def __init__(self, ...):
        # Transformerエンコーダー
        self.transformer = ...
        # アテンションウェイトを取得可能にする
    
    def forward(self, x):
        # 自己教師あり学習タスクの予測
        # アテンションウェイトを返す
        return prediction, attention_weights
```

### ステップ4: 損失関数の設計

```python
def compute_loss(prediction, target, attention_weights, noise_intervals, lambda_reg=0.1):
    # 1. 自己教師あり学習タスクの損失
    task_loss = self_supervised_loss(prediction, target)
    
    # 2. 正則化項: ノイズ区間のアテンションウェイトを下げる
    noise_attention = get_noise_interval_attention(attention_weights, noise_intervals)
    regularization = lambda_reg * noise_attention.mean()
    
    total_loss = task_loss + regularization
    return total_loss
```

### ステップ5: 評価（既存の評価関数を流用）

```python
# evaluate_noise_detection.py を参考に
# アテンションウェイトからノイズ区間を予測
def evaluate_self_supervised_model(model, dataloader, labels):
    # 1. アテンションウェイトを取得
    attention_weights = model.get_attention_weights(...)
    
    # 2. 区間ごとに平均化
    interval_attention = average_by_intervals(attention_weights, num_intervals=30)
    
    # 3. 閾値を最適化
    best_threshold = optimize_threshold(interval_attention, labels)
    
    # 4. 予測
    predictions = (interval_attention < best_threshold).argmax(dim=1)
    
    # 5. 評価（既存の評価関数を使用）
    accuracy = calculate_accuracy(predictions, labels)
    f1_score = calculate_f1_score(predictions, labels)
    roc_auc = calculate_roc_auc(interval_attention, labels)
    
    return accuracy, f1_score, roc_auc
```

---

## 発表での比較方法

### スライド構成案

1. **ベースライン手法の結果**
   - 30クラス分類のAccuracy: X%
   - F1-score: Y%
   - 損失曲線のグラフ

2. **自己教師あり学習の結果**
   - 閾値ベースのAccuracy: Z% (Z > X)
   - F1-score: W% (W > Y)
   - ROC-AUC: V
   - アテンションウェイトの可視化

3. **比較と考察**
   - 自己教師あり学習の方が精度が高い
   - アテンションウェイトがノイズ区間を正しく識別している
   - 連続値なので、より柔軟な分析が可能

---

## 実装時の注意点

### 1. **データの一貫性**
- ベースラインと自己教師あり学習で**同じデータセット**を使用
- 同じランダムシードを使用して再現性を保証

### 2. **評価の公平性**
- 同じ評価指標を使用
- 同じテストデータで評価

### 3. **ノイズパターンの統一**
- 3種類のノイズパターンすべてで評価
- 各ノイズパターンでの結果を比較

---

## 次のステップ

1. ✅ **既存コードの確認**（完了）
2. ⏳ **自己教師あり学習タスクの設計**
3. ⏳ **Transformerモデルの実装**
4. ⏳ **損失関数の実装**（正則化項を含む）
5. ⏳ **評価関数の実装**（既存の評価関数を流用）
6. ⏳ **実験と比較**

---

## まとめ

**既存のevalとノイズの乗せ方を合わせることで**：
- 公平な比較が可能
- 再現性が保証される
- 発表の説得力が増す

**変更点は**：
- モデルアーキテクチャ（CNN → Transformer）
- 損失関数（分類損失 → 自己教師あり学習損失 + 正則化項）
- ラベルの使用（学習時は使わない、評価時のみ使用）

この方針で実装を進めることをおすすめします！

