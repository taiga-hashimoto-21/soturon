# 自己教師あり学習タスク設計

## 概要

PSDデータからノイズ区間を検出するための自己教師あり学習タスクを設計する。  
主タスク（マスク予測）でデータの構造を学習し、副産物としてノイズ検出能力を獲得する。

---

## 推奨タスク: マスク予測 + 正則化項

### タスクの目的

1. **主タスク**: マスクされた区間を復元することで、PSDデータの構造を学習
2. **副産物**: ノイズ区間のアテンションウェイトが低くなることで、ノイズ検出能力を獲得

### タスクの詳細

#### ステップ1: データの準備

```
入力: PSDデータ (3000ポイント)
  ↓
30区間に分割: [区間1, 区間2, ..., 区間30]
各区間: 100ポイント (3000 / 30 = 100)
```

#### ステップ2: マスクの作成

- **マスク率**: 15%の区間をランダムにマスク
- **マスク方法**: 
  - マスクされた区間を0で置き換える（または平均値で置き換える）
  - マスク位置を記録しておく

```
元のデータ: [区間1, 区間2, 区間3, ..., 区間30]
  ↓ 15%をマスク（例: 区間5, 区間12, 区間20, 区間25）
マスク後: [区間1, 区間2, 区間3, 区間4, [MASK], 区間6, ..., [MASK], ..., 区間30]
```

#### ステップ3: CLSトークンの追加

```
[CLS, 区間1, 区間2, ..., [MASK], ..., 区間30]
 ↑
 全体のまとめトークン
```

#### ステップ4: BERTで処理

- Transformerエンコーダーで31トークン（CLS + 30区間）を処理
- マスクされた区間の位置を予測
- アテンションウェイトを取得

#### ステップ5: 損失の計算

```python
総損失 = マスク予測損失 + 正則化項

# 1. マスク予測損失
mask_loss = MSE(予測された区間, 元の区間)

# 2. 正則化項（ノイズ区間のアテンションを下げる）
noise_attention = CLSトークン → ノイズ区間へのアテンション
regularization = λ × noise_attention.mean()

# 総損失
total_loss = mask_loss + regularization
```

---

## タスクの選択肢

### タスク1: マスク予測（復元タスク）⭐推奨

**概要**: 30区間に分割し、ランダムに区間をマスクして復元させる

**詳細**:
- 30区間に分割
- 15%の区間をランダムにマスク
- マスクされた区間を予測して復元

**メリット**:
- ✅ BERTのマスク予測と同じで実装しやすい
- ✅ データの構造を学習できる
- ✅ ノイズ検出との関連が強い（正則化項を追加可能）

**デメリット**:
- ⚠️ ノイズ検出は副産物なので、直接的な学習ではない

**実装のイメージ**:
```python
# マスクを作成
masked_intervals, mask_positions = mask_random_intervals(intervals, mask_ratio=0.15)

# BERTで処理
predictions, output, attention_weights = model(masked_intervals, mask_positions)

# 損失を計算
mask_loss = mse_loss(predictions, original_intervals[mask_positions])
noise_reg = lambda_reg * attention_weights[CLS, noise_intervals].mean()
total_loss = mask_loss + noise_reg
```

---

### タスク2: 区間の順序判定（入れ替えタスク）

**概要**: 2つの区間を入れ替えて、「正しい順序か、入れ替わっているか」を判定

**詳細**:
- 30区間に分割
- ランダムに2つの区間を入れ替える
- 「正しい順序か、入れ替わっているか」を2値分類

**メリット**:
- ✅ データの連続性を学習できる
- ✅ 実装が比較的簡単

**デメリット**:
- ❌ ノイズ検出との関連が弱い
- ❌ ノイズ検出は副産物として得られにくい

**実装のイメージ**:
```python
# 区間を入れ替え
swapped_intervals, swap_positions = swap_random_intervals(intervals)

# BERTで処理
output = model(swapped_intervals)
cls_output = output[:, 0, :]  # CLSトークン

# 2値分類
is_swapped = classifier(cls_output)  # 0 or 1
loss = binary_cross_entropy(is_swapped, label)
```

---

### タスク3: コントラスト学習

**概要**: 同じPSDデータから2つのバージョンを作り、「同じデータから来たか」を判定

**詳細**:
- 同じPSDデータから2つのバージョンを作成
  - バージョン1: ノイズなし（または軽いノイズ）
  - バージョン2: ノイズあり（または重いノイズ）
- 「同じデータから来たか、違うデータから来たか」を判定

**メリット**:
- ✅ ノイズ検出と直接関連
- ✅ ノイズの特徴を学習しやすい

**デメリット**:
- ❌ 実装がやや複雑
- ❌ データ拡張の設計が必要

**実装のイメージ**:
```python
# 同じデータから2つのバージョンを作成
version1 = original_data  # ノイズなし
version2 = add_noise(original_data)  # ノイズあり

# BERTで処理
emb1 = model(version1)[:, 0, :]  # CLSトークン
emb2 = model(version2)[:, 0, :]  # CLSトークン

# コントラスト損失
loss = contrastive_loss(emb1, emb2, label=1)  # 同じデータなので1
```

---

### タスク4: マスク予測 + 正則化項 ⭐⭐最推奨

**概要**: マスク予測タスクに、ノイズ区間のアテンションを下げる正則化項を追加

**詳細**:
- タスク1（マスク予測）を基本とする
- 追加で、ノイズ区間のアテンションウェイトを下げる正則化項を追加

**メリット**:
- ✅ マスク予測でデータ構造を学習
- ✅ 正則化項でノイズ検出も同時に学習
- ✅ 先輩のアドバイスに合致（副産物としてノイズ検出）
- ✅ 実装しやすい

**デメリット**:
- ⚠️ 正則化項の重み（λ）の調整が必要

**実装のイメージ**:
```python
# マスクを作成
masked_intervals, mask_positions = mask_random_intervals(intervals, mask_ratio=0.15)

# BERTで処理
predictions, output, attention_weights = model(
    masked_intervals, 
    mask_positions=mask_positions,
    return_attention=True
)

# 1. マスク予測損失
mask_loss = mse_loss(predictions, original_intervals[mask_positions])

# 2. 正則化項: ノイズ区間のアテンションを下げる
cls_attention = attention_weights[:, :, 0, 1:]  # CLS → 各区間
noise_attention = cls_attention[:, :, noise_intervals]  # ノイズ区間のみ
regularization = lambda_reg * noise_attention.mean()

# 総損失
total_loss = mask_loss + regularization
```

---

## 各タスクの簡易実装例

### タスク1: マスク予測（復元タスク）

```python
def task1_mask_prediction(intervals, model):
    """
    タスク1: マスク予測
    """
    batch_size, num_intervals, interval_size = intervals.shape
    
    # 1. 15%の区間をランダムにマスク
    mask_ratio = 0.15
    num_masked = int(num_intervals * mask_ratio)
    
    mask_positions = torch.zeros(batch_size, num_intervals, dtype=torch.bool)
    for i in range(batch_size):
        masked_indices = torch.randperm(num_intervals)[:num_masked]
        mask_positions[i, masked_indices] = True
    
    # 2. マスクされた区間を0で置き換え
    masked_intervals = intervals.clone()
    masked_intervals[mask_positions] = 0.0
    
    # 3. CLSトークンを追加してBERTで処理
    predictions, output, attention_weights = model(
        masked_intervals,
        mask_positions=mask_positions,
        return_attention=True
    )
    
    # 4. 損失を計算（マスク予測のみ）
    masked_original = intervals[mask_positions]
    loss = F.mse_loss(predictions, masked_original)
    
    return loss, predictions, attention_weights
```

---

### タスク2: 区間の順序判定（入れ替えタスク）

```python
def task2_order_detection(intervals, model):
    """
    タスク2: 区間の順序判定
    """
    batch_size, num_intervals, interval_size = intervals.shape
    
    # 1. ランダムに2つの区間を入れ替え
    swapped_intervals = intervals.clone()
    labels = torch.zeros(batch_size, dtype=torch.long)  # 0: 正しい順序, 1: 入れ替え
    
    for i in range(batch_size):
        # 50%の確率で入れ替える
        if torch.rand(1) > 0.5:
            idx1, idx2 = torch.randperm(num_intervals)[:2]
            swapped_intervals[i, idx1], swapped_intervals[i, idx2] = \
                swapped_intervals[i, idx2].clone(), swapped_intervals[i, idx1].clone()
            labels[i] = 1
    
    # 2. CLSトークンを追加してBERTで処理
    output, attention_weights = model(swapped_intervals, return_attention=True)
    
    # 3. CLSトークンの表現を取得
    cls_output = output[:, 0, :]  # (batch_size, hidden_size)
    
    # 4. 2値分類
    classifier = nn.Linear(hidden_size, 2)
    logits = classifier(cls_output)
    
    # 5. 損失を計算
    loss = F.cross_entropy(logits, labels)
    
    return loss, logits, attention_weights
```

---

### タスク3: コントラスト学習

```python
def task3_contrastive_learning(intervals, model, noise_intervals):
    """
    タスク3: コントラスト学習
    """
    batch_size, num_intervals, interval_size = intervals.shape
    
    # 1. 同じデータから2つのバージョンを作成
    version1 = intervals.clone()  # ノイズなし（または軽いノイズ）
    
    # バージョン2: ノイズを追加（既にノイズが含まれている場合はそのまま）
    version2 = intervals.clone()
    # ノイズ区間に追加のノイズを加える（オプション）
    for i in range(batch_size):
        noise_idx = noise_intervals[i].item()
        # 軽いノイズを追加
        version2[i, noise_idx] += torch.randn(interval_size) * 0.1
    
    # 2. CLSトークンを追加してBERTで処理
    output1, attention_weights1 = model(version1, return_attention=True)
    output2, attention_weights2 = model(version2, return_attention=True)
    
    # 3. CLSトークンの表現を取得
    emb1 = output1[:, 0, :]  # (batch_size, hidden_size)
    emb2 = output2[:, 0, :]  # (batch_size, hidden_size)
    
    # 4. コントラスト損失を計算
    # 同じデータなので、埋め込み表現が近くなるように学習
    # コサイン類似度を使う
    emb1_norm = F.normalize(emb1, p=2, dim=1)
    emb2_norm = F.normalize(emb2, p=2, dim=1)
    similarity = (emb1_norm * emb2_norm).sum(dim=1)  # コサイン類似度
    
    # 同じデータなので、類似度を最大化（損失を最小化）
    loss = 1 - similarity.mean()
    
    return loss, emb1, emb2, attention_weights1
```

---

### タスク4: マスク予測 + 正則化項 ⭐⭐最推奨

```python
def task4_mask_prediction_with_regularization(intervals, model, noise_intervals, lambda_reg=0.1):
    """
    タスク4: マスク予測 + 正則化項
    """
    batch_size, num_intervals, interval_size = intervals.shape
    
    # 1. 15%の区間をランダムにマスク（タスク1と同じ）
    mask_ratio = 0.15
    num_masked = int(num_intervals * mask_ratio)
    
    mask_positions = torch.zeros(batch_size, num_intervals, dtype=torch.bool)
    for i in range(batch_size):
        masked_indices = torch.randperm(num_intervals)[:num_masked]
        mask_positions[i, masked_indices] = True
    
    masked_intervals = intervals.clone()
    masked_intervals[mask_positions] = 0.0
    
    # 2. CLSトークンを追加してBERTで処理
    predictions, output, attention_weights = model(
        masked_intervals,
        mask_positions=mask_positions,
        return_attention=True
    )
    
    # 3. マスク予測損失
    masked_original = intervals[mask_positions]
    mask_loss = F.mse_loss(predictions, masked_original)
    
    # 4. 正則化項: ノイズ区間のアテンションを下げる
    # CLSトークンから各区間へのアテンション
    cls_attention = attention_weights[:, :, 0, 1:]  # (batch_size, num_heads, 30)
    
    # ノイズ区間のアテンションを取得
    noise_attention_list = []
    for i in range(batch_size):
        noise_idx = noise_intervals[i].item()
        noise_attention_list.append(cls_attention[i, :, noise_idx])
    
    noise_attention = torch.stack(noise_attention_list)  # (batch_size, num_heads)
    reg_loss = lambda_reg * noise_attention.mean()
    
    # 5. 総損失
    total_loss = mask_loss + reg_loss
    
    return total_loss, mask_loss, reg_loss, predictions, attention_weights
```

---

## 推奨タスクの詳細実装

### データの前処理

```python
def prepare_data(psd_data, num_intervals=30):
    """
    PSDデータを30区間に分割
    
    Args:
        psd_data: (batch_size, 3000)
    
    Returns:
        intervals: (batch_size, 30, 100)
    """
    batch_size = psd_data.shape[0]
    points_per_interval = 3000 // num_intervals  # 100
    
    intervals = psd_data.view(batch_size, num_intervals, points_per_interval)
    return intervals
```

### マスクの作成

```python
def create_mask(intervals, mask_ratio=0.15):
    """
    ランダムに区間をマスク
    
    Args:
        intervals: (batch_size, 30, 100)
        mask_ratio: マスクする区間の割合（デフォルト: 15%）
    
    Returns:
        masked_intervals: マスクされた区間
        mask_positions: マスク位置 (batch_size, 30) [True/False]
        original_intervals: 元の区間（損失計算用）
    """
    batch_size, num_intervals, interval_size = intervals.shape
    num_masked = int(num_intervals * mask_ratio)
    
    # マスク位置を決定
    mask_positions = torch.zeros(batch_size, num_intervals, dtype=torch.bool)
    for i in range(batch_size):
        masked_indices = torch.randperm(num_intervals)[:num_masked]
        mask_positions[i, masked_indices] = True
    
    # マスクされた区間を0で置き換え（または平均値で置き換え）
    masked_intervals = intervals.clone()
    masked_intervals[mask_positions] = 0.0  # または intervals.mean()
    
    return masked_intervals, mask_positions, intervals
```

### 損失関数

```python
def compute_loss(
    predictions,           # マスク予測結果 (num_masked_total, 100)
    original_intervals,   # 元の区間 (batch_size, 30, 100)
    mask_positions,       # マスク位置 (batch_size, 30)
    attention_weights,    # アテンションウェイト (batch_size, num_heads, 31, 31)
    noise_intervals,      # ノイズ区間のインデックス (batch_size,)
    lambda_reg=0.1        # 正則化項の重み
):
    """
    損失関数を計算
    
    Returns:
        total_loss: 総損失
        mask_loss: マスク予測損失
        reg_loss: 正則化損失
    """
    # 1. マスク予測損失
    masked_original = original_intervals[mask_positions]
    mask_loss = F.mse_loss(predictions, masked_original)
    
    # 2. 正則化項: ノイズ区間のアテンションを下げる
    batch_size = attention_weights.shape[0]
    cls_attention = attention_weights[:, :, 0, 1:]  # CLS → 各区間
    
    noise_attention_list = []
    for i in range(batch_size):
        noise_idx = noise_intervals[i].item()
        noise_attention_list.append(cls_attention[i, :, noise_idx])
    
    if len(noise_attention_list) > 0:
        noise_attention = torch.stack(noise_attention_list)
        reg_loss = lambda_reg * noise_attention.mean()
    else:
        reg_loss = torch.tensor(0.0, device=predictions.device)
    
    total_loss = mask_loss + reg_loss
    
    return total_loss, mask_loss, reg_loss
```

---

## 学習の流れ

### 1エポックの処理

```python
def train_epoch(model, dataloader, optimizer, device, lambda_reg=0.1):
    model.train()
    total_loss = 0
    total_mask_loss = 0
    total_reg_loss = 0
    
    for batch_idx, (psd_data, noise_intervals) in enumerate(dataloader):
        # 1. データを30区間に分割
        intervals = prepare_data(psd_data)  # (batch_size, 30, 100)
        
        # 2. マスクを作成
        masked_intervals, mask_positions, original_intervals = create_mask(
            intervals, mask_ratio=0.15
        )
        
        # 3. BERTで処理
        predictions, output, attention_weights = model(
            masked_intervals,
            mask_positions=mask_positions,
            return_attention=True
        )
        
        # 4. 損失を計算
        total_loss_batch, mask_loss, reg_loss = compute_loss(
            predictions,
            original_intervals,
            mask_positions,
            attention_weights,
            noise_intervals,
            lambda_reg=lambda_reg
        )
        
        # 5. 逆伝播
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        
        # 統計
        total_loss += total_loss_batch.item()
        total_mask_loss += mask_loss.item()
        total_reg_loss += reg_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_mask_loss = total_mask_loss / len(dataloader)
    avg_reg_loss = total_reg_loss / len(dataloader)
    
    return avg_loss, avg_mask_loss, avg_reg_loss
```

---

## ハイパーパラメータ

### 推奨値

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `num_intervals` | 30 | 区間数 |
| `mask_ratio` | 0.15 | マスクする区間の割合（15%） |
| `lambda_reg` | 0.1 | 正則化項の重み（調整が必要） |
| `hidden_size` | 256 | 埋め込み次元 |
| `num_heads` | 8 | アテンションヘッド数 |
| `num_layers` | 4-6 | Transformerレイヤー数 |
| `dropout` | 0.1 | Dropout率 |

### 調整が必要なパラメータ

1. **`lambda_reg`（正則化項の重み）**
   - 小さすぎる: ノイズ検出がうまくいかない
   - 大きすぎる: マスク予測タスクがうまくいかない
   - **推奨**: 0.05 ~ 0.2の範囲で調整

2. **`mask_ratio`（マスク率）**
   - 小さすぎる: 学習が不十分
   - 大きすぎる: 復元が困難
   - **推奨**: 0.1 ~ 0.2の範囲

---

## 評価方法

### ノイズ検出の評価

```python
def evaluate_noise_detection(model, dataloader, device):
    """
    ノイズ検出の精度を評価
    """
    model.eval()
    all_attention_weights = []
    all_labels = []
    
    with torch.no_grad():
        for psd_data, noise_intervals in dataloader:
            # データを30区間に分割
            intervals = prepare_data(psd_data)
            
            # マスクを作成（評価時はマスクしない、または固定マスク）
            # ここではマスクなしで評価
            _, output, attention_weights = model(
                intervals,
                mask_positions=None,
                return_attention=True
            )
            
            # CLSトークンから各区間へのアテンションを取得
            cls_attention = attention_weights[:, :, 0, 1:]  # (batch_size, num_heads, 30)
            cls_attention_mean = cls_attention.mean(dim=1)  # (batch_size, 30)
            
            all_attention_weights.append(cls_attention_mean.cpu())
            all_labels.append(noise_intervals.cpu())
    
    # 評価指標を計算
    all_attention = torch.cat(all_attention_weights, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 閾値を最適化して精度を計算
    accuracy, f1_score, roc_auc = calculate_metrics(all_attention, all_labels)
    
    return accuracy, f1_score, roc_auc
```

---

## まとめ

### 推奨タスク

**マスク予測 + 正則化項**

- **主タスク**: マスクされた区間を復元
- **副産物**: ノイズ区間のアテンションが低くなる
- **メリット**: データ構造を学習しつつ、ノイズ検出も同時に学習

### 実装のポイント

1. ✅ 30区間に分割
2. ✅ 15%の区間をランダムにマスク
3. ✅ CLSトークンを追加
4. ✅ BERTで処理してマスク予測
5. ✅ ノイズ区間のアテンションを下げる正則化項を追加

### 次のステップ

1. モデルの実装（BERTベース）
2. データローダーの実装
3. 学習ループの実装
4. 評価関数の実装
5. ハイパーパラメータの調整

