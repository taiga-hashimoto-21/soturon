# BERTとCLSトークン完全ガイド

## 1. BERTとは何か？

### 基本概念

**BERT (Bidirectional Encoder Representations from Transformers)** は、Googleが2018年に発表した自己教師あり学習モデルです。

### BERTの特徴

1. **双方向性（Bidirectional）**: 
   - 従来のモデルは左から右（または右から左）にしか読めなかった
   - BERTは**前後の文脈を同時に見る**ことができる

2. **Transformerベース**:
   - アテンション機構を使う
   - 長距離依存関係を捉えられる

3. **自己教師あり学習**:
   - ラベルなしデータで学習できる
   - 2つのタスクで事前学習：
     - **Masked Language Modeling (MLM)**: 一部をマスクして予測
     - **Next Sentence Prediction (NSP)**: 2つの文が連続しているか予測

---

## 2. CLSトークンとは？

### CLSトークンの役割

**CLS (Classification) トークン**は、BERTの入力の**最初に追加される特別なトークン**です。

### なぜ必要か？

#### 1. **全体の要約表現**
```
[CLS] トークン1 トークン2 トークン3 ... トークンN
 ↑
 このトークンが「全体の情報を集約した表現」になる
```

- CLSトークンは、**すべてのトークンからの情報を集約**する
- アテンション機構を通じて、全体の文脈を理解した表現になる

#### 2. **分類タスクの出力**
- 文全体を分類するタスクでは、CLSトークンの表現を使う
- 例: 感情分析、スパム判定など

#### 3. **固定長の表現**
- 入力の長さが変わっても、CLSトークンは常に1つ
- 分類器への入力として使いやすい

---

## 3. BERTの入力形式

### 標準的なBERTの入力

```
入力: "私は学生です"

トークン化後:
[CLS] 私 は 学生 です [SEP]

位置エンコーディング:
[CLS] の位置: 0
私 の位置: 1
は の位置: 2
...
```

### 各要素の説明

1. **[CLS]**: 分類用の特別なトークン（最初に追加）
2. **トークン**: 実際のデータ（単語や数値）
3. **[SEP]**: セパレータ（文の区切り、2文がある場合に使用）
4. **位置エンコーディング**: 各トークンの位置情報

---

## 4. この研究への適用方法

### PSDデータへの適用

#### 入力データの構造

```
元のPSDデータ: [3000ポイント]
  ↓
トークン化: 30区間に分割（各100ポイント）
  ↓
BERTへの入力:
[CLS] 区間1 区間2 区間3 ... 区間30
```

### 具体的な実装イメージ

```python
# 入力: PSDデータ (batch_size, 3000)
# 1. 30区間に分割
intervals = split_into_intervals(psd_data, num_intervals=30)  
# shape: (batch_size, 30, 100)

# 2. 各区間を埋め込みベクトルに変換
interval_embeddings = embed_intervals(intervals)  
# shape: (batch_size, 30, hidden_size)

# 3. CLSトークンを追加
cls_token = learnable_cls_token()  
# shape: (batch_size, 1, hidden_size)

# 4. BERTへの入力を作成
bert_input = torch.cat([cls_token, interval_embeddings], dim=1)  
# shape: (batch_size, 31, hidden_size)
#         [CLS] + 30区間 = 31トークン
```

---

## 5. CLSトークンの使い方

### パターン1: 分類タスク（従来のBERT）

```python
# BERTの出力
output = bert_model(bert_input)  
# shape: (batch_size, 31, hidden_size)

# CLSトークンの表現を取得（最初のトークン）
cls_output = output[:, 0, :]  
# shape: (batch_size, hidden_size)

# 分類器に入力
prediction = classifier(cls_output)  
# shape: (batch_size, num_classes)
```

### パターン2: アテンションウェイトの取得（この研究）

```python
# BERTの出力とアテンションウェイトを取得
output, attention_weights = bert_model(bert_input, return_attention=True)  
# attention_weights: (batch_size, num_heads, 31, 31)

# CLSトークンから各区間へのアテンションウェイト
cls_attention = attention_weights[:, :, 0, 1:]  
# shape: (batch_size, num_heads, 30)
# CLS(0) → 区間1-30(1-30)へのアテンション

# 平均を取る
cls_attention_mean = cls_attention.mean(dim=1)  
# shape: (batch_size, 30)

# これが「各区間の重要度」を表す
# ノイズ区間では、この値が低くなるように学習する
```

---

## 6. 自己教師あり学習タスクの設計

### タスク1: Masked Interval Modeling（推奨）

**アイデア**: PSDデータの一部の区間をマスクして、マスクされた区間を予測する

```python
# 1. ランダムに15%の区間をマスク
masked_intervals = mask_random_intervals(intervals, mask_ratio=0.15)

# 2. BERTに入力
output = bert_model(masked_intervals)

# 3. マスクされた区間だけを予測
masked_predictions = output[masked_positions]

# 4. 元の区間と比較して損失を計算
loss = mse_loss(masked_predictions, original_intervals[masked_positions])
```

### タスク2: Contrastive Learning（代替案）

**アイデア**: 同じデータから異なる変換を施したペアを作り、埋め込み表現を学習

```python
# 1. 同じPSDデータから2つの変換を作成
aug1 = augment_psd(psd_data)  # 例: ノイズ追加
aug2 = augment_psd(psd_data)  # 例: スケーリング

# 2. BERTで埋め込み表現を取得
emb1 = bert_model(aug1)[:, 0, :]  # CLSトークン
emb2 = bert_model(aug2)[:, 0, :]  # CLSトークン

# 3. コントラスト損失（同じデータなので近くに）
loss = contrastive_loss(emb1, emb2)
```

---

## 7. 損失関数の設計（この研究用）

### 総損失 = 自己教師あり学習損失 + 正則化項

```python
def compute_loss(model_output, masked_targets, attention_weights, noise_intervals, lambda_reg=0.1):
    # 1. マスク予測タスクの損失
    task_loss = mse_loss(model_output[masked_positions], masked_targets)
    
    # 2. 正則化項: ノイズ区間のアテンションウェイトを下げる
    # CLSトークンからノイズ区間へのアテンションウェイト
    cls_attention = attention_weights[:, :, 0, 1:]  # CLS → 各区間
    noise_attention = cls_attention[:, :, noise_intervals]  # ノイズ区間のみ
    
    # ノイズ区間のアテンションウェイトを下げる（平均を小さくする）
    regularization = lambda_reg * noise_attention.mean()
    
    total_loss = task_loss + regularization
    return total_loss
```

---

## 8. 実装の全体像

### モデル構造

```python
class BERTNoiseDetector(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, num_layers=6):
        super().__init__()
        
        # 1. 埋め込み層（各区間をベクトルに変換）
        self.interval_embedding = nn.Linear(100, hidden_size)
        
        # 2. CLSトークン（学習可能なパラメータ）
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # 3. 位置エンコーディング
        self.position_embedding = nn.Embedding(31, hidden_size)  # CLS + 30区間
        
        # 4. Transformerエンコーダー（BERT）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. マスク予測用のヘッド
        self.mask_predictor = nn.Linear(hidden_size, 100)  # 各区間のサイズ
    
    def forward(self, intervals, mask_positions=None):
        # intervals: (batch_size, 30, 100)
        batch_size = intervals.shape[0]
        
        # 1. 各区間を埋め込みベクトルに変換
        interval_emb = self.interval_embedding(intervals)  
        # (batch_size, 30, hidden_size)
        
        # 2. CLSトークンを追加
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  
        # (batch_size, 1, hidden_size)
        
        # 3. 結合
        x = torch.cat([cls_tokens, interval_emb], dim=1)  
        # (batch_size, 31, hidden_size)
        
        # 4. 位置エンコーディングを追加
        positions = torch.arange(31, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_embedding(positions)
        
        # 5. Transformerエンコーダーを通す
        output = self.transformer(x)  
        # (batch_size, 31, hidden_size)
        
        # 6. マスク予測（マスクされた区間だけ）
        if mask_positions is not None:
            masked_output = output[:, 1:, :][mask_positions]  
            # CLSを除く、マスクされた区間のみ
            predictions = self.mask_predictor(masked_output)  
            # (num_masked, 100)
        else:
            predictions = None
        
        # 7. アテンションウェイトを取得（後で実装）
        attention_weights = self.get_attention_weights(x)
        
        return predictions, output, attention_weights
    
    def get_attention_weights(self, x):
        # 簡易版: 実際にはTransformerEncoderLayerから取得する必要がある
        # ここでは概念的な実装
        pass
```

---

## 9. まとめ

### CLSトークンの役割（この研究での）

1. **全体の要約**: 30区間すべての情報を集約
2. **アテンションの起点**: CLSトークンから各区間へのアテンションウェイトを取得
3. **ノイズ検出**: ノイズ区間へのアテンションウェイトが低くなるように学習

### 実装のポイント

1. ✅ **CLSトークンを最初に追加**: `[CLS] + 30区間`
2. ✅ **マスク予測タスク**: 一部の区間をマスクして予測
3. ✅ **正則化項**: ノイズ区間のアテンションウェイトを下げる
4. ✅ **アテンションウェイトの取得**: CLSトークンから各区間へのアテンションを取得

### 次のステップ

1. BERTベースのモデルを実装
2. CLSトークンを追加
3. マスク予測タスクを実装
4. アテンションウェイトを取得してノイズ検出

---

## 10. よくある質問

### Q: CLSトークンは必須ですか？

**A**: この研究では**推奨**です。理由：
- 全体の要約表現として使える
- アテンションウェイトの起点として使いやすい
- ただし、CLSトークンなしでも実装可能（各区間の表現を直接使う）

### Q: CLSトークンの位置は固定ですか？

**A**: 通常は**最初**に配置しますが、最後でも可能です。ただし、最初が一般的です。

### Q: CLSトークンのサイズは？

**A**: 他のトークンと同じサイズ（`hidden_size`）です。学習可能なパラメータとして定義します。

