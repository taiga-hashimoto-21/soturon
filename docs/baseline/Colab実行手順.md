# Colabでの実行手順（畳み込みモデル - 周波数帯域集中ノイズ）

## 📋 概要

畳み込みモデルで周波数帯域集中ノイズを使用して学習し、損失を確認する手順です。

## 🔧 必要なファイル

以下のファイル/フォルダを準備してください：

1. **data_lowF_noise.pickle** - 元データ（PSD理論値データ）
2. **ノイズの付与(共通)/** フォルダ全体
   - `add_noise.py`
   - `frequency_band_noise.py`
   - `localized_spike_noise.py`
   - `amplitude_dependent_noise.py`
   - `__init__.py`
3. **畳み込み/dataset.py** - データセット準備スクリプト
4. **畳み込み/model.py** - モデル定義
5. **eval.py** - 評価スクリプト（ルートディレクトリ）

## 📝 実行手順

### ステップ1: Google Colabのセットアップ

1. Google Colabで新しいノートブックを作成
2. **GPUを有効化**:
   - メニュー: `ランタイム` → `ランタイムのタイプを変更`
   - ハードウェアアクセラレータ: **GPU（T4）** を選択

### ステップ2: ファイルのアップロード

#### 方法1: Google Driveを使用（推奨）

1. Google Driveにプロジェクトフォルダをアップロード
   - フォルダ構造を保持したままアップロード
   - 例: `/content/drive/MyDrive/noise/`

2. ColabでDriveをマウント:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. 作業ディレクトリを変更:
```python
import os
os.chdir('/content/drive/MyDrive/noise')
```

#### 方法2: Colabのファイルブラウザから直接アップロード

1. 左側のファイルブラウザ（📁アイコン）を開く
2. `/content/` フォルダに必要なファイルをドラッグ&ドロップ
   - フォルダ構造を保持する必要があるため、方法1を推奨

### ステップ3: ライブラリのインストール

```python
!pip install torch torchvision scikit-learn matplotlib -q
```

### ステップ4: データセットの準備

```python
# データセット準備スクリプトを実行
exec(open('畳み込み/dataset.py').read())
```

または、直接実行:
```python
import sys
sys.path.insert(0, '.')
from 畳み込み.dataset import *

# データセットが生成される
# baseline_dataset.pickle が作成される
```

**確認事項:**
- ノイズタイプ: `frequency_band`（周波数帯域集中ノイズ）
- 区間数: `30区間`
- ノイズレベル: `0.3`（30%）

### ステップ5: モデルのインポート

```python
import sys
sys.path.insert(0, '畳み込み')
from model import SimpleResNet1D

# 30クラス分類モデルを作成
model = SimpleResNet1D(num_classes=30).to(device)
```

### ステップ6: 学習の実行

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle

# データセットの読み込み
with open('baseline_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

train_data = dataset['train']['data']
train_labels = dataset['train']['labels']
val_data = dataset['val']['data']
val_labels = dataset['val']['labels']

# DataLoaderの作成
class PSDDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_dataset = PSDDataset(train_data, train_labels)
val_dataset = PSDDataset(val_data, val_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # 訓練
    model.train()
    train_loss = 0.0
    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # 検証
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.6f}")
    print(f"  Val Loss: {val_loss:.6f}")
```

### ステップ7: 評価

```python
import sys
sys.path.insert(0, '.')
from eval import evaluate_baseline_model

# 評価
results = evaluate_baseline_model(model, val_loader, device='cuda', num_intervals=30)

print("評価結果:")
print(f"  Accuracy: {results['accuracy']:.4f}")
print(f"  Precision: {results['precision']:.4f}")
print(f"  Recall: {results['recall']:.4f}")
print(f"  F1-score: {results['f1_score']:.4f}")
print(f"  Loss (CrossEntropyLoss): {results['loss']:.6f}")
```

## 📊 損失の確認

学習中に以下の損失が表示されます：

- **Train Loss**: 訓練データでの損失（CrossEntropyLoss）
- **Val Loss**: 検証データでの損失（CrossEntropyLoss）

各エポックで損失が減少することを確認してください。

## ⚠️ 注意事項

1. **ノイズタイプの確認**: `畳み込み/dataset.py` の `NOISE_TYPE = 'frequency_band'` を確認
2. **区間数の確認**: `NUM_INTERVALS = 30` を確認（30クラス分類）
3. **モデルのクラス数**: `num_classes=30` を確認

## 🔍 トラブルシューティング

### ファイルが見つからない場合

```python
import os
print("現在のディレクトリ:", os.getcwd())
print("ファイル一覧:", os.listdir('.'))
```

### インポートエラーの場合

```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/noise')  # プロジェクトルートを追加
```

### GPUが使えない場合

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")
```

