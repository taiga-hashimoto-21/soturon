# Google Colabでの実行手順（畳み込みモデル - soturonフォルダ）

## 📋 概要

`soturon`フォルダをGoogle Driveにアップロードして、Google Colabで畳み込みモデルを学習する手順です。

## 🔧 事前準備

### 1. Google Driveにフォルダをアップロード

1. **ローカルの`soturon`フォルダ全体をGoogle Driveにアップロード**
   - Google Driveを開く
   - 新しいフォルダを作成（例: `soturon`）
   - ローカルの`soturon`フォルダ内の**すべてのファイルとフォルダ**をアップロード
   - フォルダ構造を保持したままアップロードすることが重要

**必要なファイル/フォルダ**:
- `data_lowF_noise.pickle` - 元データ（PSD理論値データ）
- `ノイズの付与(共通)/` フォルダ全体
- `畳み込み/` フォルダ全体
  - `dataset.py`
  - `model.py`
  - `train_colab.ipynb`
- `eval.py` - 評価スクリプト（ルートディレクトリ）

### 2. Google Driveのパスを確認

アップロード後、Google Drive内のパスを確認してください。
例: `/content/drive/MyDrive/soturon/`

---

## 📝 実行手順

### ステップ1: Google Colabのセットアップ

1. **Google Colabで新しいノートブックを作成**
   - [Google Colab](https://colab.research.google.com/) を開く
   - 「ファイル」→「ノートブックを新規作成」

2. **GPUを有効化**
   - メニュー: `ランタイム` → `ランタイムのタイプを変更`
   - ハードウェアアクセラレータ: **GPU（T4）** を選択
   - 「保存」をクリック

### ステップ2: Google Driveをマウント

**セル1**: Google Driveをマウントして作業ディレクトリを変更

```python
# Google Driveをマウント
from google.colab import drive
drive.mount('/content/drive')

# 作業ディレクトリを変更（soturonフォルダに移動）
import os
os.chdir('/content/drive/MyDrive/soturon')  # あなたのGoogle Driveのパスに合わせて変更してください

print(f"現在の作業ディレクトリ: {os.getcwd()}")
print(f"ファイル一覧: {os.listdir('.')[:10]}")  # 最初の10ファイルを表示
```

**確認事項**:
- `data_lowF_noise.pickle` が存在するか確認
- `畳み込み/` フォルダが存在するか確認
- `ノイズの付与(共通)/` フォルダが存在するか確認

### ステップ3: 既存のノートブックを使用（推奨）

**方法A: 既存のノートブックを開く（推奨）**

1. Google Driveで `soturon/畳み込み/train_colab.ipynb` を右クリック
2. 「アプリで開く」→「Google Colab」を選択
3. ノートブックが開いたら、セルを順番に実行

**方法B: 新しいノートブックで実行**

既存の `train_colab.ipynb` の内容をコピーして新しいノートブックに貼り付けて実行

### ステップ4: ノートブックの実行

`train_colab.ipynb` のセルを順番に実行してください：

#### セル0: 説明セル（実行不要）
ノートブックの説明が書かれています。

#### セル1: Google Driveのマウント（既に実行済みの場合はスキップ）
```python
# Google Driveをマウント（推奨）
from google.colab import drive
drive.mount('/content/drive')

# 作業ディレクトリを変更（プロジェクトフォルダに移動）
import os
os.chdir('/content/drive/MyDrive/soturon')  # パスを確認して変更してください

print(f"現在の作業ディレクトリ: {os.getcwd()}")
print(f"ファイル一覧: {os.listdir('.')[:10]}")  # 最初の10ファイルを表示
```

#### セル2-3: ライブラリのインストールとインポート
```python
# ライブラリのインストールとインポート
!pip install torch torchvision scikit-learn matplotlib -q

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import platform
import os
import sys
import time
import warnings

# GPUの確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")
if torch.cuda.is_available():
    print(f"GPU名: {torch.cuda.get_device_name(0)}")
```

#### セル4: データセットの準備
このセルで `baseline_dataset.pickle` が生成されます。

**重要な設定**:
- `NUM_INTERVALS = 30` - 30クラス分類
- `NOISE_TYPE = 'frequency_band'` - 周波数帯域集中ノイズ
- `NOISE_LEVEL = 0.3` - ノイズレベル30%
- `ADD_STRUCTURED_NOISE = True` - 構造化ノイズを追加

**実行時間**: 約5-10分（32000サンプルの処理）

#### セル5-10: モデルとデータローダーの準備
- モデルのインポート
- データセットの読み込み
- DataLoaderの作成
- モデルの初期化

#### セル11-12: 学習の実行
- エポック数: 10エポック（デフォルト）
- バッチサイズ: 64
- 学習率: 0.001
- Early Stopping: patience=10

**実行時間**: GPU使用時で約10-30分（エポック数による）

#### セル13-16: 評価と可視化
- テストデータでの評価
- 学習曲線の可視化
- モデルの保存

#### セル17-18: 結果ファイルのダウンロード（オプション）
学習済みモデルや結果ファイルをダウンロードできます。

---

## 📊 現在の設定

### データセット設定
- **区間数**: 30区間（30クラス分類）
- **ノイズタイプ**: `frequency_band`（周波数帯域集中ノイズ）
- **ノイズレベル**: 0.3（30%）
- **構造化ノイズ**: 有効

### モデル設定
- **モデル**: `SimpleCNN`（内部で`SimpleResNet1D`を使用）
- **クラス数**: 30クラス
- **入力サイズ**: (batch_size, 3000)

### 学習設定
- **バッチサイズ**: 64
- **学習率**: 0.001
- **エポック数**: 10
- **オプティマイザ**: Adam
- **損失関数**: CrossEntropyLoss（クラス重み付き）
- **スケジューラ**: ReduceLROnPlateau

---

## ⚠️ 注意事項

### 1. パスの確認
Google Driveのパスが正しいか確認してください：
```python
os.chdir('/content/drive/MyDrive/soturon')  # あなたのパスに合わせて変更
```

### 2. ファイルの存在確認
以下のファイルが存在するか確認：
```python
import os
print("data_lowF_noise.pickle:", os.path.exists('data_lowF_noise.pickle'))
print("畳み込みフォルダ:", os.path.exists('畳み込み'))
print("ノイズの付与(共通)フォルダ:", os.path.exists('ノイズの付与(共通)'))
```

### 3. ノイズタイプの確認
`畳み込み/dataset.py` の設定を確認：
- `NUM_INTERVALS = 30`
- `NOISE_TYPE = 'frequency_band'`
- `NOISE_LEVEL = 0.3`

### 4. GPUの確認
GPUが使用可能か確認：
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 🔍 トラブルシューティング

### エラー1: ファイルが見つからない

**症状**: `FileNotFoundError: data_lowF_noise.pickle`

**解決方法**:
1. Google Driveのパスが正しいか確認
2. ファイルが正しくアップロードされているか確認
3. 作業ディレクトリを確認：
```python
import os
print(os.getcwd())
print(os.listdir('.'))
```

### エラー2: モジュールのインポートエラー

**症状**: `ModuleNotFoundError: No module named 'noise'`

**解決方法**:
セル4でシンボリックリンクが作成されているか確認。作成されない場合は、手動でパスを追加：
```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/soturon')
sys.path.insert(0, '/content/drive/MyDrive/soturon/ノイズの付与(共通)')
```

### エラー3: GPUが使えない

**症状**: CPUで実行されている

**解決方法**:
1. ランタイムのタイプを確認（GPUが選択されているか）
2. ランタイムを再起動
3. GPUの使用状況を確認：
```python
!nvidia-smi
```

### エラー4: メモリ不足

**症状**: `RuntimeError: CUDA out of memory`

**解決方法**:
1. バッチサイズを小さくする（64 → 32）
2. ランタイムを再起動してメモリをクリア
3. 不要な変数を削除：
```python
del variable_name
torch.cuda.empty_cache()
```

---

## 📈 実行結果の確認

### 学習中の出力
各エポックで以下の情報が表示されます：
- Train Loss / Train Acc
- Val Loss / Val Acc
- 学習率
- 経過時間

### 最終結果
学習完了後、以下のファイルが生成されます：
- `baseline_dataset.pickle` - データセット
- `baseline_model.pth` - 学習済みモデル
- `baseline_training_curves.png` - 学習曲線
- `prediction_results.csv` - 予測結果
- `confusion_matrix.csv` - 混同行列

### 評価指標
- **Accuracy**: 精度
- **Precision**: 適合率
- **Recall**: 再現率
- **F1-score**: F1スコア
- **Loss**: CrossEntropyLoss

---

## 🎯 次のステップ

1. **学習結果の確認**
   - 学習曲線を確認して過学習がないかチェック
   - 混同行列を確認してどの区間を間違えやすいか確認

2. **モデルの改善**
   - ハイパーパラメータの調整
   - モデルアーキテクチャの変更（`ResNet1D`、`ImprovedCNN`など）

3. **他のノイズタイプでの実験**
   - `NOISE_TYPE`を変更して再学習
   - `localized_spike`（局所スパイクノイズ）
   - `amplitude_dependent`（振幅依存ノイズ）

---

## 📚 関連ドキュメント

- `docs/baseline/Colab実行手順.md` - 基本的な実行手順
- `docs/baseline/evaluation_metrics.md` - 評価指標の詳細
- `docs/目的/研究の目的と全体の流れ.md` - 研究の全体像

---

## 💡 ヒント

1. **長時間の学習**: 長時間の学習が必要な場合は、Colab Proを検討してください（長時間のGPU使用が可能）

2. **結果の保存**: 重要な結果はGoogle Driveに保存されるので、Colabセッションが終了してもデータは保持されます

3. **再実行**: データセットは一度生成すれば再利用可能です。`baseline_dataset.pickle`が存在する場合は、セル4をスキップできます

4. **バージョン管理**: 重要な変更はGitで管理することをおすすめします

