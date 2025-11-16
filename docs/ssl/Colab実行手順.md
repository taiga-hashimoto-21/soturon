# Google Colabでの実行手順

## 準備

### 1. ファイルの準備

以下のファイルをGoogle Driveにアップロードしてください：

```
Google Drive/
└── MyDrive/
    ├── data_lowF_noise.pickle          # データファイル
    └── noise/                           # プロジェクトフォルダ（オプション）
        ├── task/
        │   ├── __init__.py
        │   ├── dataset.py
        │   ├── model.py
        │   └── train.py
        └── eval.py
```

### 2. Google Colabでノートブックを作成

1. Google Colabを開く: https://colab.research.google.com/
2. 新しいノートブックを作成
3. `task4_colab.ipynb`の内容をコピー＆ペースト

---

## 実行手順

### ステップ1: 環境設定

**セル1: Google Driveをマウント**
```python
from google.colab import drive
drive.mount('/content/drive')
```
- 実行すると認証画面が表示されるので、指示に従って認証

**セル2: ライブラリをインストール**
```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install scikit-learn matplotlib
```
- PyTorchとその他の必要なライブラリをインストール

**セル3: 作業ディレクトリを設定**
```python
import os
import sys

project_path = '/content/drive/MyDrive/noise'  # 自分のパスに合わせて変更
if os.path.exists(project_path):
    sys.path.append(project_path)
    os.chdir(project_path)
    print(f"Working directory: {os.getcwd()}")
```
- コードをアップロードした場合のみ必要
- コードを直接Colabにコピーした場合は不要

---

### ステップ2: データの確認

**セル4: データファイルのパスを設定**
```python
pickle_path = '/content/drive/MyDrive/data_lowF_noise.pickle'  # 自分のパスに合わせて変更

import os
if os.path.exists(pickle_path):
    print(f"Data file found: {pickle_path}")
else:
    print(f"Warning: Data file not found at {pickle_path}")
```
- データファイルのパスを確認

---

### ステップ3: コードの準備（コードをアップロードした場合）

**方法A: Google Driveから読み込む**
- セル3で既に設定済み

**方法B: GitHubからクローン**
```python
!git clone https://github.com/taiga-hashimoto-21/noise4.git
!cp -r noise4/task .
!cp noise4/eval.py .
```

**方法C: コードを直接Colabにコピー**
- 各セルにコードを直接貼り付け

---

### ステップ4: モデルのインポート

**セル5: モジュールをインポート**
```python
from task.dataset import Task4Dataset
from task.model import Task4BERT
from task.train import train_task4
import eval
```
- エラーが出る場合は、コードのパスを確認

---

### ステップ5: 学習の実行

**セル6: デバイスの確認**
```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```
- GPUが使えるか確認

**セル7: 学習を実行**
```python
model, train_losses, val_losses, best_val_loss = train_task4(
    pickle_path=pickle_path,
    batch_size=8,
    num_epochs=200,
    lr=1e-4,
    val_ratio=0.2,
    device=device,
    out_dir="/content/drive/MyDrive/task4_output",
    resume=True,
    lambda_reg=0.1,
    num_intervals=30,
)
```
- 学習が開始されます
- 時間がかかるので、そのまま待ちます
- チェックポイントが自動保存されるので、途中で中断しても再開可能

---

### ステップ6: 評価の実行

**セル8: 評価を実行**
```python
from torch.utils.data import DataLoader

test_dataset = Task4Dataset(
    pickle_path=pickle_path,
    num_intervals=30,
    use_noise_labels=True,
)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

best_model_path = "/content/drive/MyDrive/task4_output/best_val_model.pth"
model.load_state_dict(torch.load(best_model_path, map_location=device))

results = eval.evaluate_model(
    model=model,
    dataloader=test_loader,
    method='self_supervised',
    device=device,
    num_intervals=30,
)

print("\n評価結果:")
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1-score: {results['f1_score']:.4f}")
if 'roc_auc' in results:
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
if 'attention_diff' in results:
    print(f"アテンションウェイトの差: {results['attention_diff']:.4f}")
```

**セル9: 混同行列を可視化**
```python
eval.plot_confusion_matrix(
    results['confusion_matrix'],
    title='タスク4: 混同行列',
    save_path="/content/drive/MyDrive/task4_output/confusion_matrix.png"
)
print("混同行列を保存しました")
```

**セル10: アテンションウェイトの分布を可視化**
```python
if 'attention_weights' in results:
    eval.plot_attention_distribution(
        results['attention_weights'],
        results['labels'],
        num_intervals=30,
        save_path="/content/drive/MyDrive/task4_output/attention_distribution.png"
    )
    print("アテンションウェイトの分布を保存しました")
```

---

## セルの実行方法

### 基本的な実行方法

1. **1つのセルを実行**: セルをクリックして、`Shift + Enter`を押す
2. **すべてのセルを実行**: `Runtime` → `Run all`
3. **セルを上から順に実行**: `Runtime` → `Run before`

### セルの実行順序

**必ず上から順に実行してください**:
1. 環境設定（セル1-3）
2. データの確認（セル4）
3. モデルのインポート（セル5）
4. 学習の実行（セル6-7）
5. 評価の実行（セル8-10）

---

## トラブルシューティング

### エラー: ModuleNotFoundError

**原因**: モジュールが見つからない

**解決策**:
- コードのパスを確認
- `sys.path.append()`でパスを追加
- または、コードを直接Colabにコピー

### エラー: FileNotFoundError

**原因**: データファイルが見つからない

**解決策**:
- データファイルのパスを確認
- Google Driveにアップロードされているか確認

### エラー: CUDA out of memory

**原因**: GPUメモリ不足

**解決策**:
- `batch_size`を小さくする（例: 8 → 4）
- `d_model`を小さくする（例: 64 → 32）

### 学習がうまくいかない

**解決策**:
- `lambda_reg`を調整（例: 0.1 → 0.05 または 0.2）
- 学習率を調整（例: 1e-4 → 5e-5）
- エポック数を増やす

---

## チェックポイントからの再開

学習が途中で中断された場合、`resume=True`にすると、チェックポイントから再開できます：

```python
model, train_losses, val_losses, best_val_loss = train_task4(
    ...,
    resume=True,  # チェックポイントから再開
    ...
)
```

---

## 出力ファイルの確認

学習後、以下のファイルがGoogle Driveに保存されます：

```
Google Drive/
└── MyDrive/
    └── task4_output/
        ├── checkpoint.pth              # チェックポイント
        ├── best_train_model.pth        # 訓練損失最小のモデル
        ├── best_val_model.pth          # 検証損失最小のモデル
        ├── final_model.pth             # 最終モデル
        ├── loss_curve.png              # 損失カーブ
        ├── confusion_matrix.png        # 混同行列
        ├── attention_distribution.png  # アテンション分布
        ├── train_losses.pkl            # 訓練損失履歴
        └── val_losses.pkl              # 検証損失履歴
```

---

## ベースラインとの比較（オプション）

既にベースラインモデルを学習済みの場合、比較できます：

```python
# ベースラインモデルを読み込み
from baseline.baseline_model import SimpleCNN
baseline_model = SimpleCNN(num_classes=30).to(device)
baseline_model.load_state_dict(torch.load("/content/drive/MyDrive/baseline_model.pth", map_location=device))

# ベースラインを評価
baseline_results = eval.evaluate_model(
    model=baseline_model,
    dataloader=test_loader,
    method='baseline',
    device=device,
    num_intervals=30,
)

# タスク4を評価
ssl_results = eval.evaluate_model(
    model=model,
    dataloader=test_loader,
    method='self_supervised',
    device=device,
    num_intervals=30,
)

# 比較
comparison = eval.compare_methods(baseline_results, ssl_results)
```

---

## まとめ

1. ✅ Google Driveにデータとコードをアップロード
2. ✅ Colabでノートブックを作成
3. ✅ セルを上から順に実行
4. ✅ 学習が完了したら評価を実行
5. ✅ 結果を確認

これで完了です！

