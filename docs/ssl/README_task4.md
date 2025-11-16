# タスク4: マスク予測 + 正則化項

## 概要

自己教師あり学習によるノイズ検出の実装です。
- **主タスク**: マスクされた区間を復元
- **副産物**: ノイズ区間のアテンションウェイトが低くなることでノイズ検出能力を獲得

## ファイル構成

```
noise/
├── task/                    # タスク4のコード
│   ├── __init__.py
│   ├── dataset.py          # データセットクラス
│   ├── model.py            # BERTモデル
│   └── train.py            # 学習スクリプト
├── eval.py                  # 統一評価コード（ベースラインとタスク4の両方に対応）
└── task4_colab.ipynb        # Google Colab用のノートブック
```

## Google Colabでの実行方法

### 1. セットアップ

1. Google Colabで新しいノートブックを作成
2. `task4_colab.ipynb`の内容をコピー＆ペースト
3. データファイル（`data_lowF_noise.pickle`）をGoogle Driveにアップロード

### 2. セルの実行順序

1. **環境設定セル**: Google Driveをマウント、ライブラリをインストール
2. **データの確認セル**: データファイルのパスを確認
3. **モデルのインポートセル**: タスク4のモジュールをインポート
4. **学習の実行セル**: モデルを学習
5. **評価の実行セル**: テストデータで評価

### 3. パスの設定

以下のパスを自分の環境に合わせて変更してください：

```python
# データファイルのパス
pickle_path = '/content/drive/MyDrive/data_lowF_noise.pickle'

# プロジェクトのパス（コードをアップロードした場合）
project_path = '/content/drive/MyDrive/noise'

# 出力先のパス
out_dir = "/content/drive/MyDrive/task4_output"
```

### 4. コードのアップロード方法

**方法1: Google Driveにアップロード**
1. `task`フォルダと`eval.py`をGoogle Driveにアップロード
2. ノートブックでパスを設定

**方法2: GitHubからクローン**
```python
!git clone https://github.com/taiga-hashimoto-21/noise4.git
!cp -r noise4/task .
!cp noise4/eval.py .
```

## 使用方法

### 学習

```python
from task.train import train_task4

model, train_losses, val_losses, best_val_loss = train_task4(
    pickle_path="data_lowF_noise.pickle",
    batch_size=8,
    num_epochs=200,
    lr=1e-4,
    val_ratio=0.2,
    device="cuda",
    out_dir="task4_output",
    resume=True,
    lambda_reg=0.1,  # 正則化項の重み（調整可能）
    num_intervals=30,
)
```

### 評価

```python
import eval
from torch.utils.data import DataLoader
from task.dataset import Task4Dataset

# テストデータセットを作成
test_dataset = Task4Dataset(
    pickle_path="data_lowF_noise.pickle",
    num_intervals=30,
    use_noise_labels=True,
)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# モデルを読み込み
model.load_state_dict(torch.load("task4_output/best_val_model.pth"))

# 評価を実行
results = eval.evaluate_model(
    model=model,
    dataloader=test_loader,
    method='self_supervised',
    device='cuda',
    num_intervals=30,
)

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1-score: {results['f1_score']:.4f}")
```

### ベースラインとの比較

```python
# ベースラインモデルを評価
baseline_results = eval.evaluate_model(
    model=baseline_model,
    dataloader=test_loader,
    method='baseline',
    device='cuda',
    num_intervals=30,
)

# タスク4モデルを評価
ssl_results = eval.evaluate_model(
    model=task4_model,
    dataloader=test_loader,
    method='self_supervised',
    device='cuda',
    num_intervals=30,
)

# 比較
comparison = eval.compare_methods(baseline_results, ssl_results)
```

## ハイパーパラメータ

### 推奨値

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `batch_size` | 8 | バッチサイズ |
| `num_epochs` | 200 | エポック数 |
| `lr` | 1e-4 | 学習率 |
| `lambda_reg` | 0.1 | 正則化項の重み（調整可能） |
| `num_intervals` | 30 | 区間数 |
| `d_model` | 64 | 埋め込み次元 |
| `n_heads` | 2 | アテンションヘッド数 |
| `num_layers` | 2 | Transformerレイヤー数 |

### 調整が必要なパラメータ

- **`lambda_reg`**: 正則化項の重み
  - 小さすぎる（0.01など）: ノイズ検出が効かない
  - 大きすぎる（1.0など）: マスク予測がうまくいかない
  - 推奨: 0.05〜0.2の範囲で調整

## 出力ファイル

学習後、`out_dir`に以下のファイルが保存されます：

- `checkpoint.pth`: チェックポイント（再開用）
- `best_train_model.pth`: 訓練損失が最小のモデル
- `best_val_model.pth`: 検証損失が最小のモデル
- `final_model.pth`: 最終エポックのモデル
- `loss_curve.png`: 損失カーブのグラフ
- `train_losses.pkl`: 訓練損失の履歴
- `val_losses.pkl`: 検証損失の履歴

## トラブルシューティング

### アテンションウェイトが取得できない

PyTorchの標準実装では、アテンションウェイトを直接取得できない場合があります。
その場合は、簡易版の正則化項を使用してください（予測値を使う方法）。

### メモリ不足

- `batch_size`を小さくする（例: 8 → 4）
- `d_model`を小さくする（例: 64 → 32）

### 学習がうまくいかない

- `lambda_reg`を調整する
- 学習率を調整する（例: 1e-4 → 5e-5）
- エポック数を増やす

## 参考

- [自己教師あり学習タスク設計.md](自己教師あり学習タスク設計.md)
- [BERTとCLSトークンの説明.md](BERTとCLSトークンの説明.md)

