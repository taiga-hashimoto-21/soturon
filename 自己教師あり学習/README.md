# Self-Supervised Learning（自己教師あり学習）

このフォルダには、Transformer/BERTベースの自己教師あり学習モデルが含まれています。

## ファイル構成

- `task/`: タスク関連コード
  - `dataset.py`: データセットクラス（Task4Dataset）
  - `model.py`: BERTモデル（Task4BERT）
  - `train.py`: 学習スクリプト
- `train_colab.ipynb`: Google Colab用の学習ノートブック
- `output/`: 学習結果（モデル、損失カーブなど）
- `utils/`: ユーティリティスクリプト
  - `debug_attention.py`: アテンションウェイトのデバッグ

## 使用方法

1. **固定データセットの準備**（評価用）:
   ```bash
   cd 自己教師あり学習
   python prepare_dataset.py
   ```
   これで `ssl_dataset.pickle` が作成されます（畳み込みと同じ方法で準備）。

2. Colabで学習:
   - `ssl/train_colab.ipynb` をGoogle Colabで開く
   - 必要なファイルをアップロード
   - セルを順番に実行
   - 学習時は動的にデータを生成（`use_fixed_dataset=False`）

3. 評価:
   - ルートの `eval.py` を使用（ベースラインと共通）
   - 評価時は固定データセットを使用（`use_fixed_dataset=True`）
   ```python
   from task.dataset import Task4Dataset
   
   # 評価時は固定データセットを使用
   test_dataset = Task4Dataset(
       pickle_path="data_lowF_noise.pickle",
       use_fixed_dataset=True,
       split='test',  # 'train', 'val', 'test'
   )
   ```

## タスク4: マスク予測 + 正則化項

- **マスク予測**: 15%の区間をマスクし、元の値を予測
- **正則化項**: ランキング損失を使用して、ノイズ区間のアテンション < 正常区間のアテンション になるように学習

## データの前処理

- スケーリング: `2.5e24`
- ログ変換: `np.log(x.clamp(min=1e-30))`
- 正規化: ログ変換後のデータを正規化（mean=0, std=1）
- 構造化ノイズ: 実験データに近づけるため、全体的な構造化ノイズを付与
- 区間ノイズ: 30区間のうち1区間にノイズを付与（ノイズ検知の対象）

## 詳細

詳細な設計については `docs/ssl/task_design.md` を参照してください。

