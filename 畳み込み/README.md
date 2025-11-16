# ベースライン（畳み込みモデル）

このフォルダには、畳み込みニューラルネットワーク（CNN）ベースのノイズ検知モデルが含まれています。

## ファイル構成

- `model.py`: ベースラインモデル（CNN1d_with_resnetなど）
- `dataset.py`: データセット準備スクリプト（ノイズ付与、前処理）
- `train.py`: 学習スクリプト
- `train_colab.ipynb`: Google Colab用の学習ノートブック
- `utils/`: ユーティリティスクリプト
  - `visualize.py`: データ可視化
  - `analyze.py`: データ分析
  - `evaluate.py`: 評価スクリプト
  - `verify.py`: データセット検証

## 使用方法

1. データセットの準備:
```bash
python baseline/dataset.py
```

2. 学習:
```bash
python baseline/train.py
```

または、Colabで `baseline/train_colab.ipynb` を実行

## データの前処理

- スケーリング: `2.5e24`
- ログ変換: `torch.log(x.clamp(min=1e-30))`
- 正規化: ログ変換後のデータを正規化（mean=0, std=1）
- 構造化ノイズ: 実験データに近づけるため、全体的な構造化ノイズを付与

## ノイズタイプ

- `frequency_band`: 周波数帯域集中ノイズ
- `localized_spike`: 局所スパイクノイズ
- `amplitude_dependent`: 振幅依存ノイズ

