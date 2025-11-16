# ノイズ検知プロジェクト

PSD（Power Spectral Density）データにおけるノイズ検知の研究プロジェクトです。

## プロジェクト構成

```
noise/
├── baseline/          # ベースライン（畳み込みモデル）
├── ssl/              # Self-Supervised Learning（自己教師あり学習）
├── noise/            # 共通：ノイズ生成モジュール
├── eval.py           # 共通：評価スクリプト
├── data_lowF_noise.pickle  # データ
└── docs/             # ドキュメント
```

## クイックスタート

### ベースライン（畳み込みモデル）

```bash
# データセット準備
python baseline/dataset.py

# 学習
python baseline/train.py
```

詳細は `baseline/README.md` を参照してください。

### 自己教師あり学習

Google Colabで `ssl/train_colab.ipynb` を実行してください。

詳細は `ssl/README.md` を参照してください。

## 評価

両方のモデルを評価するには、ルートの `eval.py` を使用します：

```python
from eval import evaluate_model, compare_methods

# ベースラインモデルの評価
baseline_results = evaluate_model(...)

# SSLモデルの評価
ssl_results = evaluate_model(...)

# 比較
compare_methods(baseline_results, ssl_results)
```

## データ

- `data_lowF_noise.pickle`: PSD理論値データ（32000サンプル、各3000ポイント）

## ノイズ生成

`noise/` フォルダには以下のノイズ生成関数が含まれています：

- `frequency_band_noise.py`: 周波数帯域集中ノイズ
- `localized_spike_noise.py`: 局所スパイクノイズ
- `amplitude_dependent_noise.py`: 振幅依存ノイズ

## ドキュメント

- `docs/baseline/`: ベースライン関連のドキュメント
- `docs/ssl/`: 自己教師あり学習関連のドキュメント

