# 活性化エネルギー予測

ノイズ除去前後のPSDデータから活性化エネルギーを予測し、精度を比較するモジュール。

## 目的

第2目標: ノイズ除去後のデータで解析結果（活性化エネルギーの予測精度）が向上することを示す。

## フォルダ構造

```
活性化エネルギー予測/
├── README.md                    # このファイル
├── activation_energy.py         # 活性化エネルギー計算の理論式
├── evaluate_activation_energy.py # ノイズ除去前後で精度を比較
└── visualize_results.py         # 結果の可視化
```

## 理論式

### 式(1): PSDの理論式
```
S_I(f) = (A_α(I_bias)I_bias^2) / (1 + (f/f_α)^2) + (A_β(I_bias)I_bias^2) / (1 + (f/f_β)^2) + C
```

### 式(2): 緩和時間と活性化エネルギーの関係
```
τ(T) = 1 / (2πf(T)) = 1 / (σ^(0) × exp(-E / (k_B T)) × N_T × ν)
```

### パラメータ
- **Eα**: 0.1 ~ 20 meV (活性化エネルギーα)
- **Eβ**: 12.5 ~ 32.5 meV (活性化エネルギーβ)
- **NT**: (4 × 10^11)^1.5 × 1.0~1.3 /cm³
- **V**: 10^7 cm/s
- **KB**: 8.6 × 10^-5 eV/k (ボルツマン定数)
- **T**: 300 k (温度)
- **σα**: 10^-23 cm²
- **σβ**: σα × 830 cm²

## 使用方法

### 1. 活性化エネルギーの計算

```python
from activation_energy import calculate_activation_energy_from_psd

# PSDデータから活性化エネルギーを計算
psd_data = ...  # (3000,) のPSDデータ
E_alpha, E_beta = calculate_activation_energy_from_psd(psd_data)
```

### 2. ノイズ除去前後で精度を比較

```python
from evaluate_activation_energy import evaluate_activation_energy_prediction

# 学習済みモデルを使って評価（デフォルト: ランダムにノイズタイプを選択）
results = evaluate_activation_energy_prediction(
    model_path='自己教師あり学習/task4_output/best_val_model.pth',
    pickle_path='data_lowF_noise.pickle',
    device='cuda'
)

print(f"ノイズありの誤差: {results['error_before']:.6f}")
print(f"ノイズ除去後の誤差: {results['error_after']:.6f}")
print(f"改善度: {results['improvement']:.6f}")
```

### 3. 特定のノイズタイプで評価

```python
# 特定のノイズタイプを指定して評価
results = evaluate_activation_energy_prediction(
    model_path='自己教師あり学習/task4_output/best_val_model.pth',
    pickle_path='data_lowF_noise.pickle',
    device='cuda',
    noise_type='power_supply',  # 'power_supply', 'interference', 'clock_leakage'
    use_random_noise=False,  # ランダム選択を無効化
    noise_level=0.3,
)
```

### 4. 新しいノイズタイプを追加する方法

1. **ノイズ生成関数を作成**
   - `ノイズの付与(共通)/`フォルダに新しいノイズ生成関数を追加
   - 例: `new_noise.py`に`add_new_noise()`関数を実装

2. **`add_noise.py`に登録**
   ```python
   # ノイズの付与(共通)/add_noise.py
   from .new_noise import add_new_noise
   
   def add_noise_to_interval(psd_data, interval_idx, noise_type='power_supply', **kwargs):
       ...
       elif noise_type == 'new_noise':
           return add_new_noise(psd_data, interval_idx, **kwargs)
   ```

3. **評価時に使用**
   ```python
   results = evaluate_activation_energy_prediction(
       ...
       noise_type='new_noise',  # 新しいノイズタイプを指定
       use_random_noise=False,
   )
   ```

これにより、学習済みモデルの重みを使って、新しいノイズタイプでも精度が向上するかどうかを確認できます。

## 評価指標

- **誤差（MSE）**: 予測した活性化エネルギーと正解データ（`y`）の平均二乗誤差
- **改善度**: ノイズ除去前の誤差 - ノイズ除去後の誤差（正の値なら改善）

## データの対応関係

- **正解データ**: `data_lowF_noise.pickle`の`y`
  - `y[:, 0]` = Eα / 10 (活性化エネルギーα)
  - `y[:, 1]` = Eβ / 10 (活性化エネルギーβ)

