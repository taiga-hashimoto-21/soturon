"""
活性化エネルギー予測の評価を実行するスクリプト
"""
import sys
import os

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ノイズモジュールのパスを追加
noise_module_path = os.path.join(project_root, 'ノイズの付与(共通)')
sys.path.insert(0, noise_module_path)

# ノイズモジュールをパッケージとして設定
import importlib.util
noise_init_path = os.path.join(noise_module_path, '__init__.py')
noise_init_spec = importlib.util.spec_from_file_location("noise", noise_init_path)
noise_init_module = importlib.util.module_from_spec(noise_init_spec)
sys.modules['noise'] = noise_init_module
noise_init_module.__file__ = noise_init_path
noise_init_module.__path__ = [noise_module_path]
noise_init_spec.loader.exec_module(noise_init_module)

# add_noise.pyをロード
add_noise_path = os.path.join(noise_module_path, 'add_noise.py')
add_noise_spec = importlib.util.spec_from_file_location("noise.add_noise", add_noise_path)
add_noise_module = importlib.util.module_from_spec(add_noise_spec)
sys.modules['noise.add_noise'] = add_noise_module
add_noise_module.__file__ = add_noise_path
add_noise_spec.loader.exec_module(add_noise_module)

# 活性化エネルギー予測フォルダをパスに追加
activation_energy_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, activation_energy_dir)

# 自己教師あり学習フォルダをパスに追加
ssl_folder = os.path.join(project_root, '自己教師あり学習')
sys.path.insert(0, ssl_folder)
sys.path.insert(0, os.path.join(ssl_folder, 'task'))

# 評価を実行
from evaluate_activation_energy import evaluate_activation_energy_prediction

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='自己教師あり学習/task4_output/best_val_model.pth')
    parser.add_argument('--pickle_path', type=str, default='data_lowF_noise.pickle')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--noise_type', type=str, default='power_supply')
    parser.add_argument('--use_random_noise', action='store_true', default=False)
    parser.add_argument('--noise_level', type=float, default=0.3)
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('活性化エネルギー予測の評価を開始します')
    print('=' * 60)
    print(f'モデルパス: {args.model_path}')
    print(f'データパス: {args.pickle_path}')
    print(f'デバイス: {args.device}')
    print(f'サンプル数: {args.num_samples}')
    print(f'ノイズタイプ: {args.noise_type}')
    print('=' * 60)
    
    results = evaluate_activation_energy_prediction(
        model_path=args.model_path,
        pickle_path=args.pickle_path,
        device=args.device,
        num_samples=args.num_samples,
        noise_type=args.noise_type,
        use_random_noise=args.use_random_noise,
        noise_level=args.noise_level,
    )
    
    print('\n' + '=' * 60)
    print('実行完了')
    print('=' * 60)
    if results:
        print(f'評価サンプル数: {results["num_samples"]}')
        print(f'ノイズありの平均誤差: {results["error_before"]:.6f} meV')
        print(f'ノイズ除去後の平均誤差: {results["error_after"]:.6f} meV')
        print(f'改善度: {results["improvement"]:.6f} meV')
        print(f'改善率: {results["improvement_rate"]:.2f}%')
    else:
        print('エラー: 結果が取得できませんでした')

