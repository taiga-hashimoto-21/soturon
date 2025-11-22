"""
統一評価コード
ベースライン（畳み込み）と自己教師あり学習（タスク4）の両方に対応
評価時の損失関数を統一（CrossEntropyLoss）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import platform

# 日本語フォントの設定
if platform.system() == 'Darwin':  # Mac
    plt.rcParams['font.family'] = 'Hiragino Sans'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def evaluate_baseline_model(model, dataloader, device='cuda', num_intervals=30):
    """
    ベースライン（畳み込み）モデルの評価
    
    Args:
        model: ベースラインモデル（30クラス分類）
        dataloader: データローダー
        device: デバイス
        num_intervals: 区間数
    
    Returns:
        dict: 評価指標の辞書
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch['input'].to(device)
                labels = batch.get('noise_interval', None)
                if labels is None:
                    # ラベルがない場合はスキップ
                    continue
            else:
                x, labels = batch
                x = x.to(device)
                labels = labels.to(device)
            
            # 予測
            logits = model(x)  # (batch_size, num_intervals)
            predictions = logits.argmax(dim=1)  # 最も確率が高い区間
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits_tensor = torch.cat(all_logits, dim=0)  # (N, num_intervals)
    labels_tensor = torch.from_numpy(all_labels).long()
    
    # baselineと同じ方法: CrossEntropyLossで評価
    criterion = nn.CrossEntropyLoss()
    evaluation_loss = criterion(all_logits_tensor, labels_tensor).item()
    
    # 評価指標を計算
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 復元損失: baselineモデルは復元タスクをしていないため、Noneを返す
    reconstruction_loss = None
    
    # 主要な損失: 復元損失が最も重要だが、baselineモデルは復元タスクをしていない
    # そのため、マスク予測の損失（CrossEntropyLoss）を返す
    # ただし、復元損失が利用可能な場合はそれを優先する
    primary_loss = reconstruction_loss if reconstruction_loss is not None else evaluation_loss
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'loss': primary_loss,  # 復元損失が利用可能な場合は復元損失、そうでなければマスク予測損失
        'reconstruction_loss': reconstruction_loss,  # 復元損失（baselineモデルではNone）
        'mask_evaluation_loss': evaluation_loss,  # CrossEntropyLoss（マスク予測の損失）
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
    }


def evaluate_self_supervised_model(model, dataloader, device='cuda', num_intervals=30):
    """
    自己教師あり学習（タスク4）モデルの評価
    
    Args:
        model: タスク4モデル（BERT）
        dataloader: データローダー
        device: デバイス
        num_intervals: 区間数
    
    Returns:
        dict: 評価指標の辞書
    """
    model.eval()
    all_attention_weights = []
    all_labels = []
    
    print("アテンションウェイトを計算中...")
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 100 == 0:
                print(f"  バッチ {batch_idx}/{total_batches} を処理中...")
            
            x = batch['input'].to(device)
            m = batch['mask'].to(device)
            labels = batch['noise_interval'].to(device)
            
            # アテンションウェイトを取得（4つの値を返す: out, cls_out, attention_weights, reconstructed_interval）
            _, _, attention_weights, _ = model(x, m, return_attention=True)
            
            # デバッグ: 最初のバッチでアテンションウェイトの形状と値を確認
            if batch_idx == 0 and attention_weights is not None:
                print(f"\nデバッグ情報（最初のバッチ）:")
                print(f"  アテンションウェイトの形状: {attention_weights.shape}")
                print(f"  アテンションウェイトの最小値: {attention_weights.min().item():.6f}")
                print(f"  アテンションウェイトの最大値: {attention_weights.max().item():.6f}")
                print(f"  アテンションウェイトの平均値: {attention_weights.mean().item():.6f}")
                print(f"  CLSトークン(0)から系列(1:)へのアテンション: {attention_weights[0, :, 0, 1:10].mean().item():.6f}")
            
            if attention_weights is not None:
                # CLSトークンから各区間へのアテンションを取得
                if hasattr(model, 'get_cls_attention_to_intervals'):
                    cls_attention = model.get_cls_attention_to_intervals(
                        attention_weights, num_intervals=num_intervals
                    )
                else:
                    # フォールバック: 手動で計算
                    B = attention_weights.shape[0]
                    L = x.shape[1]
                    points_per_interval = L // num_intervals
                    
                    # CLSトークンはインデックス0
                    cls_attention_full = attention_weights[:, :, 0, 1:]  # (B, n_heads, L)
                    cls_attention_mean = cls_attention_full.mean(dim=1)  # (B, L)
                    
                    # 区間ごとに平均化
                    cls_attention_intervals = []
                    for i in range(num_intervals):
                        start_idx = i * points_per_interval
                        end_idx = min(start_idx + points_per_interval, L)
                        interval_attn = cls_attention_mean[:, start_idx:end_idx].mean(dim=1)
                        cls_attention_intervals.append(interval_attn)
                    
                    cls_attention = torch.stack(cls_attention_intervals, dim=1)  # (B, num_intervals)
                
                # 学習時と同じスケーリングを適用（train.pyと一致させる）
                if cls_attention is not None:
                    attention_scale = 100.0  # train.pyと同じ値（10000 → 100に変更）
                    cls_attention = cls_attention * attention_scale
                    all_attention_weights.append(cls_attention.cpu())
                    all_labels.append(labels.cpu())
    
    if len(all_attention_weights) == 0:
        print("Warning: No attention weights collected")
        return None
    
    all_attention = torch.cat(all_attention_weights, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # デバッグ: アテンションウェイトの統計情報を表示
    print(f"\nアテンションウェイトの統計情報:")
    print(f"  形状: {all_attention.shape}")
    print(f"  最小値: {all_attention.min():.6f}")
    print(f"  最大値: {all_attention.max():.6f}")
    print(f"  平均値: {all_attention.mean():.6f}")
    print(f"  標準偏差: {all_attention.std():.6f}")
    print(f"  サンプル数: {len(all_labels)}")
    
    # GitHubリポジトリと同じ評価方法（閾値最適化）
    # ノイズ区間と正常区間のアテンションウェイトを比較
    noise_attention = []
    normal_attention = []
    for i, label in enumerate(all_labels):
        noise_attention.append(all_attention[i, label])
        normal_indices = [j for j in range(num_intervals) if j != label]
        normal_attention.append(all_attention[i, normal_indices].mean())
    
    noise_attention = np.array(noise_attention)
    normal_attention = np.array(normal_attention)
    
    # アテンションウェイトの平均値比較
    attention_diff = normal_attention.mean() - noise_attention.mean()
    
    print(f"\n区間別アテンションウェイト:")
    print(f"  ノイズ区間の平均: {noise_attention.mean():.6f} (最小: {noise_attention.min():.6f}, 最大: {noise_attention.max():.6f})")
    print(f"  正常区間の平均: {normal_attention.mean():.6f} (最小: {normal_attention.min():.6f}, 最大: {normal_attention.max():.6f})")
    print(f"  差（正常 - ノイズ）: {attention_diff:.6f}")
    
    # 閾値を最適化（F1-scoreが最大になる閾値を選択）
    # 設計意図: ノイズ区間のアテンション < 正常区間のアテンション
    # → アテンションが低い区間をノイズと判定
    print("閾値を最適化中...")
    thresholds = np.linspace(all_attention.min(), all_attention.max(), 50)  # 100 → 50に減らす
    best_threshold = None
    best_f1 = 0
    best_accuracy = 0
    
    for idx, threshold in enumerate(thresholds):
        if idx % 10 == 0:
            print(f"  閾値 {idx}/{len(thresholds)} を評価中...")
        # 各サンプルで、閾値以下の区間を「ノイズあり」と判定
        predictions = []
        for i in range(len(all_labels)):
            noisy_intervals = np.where(all_attention[i] < threshold)[0]
            if len(noisy_intervals) > 0:
                # 最もアテンションウェイトが低い区間を予測
                predictions.append(noisy_intervals[np.argmin(all_attention[i, noisy_intervals])])
            else:
                # 閾値以下の区間がない場合、最もアテンションウェイトが低い区間を予測
                predictions.append(np.argmin(all_attention[i]))
        
        predictions = np.array(predictions)
        f1 = f1_score(all_labels, predictions, average='macro', zero_division=0)
        accuracy = accuracy_score(all_labels, predictions)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_accuracy = accuracy
    
    # best_thresholdがNoneの場合のフォールバック処理
    if best_threshold is None:
        print("警告: 最適な閾値が見つかりませんでした。デフォルトの方法を使用します。")
        # 最もアテンションウェイトが低い区間を予測する方法にフォールバック
        final_predictions = np.array([np.argmin(all_attention[i]) for i in range(len(all_labels))])
        best_threshold = all_attention.mean()  # 平均値をデフォルト閾値として設定
        best_f1 = f1_score(all_labels, final_predictions, average='macro', zero_division=0)
        best_accuracy = accuracy_score(all_labels, final_predictions)
    else:
        # 最適な閾値で最終予測
        final_predictions = []
        for i in range(len(all_labels)):
            noisy_intervals = np.where(all_attention[i] < best_threshold)[0]
            if len(noisy_intervals) > 0:
                final_predictions.append(noisy_intervals[np.argmin(all_attention[i, noisy_intervals])])
            else:
                final_predictions.append(np.argmin(all_attention[i]))
        final_predictions = np.array(final_predictions)
    
    # baselineと同じ方法: CrossEntropyLossで評価
    # アテンションウェイトからlogitsを生成（最もアテンションウェイトが低い区間がノイズ）
    # アテンションウェイトが低いほど、ノイズの可能性が高い
    # logits = -attention_weights（アテンションウェイトが低いほどスコアが高い）
    attention_logits = -all_attention  # (N, num_intervals)
    
    # CrossEntropyLossで評価（baselineと同じ方法）
    criterion = nn.CrossEntropyLoss()
    attention_logits_tensor = torch.from_numpy(attention_logits).float()
    labels_tensor = torch.from_numpy(all_labels).long()
    evaluation_loss = criterion(attention_logits_tensor, labels_tensor).item()
    
    # 評価指標を計算
    accuracy = best_accuracy
    precision = precision_score(all_labels, final_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, final_predictions, average='macro', zero_division=0)
    f1 = best_f1
    cm = confusion_matrix(all_labels, final_predictions)
    
    # ROC-AUC（二値分類として扱う）
    # アテンションウェイトが低いほどノイズの可能性が高い
    y_true_binary = []
    y_score_binary = []
    for i, label in enumerate(all_labels):
        y_true_binary.append(1)  # ノイズ区間
        y_score_binary.append(1 - all_attention[i, label])  # アテンションウェイトが低いほどスコアが高い
        for j in range(num_intervals):
            if j != label:
                y_true_binary.append(0)  # 正常区間
                y_score_binary.append(1 - all_attention[i, j])
    
    try:
        roc_auc = roc_auc_score(y_true_binary, y_score_binary)
    except:
        roc_auc = 0.0
    
    # 復元損失を計算（ノイズ検知した区間の復元精度）
    # データローダーを再度ループして復元損失を計算
    model.eval()
    all_reconstruction_losses = []
    all_reconstruction_accuracies = []
    points_per_interval = 3000 // num_intervals  # 100ポイント
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x = batch['input'].to(device)  # マスクされたデータ
            mask = batch['mask'].to(device)  # マスク位置
            noise_intervals = batch['noise_interval'].to(device)  # 真のノイズ区間
            original_data = batch.get('original', None)  # 元のデータ（ノイズ付与前）
            
            if original_data is None:
                # 固定データセットの場合は復元損失を計算できない
                continue
            
            original_data = original_data.to(device)
            
            # モデルの出力（復元機能付き）
            _, _, attention_weights, reconstructed_interval = model(
                x, mask, return_attention=True, num_intervals=num_intervals
            )
            # reconstructed_interval: (B, 100) - 予測したノイズ区間のみ復元
            
            if reconstructed_interval is None:
                continue
            
            # アテンションウェイトからノイズ区間を予測
            if attention_weights is not None:
                # CLSトークンから各区間へのアテンションを取得
                if hasattr(model, 'get_cls_attention_to_intervals'):
                    cls_attention = model.get_cls_attention_to_intervals(
                        attention_weights, num_intervals=num_intervals
                    )
                else:
                    # フォールバック: 手動で計算
                    B = attention_weights.shape[0]
                    L = x.shape[1]
                    cls_attention_full = attention_weights[:, :, 0, 1:]  # (B, n_heads, L)
                    cls_attention_mean = cls_attention_full.mean(dim=1)  # (B, L)
                    
                    cls_attention_intervals = []
                    for i in range(num_intervals):
                        start_idx = i * points_per_interval
                        end_idx = min(start_idx + points_per_interval, L)
                        interval_attn = cls_attention_mean[:, start_idx:end_idx].mean(dim=1)
                        cls_attention_intervals.append(interval_attn)
                    
                    cls_attention = torch.stack(cls_attention_intervals, dim=1)  # (B, num_intervals)
                
                # 最もアテンションが低い区間をノイズ区間として予測
                predicted_noise_intervals = cls_attention.argmin(dim=1)  # (B,)
                
                # 復元損失を計算（予測が正しい場合のみ - アプローチ2）
                for i in range(x.size(0)):
                    pred_interval = predicted_noise_intervals[i].item()
                    true_interval = noise_intervals[i].item()
                    
                    # 予測が正しい場合のみ復元損失を計算
                    if pred_interval == true_interval:
                        # 真のノイズ区間のインデックス範囲を計算
                        start_idx = true_interval * points_per_interval
                        end_idx = min(start_idx + points_per_interval, original_data.size(1))
                        
                        # 復元した区間と元のデータを比較
                        pred_reconstructed = reconstructed_interval[i]  # (100,)
                        true_original = original_data[i, start_idx:end_idx]  # (100,)
                        
                        # MSE損失
                        interval_loss = F.mse_loss(pred_reconstructed, true_original)
                        all_reconstruction_losses.append(interval_loss.item())
                        
                        # 復元精度（%）を計算
                        # 正規化されたデータでは相対誤差が厳しすぎるため、
                        # MSE損失から復元精度を計算する方法に変更
                        # 復元精度 = max(0, (1 - sqrt(MSE) / std) * 100)
                        # std: データの標準偏差（約0.82）
                        mse_loss = interval_loss.item()
                        data_std = 0.82  # 正規化後の標準偏差（約0.82）
                        rmse = np.sqrt(mse_loss)
                        relative_rmse = rmse / data_std
                        reconstruction_accuracy = max(0.0, (1.0 - relative_rmse) * 100.0)
                        all_reconstruction_accuracies.append(reconstruction_accuracy)
    
    # 復元損失の平均
    if len(all_reconstruction_losses) > 0:
        reconstruction_loss = np.mean(all_reconstruction_losses)
        reconstruction_accuracy = np.mean(all_reconstruction_accuracies)
    else:
        reconstruction_loss = None
        reconstruction_accuracy = None
    
    # 主要な損失: 復元損失（MSE損失）
    # 復元損失が最も重要なので、'loss'キーに復元損失を設定
    primary_loss = reconstruction_loss if reconstruction_loss is not None else float('inf')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'loss': primary_loss,  # 復元損失（MSE損失）- 最も重要な指標
        'reconstruction_loss': reconstruction_loss,  # 復元損失（明示的に）
        'reconstruction_accuracy': reconstruction_accuracy,  # 復元精度（%）
        'mask_evaluation_loss': evaluation_loss,  # CrossEntropyLoss（ノイズ検知の損失）
        'attention_diff': attention_diff,
        'best_threshold': best_threshold,
        'confusion_matrix': cm,
        'noise_attention_mean': noise_attention.mean(),
        'normal_attention_mean': normal_attention.mean(),
        'predictions': final_predictions,
        'labels': all_labels,
        'attention_weights': all_attention,
    }


def evaluate_model(model, dataloader, method='baseline', device='cuda', num_intervals=30):
    """
    統一評価関数
    
    Args:
        model: モデル（ベースライン or タスク4）
        dataloader: データローダー
        method: 'baseline' or 'self_supervised'
        device: デバイス
        num_intervals: 区間数
    
    Returns:
        dict: 評価指標の辞書
    """
    if method == 'baseline':
        return evaluate_baseline_model(model, dataloader, device, num_intervals)
    elif method == 'self_supervised':
        return evaluate_self_supervised_model(model, dataloader, device, num_intervals)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'baseline' or 'self_supervised'")


def plot_confusion_matrix(cm, title='混同行列', save_path=None):
    """混同行列を可視化"""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 軸ラベル
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xlabel='予測区間',
           ylabel='正解区間',
           title=title)
    
    # 数値を表示
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_distribution(attention_weights, labels, num_intervals=30, save_path=None):
    """アテンションウェイトの分布を可視化"""
    attention_weights = np.array(attention_weights)
    labels = np.array(labels)
    
    # ノイズ区間と正常区間のアテンションウェイトを分離
    noise_attention = []
    normal_attention = []
    for i, label in enumerate(labels):
        noise_attention.append(attention_weights[i, label])
        normal_indices = [j for j in range(num_intervals) if j != label]
        normal_attention.extend(attention_weights[i, normal_indices].tolist())
    
    noise_attention = np.array(noise_attention)
    normal_attention = np.array(normal_attention)
    
    # ヒストグラム
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(normal_attention, bins=50, alpha=0.7, label='正常区間', color='blue')
    ax.hist(noise_attention, bins=50, alpha=0.7, label='ノイズ区間', color='red')
    ax.set_xlabel('アテンションウェイト')
    ax.set_ylabel('頻度')
    ax.set_title('アテンションウェイトの分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_noise_detection_reconstruction_model(
    model, dataloader, device='cuda', num_intervals=30, points_per_interval=100
):
    """
    ノイズ検出 + 復元モデルの評価
    
    評価項目:
    1. マスク予測精度（どの区間にノイズがあるか）
    2. 復元精度（真のノイズ区間のみで評価）
    
    Args:
        model: ノイズ検出 + 復元モデル（NoiseDetectionAndReconstructionModel）
        dataloader: データローダー
        device: デバイス
        num_intervals: 区間数（デフォルト: 30）
        points_per_interval: 1区間あたりのポイント数（デフォルト: 100）
    
    Returns:
        dict: 評価指標の辞書
    """
    model.eval()
    
    all_mask_predictions = []
    all_mask_labels = []
    all_reconstruction_losses = []
    all_reconstruction_accuracies = []  # 復元精度（%）
    all_mask_correct = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                noisy_psd = batch['noisy_psd'].to(device)
                original_psd = batch['original_psd'].to(device)  # ノイズ付与前
                true_noise_intervals = batch['noise_intervals'].to(device)  # 真のノイズ区間
            else:
                # タプルの場合: (noisy_psd, original_psd, true_noise_intervals)
                noisy_psd, original_psd, true_noise_intervals = batch
                noisy_psd = noisy_psd.to(device)
                original_psd = original_psd.to(device)
                true_noise_intervals = true_noise_intervals.to(device)
            
            # モデルの出力
            predicted_mask, reconstructed_interval = model(noisy_psd)
            # predicted_mask: (batch_size, 30) - 各区間のノイズ強度
            # reconstructed_interval: (batch_size, 100) - 予測した区間のみ復元
            
            batch_size = noisy_psd.size(0)
            
            # 1. マスク予測の評価
            # 予測された区間（最も確信度の高い区間）
            predicted_intervals = predicted_mask.argmax(dim=1)  # (batch_size,)
            
            # マスク予測が正しいかチェック
            for i in range(batch_size):
                pred_interval = predicted_intervals[i].item()
                true_interval = true_noise_intervals[i].item()
                is_correct = (pred_interval == true_interval)
                all_mask_correct.append(is_correct)
                
                all_mask_predictions.append(pred_interval)
                all_mask_labels.append(true_interval)
            
            # 2. 復元精度の評価
            # reconstructed_intervalは予測した区間のみ（100ポイント）
            for i in range(batch_size):
                pred_interval = predicted_intervals[i].item()
                true_interval = true_noise_intervals[i].item()
                
                # 予測した区間が真のノイズ区間と一致する場合のみ復元損失を計算
                if pred_interval == true_interval:
                    # 真のノイズ区間のインデックス範囲を計算
                    start_idx = true_interval * points_per_interval
                    end_idx = min(start_idx + points_per_interval, original_psd.size(1))
                    
                    # 予測した区間の復元精度を計算
                    pred_interval_reconstructed = reconstructed_interval[i]  # (100,)
                    true_interval_original = original_psd[i, start_idx:end_idx]  # (100,)
                    
                    # MSE損失
                    interval_loss = F.mse_loss(pred_interval_reconstructed, true_interval_original)
                    all_reconstruction_losses.append(interval_loss.item())
                    
                    # 復元精度（%）を計算
                    # 正規化されたデータでは相対誤差が厳しすぎるため、
                    # MSE損失から復元精度を計算する方法に変更
                    # 復元精度 = max(0, (1 - sqrt(MSE) / std) * 100)
                    # std: データの標準偏差（約0.82）
                    mse_loss = interval_loss.item()
                    data_std = 0.82  # 正規化後の標準偏差（約0.82）
                    rmse = np.sqrt(mse_loss)
                    relative_rmse = rmse / data_std
                    reconstruction_accuracy = max(0.0, (1.0 - relative_rmse) * 100.0)
                    all_reconstruction_accuracies.append(reconstruction_accuracy)
                # 予測が間違っている場合は復元損失を計算しない（アプローチ2に従う）
    
    # 評価指標の計算
    all_mask_predictions = np.array(all_mask_predictions)
    all_mask_labels = np.array(all_mask_labels)
    
    # 1. マスク予測精度
    mask_accuracy = accuracy_score(all_mask_labels, all_mask_predictions)
    mask_precision = precision_score(all_mask_labels, all_mask_predictions, average='macro', zero_division=0)
    mask_recall = recall_score(all_mask_labels, all_mask_predictions, average='macro', zero_division=0)
    mask_f1 = f1_score(all_mask_labels, all_mask_predictions, average='macro', zero_division=0)
    mask_cm = confusion_matrix(all_mask_labels, all_mask_predictions)
    
    # 2. 復元精度（マスク予測が正しい場合のみ）
    # all_reconstruction_lossesには、マスク予測が正しい場合のみ復元損失が含まれる
    if len(all_reconstruction_losses) > 0:
        overall_reconstruction_loss = np.mean(all_reconstruction_losses)
        overall_reconstruction_accuracy = np.mean(all_reconstruction_accuracies)  # 平均復元精度（%）
    else:
        overall_reconstruction_loss = None
        overall_reconstruction_accuracy = None
    
    # マスク予測が正しい場合と間違った場合の復元精度
    correct_indices = [i for i, correct in enumerate(all_mask_correct) if correct]
    incorrect_indices = [i for i, correct in enumerate(all_mask_correct) if not correct]
    
    if len(correct_indices) > 0 and len(all_reconstruction_losses) > 0:
        # all_reconstruction_lossesのインデックスは、マスク予測が正しいサンプルのみに対応
        reconstruction_loss_when_correct = overall_reconstruction_loss
    else:
        reconstruction_loss_when_correct = None
    
    # マスク予測が間違った場合は復元損失を計算しない（アプローチ2）
    reconstruction_loss_when_incorrect = None
    
    # 3. 主要な損失: 復元損失（MSE損失）
    # 復元損失が最も重要なので、'loss'キーに復元損失を設定
    # マスク予測が正しい場合のみ復元損失を計算（アプローチ2）
    primary_loss = overall_reconstruction_loss if overall_reconstruction_loss is not None else float('inf')
    
    return {
        'accuracy': mask_accuracy,  # baseline/self_supervisedと共通のキー名
        'precision': mask_precision,  # baseline/self_supervisedと共通のキー名
        'recall': mask_recall,  # baseline/self_supervisedと共通のキー名
        'f1_score': mask_f1,  # baseline/self_supervisedと共通のキー名
        'loss': primary_loss,  # 復元損失（MSE損失）- 最も重要な指標
        'mask_accuracy': mask_accuracy,  # 後方互換性のため残す
        'mask_precision': mask_precision,
        'mask_recall': mask_recall,
        'mask_f1_score': mask_f1,
        'mask_confusion_matrix': mask_cm,
        'overall_reconstruction_loss': overall_reconstruction_loss,  # 復元損失（MSE損失）
        'reconstruction_accuracy': overall_reconstruction_accuracy,  # 復元精度（%）
        'reconstruction_loss_when_correct': reconstruction_loss_when_correct,
        'reconstruction_loss_when_incorrect': reconstruction_loss_when_incorrect,
        'num_correct_masks': len(correct_indices),
        'num_incorrect_masks': len(incorrect_indices),
        'predictions': all_mask_predictions,
        'labels': all_mask_labels,
        'reconstruction_losses': all_reconstruction_losses,
        'reconstruction_accuracies': all_reconstruction_accuracies,  # 各サンプルの復元精度（%）
    }


def compare_methods(baseline_results, ssl_results):
    """
    2つの手法の結果を比較
    
    Args:
        baseline_results: ベースラインの評価結果
        ssl_results: 自己教師あり学習の評価結果
    
    Returns:
        dict: 比較結果
    """
    comparison = {
        'accuracy_diff': ssl_results['accuracy'] - baseline_results['accuracy'],
        'f1_diff': ssl_results['f1_score'] - baseline_results['f1_score'],
        'baseline': baseline_results,
        'self_supervised': ssl_results
    }
    
    print("=" * 60)
    print("手法の比較結果")
    print("=" * 60)
    print(f"\nベースライン（畳み込み）:")
    print(f"  Accuracy: {baseline_results['accuracy']:.4f}")
    print(f"  Precision: {baseline_results['precision']:.4f}")
    print(f"  Recall: {baseline_results['recall']:.4f}")
    print(f"  F1-score: {baseline_results['f1_score']:.4f}")
    if 'loss' in baseline_results:
        print(f"  Loss (CrossEntropyLoss): {baseline_results['loss']:.6f}")
    
    print(f"\n自己教師あり学習（タスク4）:")
    print(f"  Accuracy: {ssl_results['accuracy']:.4f}")
    print(f"  Precision: {ssl_results['precision']:.4f}")
    print(f"  Recall: {ssl_results['recall']:.4f}")
    print(f"  F1-score: {ssl_results['f1_score']:.4f}")
    if 'loss' in ssl_results:
        print(f"  Loss (CrossEntropyLoss): {ssl_results['loss']:.6f}")
    if 'roc_auc' in ssl_results:
        print(f"  ROC-AUC: {ssl_results['roc_auc']:.4f}")
    if 'attention_diff' in ssl_results:
        print(f"  アテンションウェイトの差: {ssl_results['attention_diff']:.4f}")
        print(f"    (正常区間 - ノイズ区間)")
    if 'best_threshold' in ssl_results:
        print(f"  最適閾値: {ssl_results['best_threshold']:.6f}")
    if 'noise_attention_mean' in ssl_results:
        print(f"  ノイズ区間の平均アテンション: {ssl_results['noise_attention_mean']:.6f}")
    if 'normal_attention_mean' in ssl_results:
        print(f"  正常区間の平均アテンション: {ssl_results['normal_attention_mean']:.6f}")
    
    print(f"\n改善度:")
    print(f"  Accuracy: {comparison['accuracy_diff']:+.4f} ({comparison['accuracy_diff']/baseline_results['accuracy']*100:+.2f}%)")
    print(f"  F1-score: {comparison['f1_diff']:+.4f} ({comparison['f1_diff']/baseline_results['f1_score']*100:+.2f}%)")
    
    return comparison

