"""
統一評価コード
ベースライン（畳み込み）と自己教師あり学習（タスク4）の両方に対応
評価時の損失関数を統一（CrossEntropyLoss）
"""

import torch
import torch.nn as nn
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
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'loss': evaluation_loss,  # CrossEntropyLoss
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
            
            # アテンションウェイトを取得
            _, _, attention_weights = model(x, m, return_attention=True)
            
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
                
                if cls_attention is not None:
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
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'loss': evaluation_loss,  # baselineと同じCrossEntropyLoss
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

