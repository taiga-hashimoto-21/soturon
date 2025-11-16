"""
ノイズ検出の評価関数
愚直な手法と自己教師あり学習の両方に対応
"""

import torch
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


def evaluate_baseline_model(predictions, labels):
    """
    愚直な手法（30クラス分類）の評価
    
    Args:
        predictions: 予測された区間番号 (batch_size,)
        labels: 正解の区間番号 (batch_size,)
    
    Returns:
        dict: 評価指標の辞書
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Precision, Recall, F1-score（マクロ平均）
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    # 混同行列
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }


def evaluate_attention_weights(attention_weights, labels, num_intervals=30):
    """
    自己教師あり学習（アテンションウェイト）の評価
    
    Args:
        attention_weights: アテンションウェイト (batch_size, seq_len) または (batch_size, num_intervals)
        labels: 正解の区間番号 (batch_size,)
        num_intervals: 区間数
    
    Returns:
        dict: 評価指標の辞書
    """
    attention_weights = np.array(attention_weights)
    labels = np.array(labels)
    
    # アテンションウェイトを区間ごとに平均化
    if attention_weights.shape[1] != num_intervals:
        # seq_lenをnum_intervalsに変換（平均プーリング）
        points_per_interval = attention_weights.shape[1] // num_intervals
        interval_attention = []
        for i in range(num_intervals):
            start_idx = i * points_per_interval
            end_idx = start_idx + points_per_interval
            interval_attention.append(attention_weights[:, start_idx:end_idx].mean(axis=1))
        interval_attention = np.array(interval_attention).T  # (batch_size, num_intervals)
    else:
        interval_attention = attention_weights
    
    # ノイズ区間と正常区間のアテンションウェイトを比較
    noise_attention = []
    normal_attention = []
    for i, label in enumerate(labels):
        noise_attention.append(interval_attention[i, label])
        normal_indices = [j for j in range(num_intervals) if j != label]
        normal_attention.append(interval_attention[i, normal_indices].mean())
    
    noise_attention = np.array(noise_attention)
    normal_attention = np.array(normal_attention)
    
    # アテンションウェイトの平均値比較
    attention_diff = normal_attention.mean() - noise_attention.mean()
    
    # 閾値を最適化（F1-scoreが最大になる閾値を選択）
    thresholds = np.linspace(interval_attention.min(), interval_attention.max(), 100)
    best_threshold = None
    best_f1 = 0
    best_accuracy = 0
    
    for threshold in thresholds:
        # 各サンプルで、閾値以下の区間を「ノイズあり」と判定
        predictions = []
        for i in range(len(labels)):
            noisy_intervals = np.where(interval_attention[i] < threshold)[0]
            if len(noisy_intervals) > 0:
                # 最もアテンションウェイトが低い区間を予測
                predictions.append(noisy_intervals[np.argmin(interval_attention[i, noisy_intervals])])
            else:
                # 閾値以下の区間がない場合、最もアテンションウェイトが低い区間を予測
                predictions.append(np.argmin(interval_attention[i]))
        
        predictions = np.array(predictions)
        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        accuracy = accuracy_score(labels, predictions)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_accuracy = accuracy
    
    # 最適な閾値で最終予測
    final_predictions = []
    for i in range(len(labels)):
        noisy_intervals = np.where(interval_attention[i] < best_threshold)[0]
        if len(noisy_intervals) > 0:
            final_predictions.append(noisy_intervals[np.argmin(interval_attention[i, noisy_intervals])])
        else:
            final_predictions.append(np.argmin(interval_attention[i]))
    final_predictions = np.array(final_predictions)
    
    # Precision, Recall
    precision = precision_score(labels, final_predictions, average='macro', zero_division=0)
    recall = recall_score(labels, final_predictions, average='macro', zero_division=0)
    
    # ROC-AUC（二値分類として扱う）
    # 各サンプルについて、ノイズ区間のアテンションウェイトを1、それ以外を0として扱う
    y_true_binary = []
    y_score_binary = []
    for i, label in enumerate(labels):
        y_true_binary.append(1)  # ノイズ区間
        y_score_binary.append(1 - interval_attention[i, label])  # アテンションウェイトが低いほどスコアが高い
        for j in range(num_intervals):
            if j != label:
                y_true_binary.append(0)  # 正常区間
                y_score_binary.append(1 - interval_attention[i, j])
    
    try:
        roc_auc = roc_auc_score(y_true_binary, y_score_binary)
    except:
        roc_auc = 0.0
    
    # 混同行列
    cm = confusion_matrix(labels, final_predictions)
    
    return {
        'accuracy': best_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': best_f1,
        'roc_auc': roc_auc,
        'attention_diff': attention_diff,  # 正常区間 - ノイズ区間のアテンションウェイトの差
        'best_threshold': best_threshold,
        'confusion_matrix': cm,
        'noise_attention_mean': noise_attention.mean(),
        'normal_attention_mean': normal_attention.mean(),
        'interval_attention': interval_attention  # 後で可視化用
    }


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
    
    # 区間ごとに平均化
    if attention_weights.shape[1] != num_intervals:
        points_per_interval = attention_weights.shape[1] // num_intervals
        interval_attention = []
        for i in range(num_intervals):
            start_idx = i * points_per_interval
            end_idx = start_idx + points_per_interval
            interval_attention.append(attention_weights[:, start_idx:end_idx].mean(axis=1))
        interval_attention = np.array(interval_attention).T
    else:
        interval_attention = attention_weights
    
    # ノイズ区間と正常区間のアテンションウェイトを分離
    noise_attention = []
    normal_attention = []
    for i, label in enumerate(labels):
        noise_attention.append(interval_attention[i, label])
        normal_indices = [j for j in range(num_intervals) if j != label]
        normal_attention.extend(interval_attention[i, normal_indices].tolist())
    
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


def compare_methods(baseline_results, attention_results):
    """
    2つの手法の結果を比較
    
    Args:
        baseline_results: 愚直な手法の評価結果
        attention_results: 自己教師あり学習の評価結果
    
    Returns:
        dict: 比較結果
    """
    comparison = {
        'accuracy_diff': attention_results['accuracy'] - baseline_results['accuracy'],
        'f1_diff': attention_results['f1_score'] - baseline_results['f1_score'],
        'baseline': baseline_results,
        'attention': attention_results
    }
    
    print("=" * 60)
    print("手法の比較結果")
    print("=" * 60)
    print(f"\n愚直な手法:")
    print(f"  Accuracy: {baseline_results['accuracy']:.4f}")
    print(f"  F1-score: {baseline_results['f1_score']:.4f}")
    
    print(f"\n自己教師あり学習:")
    print(f"  Accuracy: {attention_results['accuracy']:.4f}")
    print(f"  F1-score: {attention_results['f1_score']:.4f}")
    print(f"  ROC-AUC: {attention_results['roc_auc']:.4f}")
    print(f"  アテンションウェイトの差: {attention_results['attention_diff']:.4f}")
    print(f"    (正常区間 - ノイズ区間)")
    
    print(f"\n改善度:")
    print(f"  Accuracy: {comparison['accuracy_diff']:+.4f} ({comparison['accuracy_diff']/baseline_results['accuracy']*100:+.2f}%)")
    print(f"  F1-score: {comparison['f1_diff']:+.4f} ({comparison['f1_diff']/baseline_results['f1_score']*100:+.2f}%)")
    
    return comparison


