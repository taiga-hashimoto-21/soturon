"""
愚直な手法（30クラス分類）の学習スクリプト
"""

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from baseline_model import SimpleCNN
from evaluate_noise_detection import evaluate_baseline_model, plot_confusion_matrix
import matplotlib.pyplot as plt
import platform

# 日本語フォントの設定
if platform.system() == 'Darwin':  # Mac
    plt.rcParams['font.family'] = 'Hiragino Sans'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class PSDDataset(Dataset):
    """PSDデータセット"""
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_epoch(model, dataloader, criterion, optimizer, device):
    """1エポックの学習"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, labels in dataloader:
        data = data.to(device)
        labels = labels.to(device)
        
        # 順伝播
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # 逆伝播
        loss.backward()
        optimizer.step()
        
        # 統計
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """検証"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, all_predictions, all_labels


def main():
    # ハイパーパラメータ
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    NUM_CLASSES = 30
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データセットの読み込み
    print("\nデータセットを読み込み中...")
    with open('baseline_dataset.pickle', 'rb') as f:
        dataset = pickle.load(f)
    
    train_data = dataset['train']['data']
    train_labels = dataset['train']['labels']
    val_data = dataset['val']['data']
    val_labels = dataset['val']['labels']
    test_data = dataset['test']['data']
    test_labels = dataset['test']['labels']
    
    print(f"訓練データ: {len(train_data):,}サンプル")
    print(f"検証データ: {len(val_data):,}サンプル")
    print(f"テストデータ: {len(test_data):,}サンプル")
    
    # DataLoaderの作成
    train_dataset = PSDDataset(train_data, train_labels)
    val_dataset = PSDDataset(val_data, val_labels)
    test_dataset = PSDDataset(test_data, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # モデルの作成
    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    print(f"\nモデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 損失関数とオプティマイザ
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 学習履歴
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0
    best_model_state = None
    
    print("\n学習開始...")
    print("=" * 60)
    
    for epoch in range(NUM_EPOCHS):
        # 学習
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 検証
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # 学習率の調整
        scheduler.step(val_loss)
        
        # 履歴の保存
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # ベストモデルの保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # ログ出力
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print()
    
    # ベストモデルをロード
    model.load_state_dict(best_model_state)
    print(f"ベスト検証精度: {best_val_acc:.2f}%")
    
    # テストデータで評価
    print("\nテストデータで評価中...")
    test_loss, test_acc, test_predictions, test_labels_list = validate(
        model, test_loader, criterion, device
    )
    
    print(f"テスト精度: {test_acc:.2f}%")
    
    # 詳細な評価
    results = evaluate_baseline_model(test_predictions, test_labels_list)
    print(f"\n詳細な評価結果:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1-score: {results['f1_score']:.4f}")
    
    # 混同行列を可視化
    plot_confusion_matrix(
        results['confusion_matrix'],
        title='混同行列（テストデータ）',
        save_path='baseline_confusion_matrix.png'
    )
    print("\n混同行列を 'baseline_confusion_matrix.png' に保存しました")
    
    # 学習曲線を可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_losses, label='訓練損失', color='blue')
    axes[0].plot(val_losses, label='検証損失', color='red')
    axes[0].set_xlabel('エポック')
    axes[0].set_ylabel('損失')
    axes[0].set_title('学習曲線（損失）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accuracies, label='訓練精度', color='blue')
    axes[1].plot(val_accuracies, label='検証精度', color='red')
    axes[1].set_xlabel('エポック')
    axes[1].set_ylabel('精度 (%)')
    axes[1].set_title('学習曲線（精度）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baseline_training_curves.png', dpi=150, bbox_inches='tight')
    print("学習曲線を 'baseline_training_curves.png' に保存しました")
    
    # モデルを保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_acc,
        'results': results
    }, 'baseline_model.pth')
    print("モデルを 'baseline_model.pth' に保存しました")
    
    print("\n学習完了！")


if __name__ == '__main__':
    main()

