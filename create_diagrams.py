"""
自己教師あり学習モデルの研究方法の図解を生成
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
import os

# 日本語フォントの設定
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_data_flow_diagram():
    """データフロー図を作成"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # タイトル
    ax.text(5, 5.5, '自己教師あり学習タスク4: データフロー', 
            fontsize=20, fontweight='bold', ha='center')
    
    # ステップ1: PSDデータ
    box1 = FancyBboxPatch((0.5, 4), 1.5, 0.8, boxstyle="round,pad=0.1", 
                          facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(box1)
    ax.text(1.25, 4.4, 'PSDデータ\n(3000ポイント)', ha='center', va='center', fontsize=11)
    
    # 矢印1
    ax.arrow(2.2, 4.4, 0.6, 0, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # ステップ2: 30区間分割
    box2 = FancyBboxPatch((2.9, 4), 1.5, 0.8, boxstyle="round,pad=0.1", 
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(box2)
    ax.text(3.65, 4.4, '30区間に分割\n(各区間100ポイント)', ha='center', va='center', fontsize=11)
    
    # 矢印2
    ax.arrow(4.5, 4.4, 0.6, 0, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # ステップ3: ノイズ付与
    box3 = FancyBboxPatch((5.2, 4), 1.5, 0.8, boxstyle="round,pad=0.1", 
                          facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(box3)
    ax.text(5.95, 4.4, '1区間にノイズ付与\n(検知対象)', ha='center', va='center', fontsize=11)
    
    # 矢印3（下へ）
    ax.arrow(5.95, 3.9, 0, -0.6, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # ステップ4: マスク
    box4 = FancyBboxPatch((4.7, 2.5), 2.5, 0.8, boxstyle="round,pad=0.1", 
                          facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(box4)
    ax.text(5.95, 2.9, '15%の区間をランダムにマスク\n(学習用)', ha='center', va='center', fontsize=11)
    
    # 矢印4（右へ）
    ax.arrow(7.3, 2.9, 0.6, 0, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # ステップ5: BERTモデル
    box5 = FancyBboxPatch((8, 2), 1.5, 1.8, boxstyle="round,pad=0.1", 
                          facecolor='lightpink', edgecolor='black', linewidth=2)
    ax.add_patch(box5)
    ax.text(8.75, 3.4, 'BERTモデル\n(Transformer)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(8.75, 3.0, '• CLSトークン追加\n• マスク予測\n• アテンション取得', 
            ha='center', va='center', fontsize=9)
    
    # 出力1: マスク予測
    box6 = FancyBboxPatch((0.5, 1), 1.5, 0.8, boxstyle="round,pad=0.1", 
                          facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(box6)
    ax.text(1.25, 1.4, 'マスク予測\n(主タスク)', ha='center', va='center', fontsize=11)
    
    # 出力2: アテンション
    box7 = FancyBboxPatch((2.9, 1), 1.5, 0.8, boxstyle="round,pad=0.1", 
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(box7)
    ax.text(3.65, 1.4, 'アテンション\nウェイト', ha='center', va='center', fontsize=11)
    
    # 出力3: ノイズ検知
    box8 = FancyBboxPatch((5.2, 1), 1.5, 0.8, boxstyle="round,pad=0.1", 
                          facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(box8)
    ax.text(5.95, 1.4, 'ノイズ検知\n(副産物)', ha='center', va='center', fontsize=11)
    
    # 矢印（BERTから出力へ）
    ax.arrow(8, 2.9, -0.6, -0.9, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    ax.arrow(8, 2.9, -1.9, -0.9, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    ax.arrow(8, 2.9, -3.2, -0.9, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    plt.tight_layout()
    output_path = os.path.expanduser('~/Desktop/自己教師あり学習_データフロー.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"データフロー図を保存しました: {output_path}")
    plt.close()


def create_model_architecture_diagram():
    """モデルアーキテクチャ図を作成"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # タイトル
    ax.text(5, 7.5, 'BERTモデルアーキテクチャ', 
            fontsize=20, fontweight='bold', ha='center')
    
    # 入力層
    ax.text(1, 6.5, '入力:', fontsize=14, fontweight='bold')
    input_box = FancyBboxPatch((1, 5.5), 1.5, 0.8, boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.75, 5.9, 'マスクされた\nPSDデータ\n(3000ポイント)', 
            ha='center', va='center', fontsize=10)
    
    # 埋め込み層
    ax.arrow(2.7, 5.9, 0.6, 0, head_width=0.1, head_length=0.08, 
             fc='black', ec='black', linewidth=2)
    embed_box = FancyBboxPatch((3.4, 5.5), 1.5, 0.8, boxstyle="round,pad=0.1", 
                               facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(embed_box)
    ax.text(4.15, 5.9, '埋め込み層\n(Linear)', ha='center', va='center', fontsize=10)
    
    # CLSトークン追加
    ax.arrow(5.1, 5.9, 0.6, 0, head_width=0.1, head_length=0.08, 
             fc='black', ec='black', linewidth=2)
    cls_box = FancyBboxPatch((5.8, 5.5), 1.5, 0.8, boxstyle="round,pad=0.1", 
                             facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(cls_box)
    ax.text(6.55, 5.9, 'CLSトークン\n追加\n(31トークン)', 
            ha='center', va='center', fontsize=10)
    
    # Transformer Encoder
    ax.arrow(7.5, 5.9, 0.3, -1.5, head_width=0.1, head_length=0.08, 
             fc='black', ec='black', linewidth=2)
    transformer_box = FancyBboxPatch((6, 3), 2, 1.5, boxstyle="round,pad=0.1", 
                                      facecolor='lightpink', edgecolor='black', linewidth=2)
    ax.add_patch(transformer_box)
    ax.text(7, 4.2, 'Transformer\nEncoder', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(7, 3.7, '• Multi-Head Attention\n• Feed Forward\n• Layer Normalization', 
            ha='center', va='center', fontsize=9)
    
    # 出力1: マスク予測
    ax.arrow(6, 3.75, -1.2, 0, head_width=0.1, head_length=0.08, 
             fc='black', ec='black', linewidth=2)
    output1_box = FancyBboxPatch((1, 3.3), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(output1_box)
    ax.text(1.75, 3.7, 'マスク予測\n(回帰)', ha='center', va='center', fontsize=10)
    
    # 出力2: CLS表現
    ax.arrow(7, 3.75, 0, -0.8, head_width=0.1, head_length=0.08, 
             fc='black', ec='black', linewidth=2)
    output2_box = FancyBboxPatch((6, 1.7), 2, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(output2_box)
    ax.text(7, 2.1, 'CLSトークンの表現', ha='center', va='center', fontsize=10)
    
    # 出力3: アテンションウェイト
    ax.arrow(8, 3.75, 1.2, 0, head_width=0.1, head_length=0.08, 
             fc='black', ec='black', linewidth=2)
    output3_box = FancyBboxPatch((9.3, 3.3), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(output3_box)
    ax.text(10.05, 3.7, 'アテンション\nウェイト\n(31×31)', 
            ha='center', va='center', fontsize=10)
    
    # アテンションの説明
    ax.text(1, 0.8, 'アテンションウェイトからノイズ区間を検知:', 
            fontsize=12, fontweight='bold')
    ax.text(1, 0.4, '• CLSトークンから各区間へのアテンションを取得', 
            fontsize=10)
    ax.text(1, 0.1, '• アテンションが最も低い区間 = ノイズ区間', 
            fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.expanduser('~/Desktop/自己教師あり学習_モデルアーキテクチャ.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"モデルアーキテクチャ図を保存しました: {output_path}")
    plt.close()


def create_loss_function_diagram():
    """損失関数の図解を作成"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # タイトル
    ax.text(5, 5.5, '損失関数の構成', fontsize=20, fontweight='bold', ha='center')
    
    # 総損失
    total_box = FancyBboxPatch((3.5, 4.2), 3, 0.8, boxstyle="round,pad=0.1", 
                               facecolor='lightcoral', edgecolor='black', linewidth=3)
    ax.add_patch(total_box)
    ax.text(5, 4.6, '総損失 = マスク予測損失 + λ × ランキング損失', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # マスク予測損失
    ax.arrow(5, 4.2, -1.5, -1, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    mask_box = FancyBboxPatch((1, 2.5), 2, 1, boxstyle="round,pad=0.1", 
                              facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(mask_box)
    ax.text(2, 3.2, 'マスク予測損失', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(2, 2.9, 'MSE(予測, 正解)', ha='center', va='center', fontsize=10)
    ax.text(2, 2.6, '主タスク: データ構造の学習', ha='center', va='center', fontsize=9)
    
    # ランキング損失
    ax.arrow(5, 4.2, 1.5, -1, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    rank_box = FancyBboxPatch((7, 2.5), 2, 1, boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(rank_box)
    ax.text(8, 3.2, 'ランキング損失', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(8, 2.9, 'max(m, normal - noise)', ha='center', va='center', fontsize=10)
    ax.text(8, 2.6, '副タスク: ノイズ検知', ha='center', va='center', fontsize=9)
    
    # 数式の説明
    ax.text(5, 1.8, 'ランキング損失の詳細:', fontsize=12, fontweight='bold', ha='center')
    ax.text(1, 1.3, '目的: ノイズ区間のアテンション < 正常区間のアテンション', 
            fontsize=10, ha='left')
    ax.text(1, 1.0, '• normal_attn - noise_attn > m になるように学習', 
            fontsize=10, ha='left')
    ax.text(1, 0.7, '• m: マージン（最小の差を保証）', fontsize=10, ha='left')
    ax.text(1, 0.4, '• λ: 正則化項の重み（lambda_reg）', fontsize=10, ha='left')
    
    plt.tight_layout()
    output_path = os.path.expanduser('~/Desktop/自己教師あり学習_損失関数.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"損失関数図を保存しました: {output_path}")
    plt.close()


def create_learning_process_diagram():
    """学習プロセスの図解を作成"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # タイトル
    ax.text(5, 5.5, '学習プロセス', fontsize=20, fontweight='bold', ha='center')
    
    # ステップ1: データ準備
    step1_box = FancyBboxPatch((0.5, 4), 2, 0.8, boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(step1_box)
    ax.text(1.5, 4.4, '1. データ準備', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(1.5, 4.1, '• 30区間に分割\n• ノイズ付与\n• マスク作成', 
            ha='center', va='center', fontsize=9)
    
    # 矢印1
    ax.arrow(2.6, 4.4, 0.6, 0, head_width=0.1, head_length=0.08, 
             fc='black', ec='black', linewidth=2)
    
    # ステップ2: モデル処理
    step2_box = FancyBboxPatch((3.3, 4), 2, 0.8, boxstyle="round,pad=0.1", 
                               facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(step2_box)
    ax.text(4.3, 4.4, '2. BERTで処理', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(4.3, 4.1, '• マスク予測\n• アテンション取得', 
            ha='center', va='center', fontsize=9)
    
    # 矢印2
    ax.arrow(5.4, 4.4, 0.6, 0, head_width=0.1, head_length=0.08, 
             fc='black', ec='black', linewidth=2)
    
    # ステップ3: 損失計算
    step3_box = FancyBboxPatch((6.1, 4), 2, 0.8, boxstyle="round,pad=0.1", 
                               facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(step3_box)
    ax.text(7.1, 4.4, '3. 損失計算', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7.1, 4.1, '• マスク予測損失\n• ランキング損失', 
            ha='center', va='center', fontsize=9)
    
    # 矢印3（下へ）
    ax.arrow(7.1, 3.9, 0, -0.6, head_width=0.1, head_length=0.08, 
             fc='black', ec='black', linewidth=2)
    
    # ステップ4: 逆伝播
    step4_box = FancyBboxPatch((6.1, 2.5), 2, 0.8, boxstyle="round,pad=0.1", 
                               facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(step4_box)
    ax.text(7.1, 2.9, '4. 逆伝播', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7.1, 2.6, '• パラメータ更新', ha='center', va='center', fontsize=9)
    
    # 矢印4（左へ、ループ）
    ax.arrow(6.1, 2.9, -0.6, 1.1, head_width=0.1, head_length=0.08, 
             fc='black', ec='black', linewidth=2)
    
    # 学習の目標
    goal_box = FancyBboxPatch((0.5, 1), 4, 1, boxstyle="round,pad=0.1", 
                              facecolor='lightpink', edgecolor='black', linewidth=2)
    ax.add_patch(goal_box)
    ax.text(2.5, 1.6, '学習の目標:', fontsize=12, fontweight='bold', ha='center')
    ax.text(2.5, 1.3, '• マスクされた区間を正確に復元', ha='center', va='center', fontsize=10)
    ax.text(2.5, 1.0, '• ノイズ区間のアテンション < 正常区間のアテンション', 
            ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.expanduser('~/Desktop/自己教師あり学習_学習プロセス.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"学習プロセス図を保存しました: {output_path}")
    plt.close()


def create_comparison_diagram():
    """ベースラインとの比較図を作成"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # タイトル
    ax.text(5, 5.5, 'ベースライン vs 自己教師あり学習', 
            fontsize=20, fontweight='bold', ha='center')
    
    # ベースライン
    baseline_box = FancyBboxPatch((0.5, 3.5), 4, 1.5, boxstyle="round,pad=0.1", 
                                  facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(baseline_box)
    ax.text(2.5, 4.8, 'ベースライン（畳み込みモデル）', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(2.5, 4.3, '• CNNベース（1D畳み込み + ResNet）', 
            ha='center', va='center', fontsize=11)
    ax.text(2.5, 4.0, '• 教師あり学習（30クラス分類）', 
            ha='center', va='center', fontsize=11)
    ax.text(2.5, 3.7, '• ラベル（ノイズ区間のインデックス）を直接使用', 
            ha='center', va='center', fontsize=11)
    
    # 自己教師あり学習
    ssl_box = FancyBboxPatch((5.5, 3.5), 4, 1.5, boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(ssl_box)
    ax.text(7.5, 4.8, '自己教師あり学習（タスク4）', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(7.5, 4.3, '• BERTベース（Transformer）', 
            ha='center', va='center', fontsize=11)
    ax.text(7.5, 4.0, '• 自己教師あり学習（マスク予測）', 
            ha='center', va='center', fontsize=11)
    ax.text(7.5, 3.7, '• ラベルを使わず、入力データのみで学習', 
            ha='center', va='center', fontsize=11)
    
    # 共通点
    common_box = FancyBboxPatch((0.5, 1.5), 9, 1, boxstyle="round,pad=0.1", 
                               facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(common_box)
    ax.text(5, 2.2, '共通点:', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 1.9, '• 同じデータセット（data_lowF_noise.pickle）', 
            ha='center', va='center', fontsize=11)
    ax.text(5, 1.6, '• 同じノイズ生成方法（3種類のノイズパターン）', 
            ha='center', va='center', fontsize=11)
    ax.text(5, 1.3, '• 同じ評価指標（Accuracy, F1-score, CrossEntropyLoss）', 
            ha='center', va='center', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.expanduser('~/Desktop/自己教師あり学習_比較.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"比較図を保存しました: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("図解を生成中...")
    create_data_flow_diagram()
    create_model_architecture_diagram()
    create_loss_function_diagram()
    create_learning_process_diagram()
    create_comparison_diagram()
    print("\nすべての図解をデスクトップに保存しました！")

