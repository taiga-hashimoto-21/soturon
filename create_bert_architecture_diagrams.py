"""
BERTベースのTransformerモデルのアーキテクチャ説明用図解を作成
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from matplotlib import rcParams

# 日本語フォントの設定
import platform
if platform.system() == 'Darwin':  # macOS
    rcParams['font.family'] = 'Hiragino Sans'
elif platform.system() == 'Windows':
    rcParams['font.family'] = 'MS Gothic'
else:
    rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 12

# 図1: Transformerとの関係
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 10)
ax1.axis('off')

# Transformer全体
transformer_box = FancyBboxPatch((1, 6), 10, 3, 
                                  boxstyle="round,pad=0.3", 
                                  edgecolor='black', 
                                  facecolor='lightblue',
                                  linewidth=2)
ax1.add_patch(transformer_box)
ax1.text(6, 7.5, 'Transformer', ha='center', va='center', fontsize=16, fontweight='bold')

# Encoder
encoder_box = FancyBboxPatch((2, 7), 4, 1.5, 
                              boxstyle="round,pad=0.2", 
                              edgecolor='darkblue', 
                              facecolor='lightcyan',
                              linewidth=2)
ax1.add_patch(encoder_box)
ax1.text(4, 7.75, 'Encoder', ha='center', va='center', fontsize=14, fontweight='bold')

# Decoder
decoder_box = FancyBboxPatch((7, 7), 4, 1.5, 
                              boxstyle="round,pad=0.2", 
                              edgecolor='darkgreen', 
                              facecolor='lightgreen',
                              linewidth=2)
ax1.add_patch(decoder_box)
ax1.text(9, 7.75, 'Decoder', ha='center', va='center', fontsize=14, fontweight='bold')

# BERT
bert_box = FancyBboxPatch((1, 2), 10, 3, 
                          boxstyle="round,pad=0.3", 
                          edgecolor='red', 
                          facecolor='lightyellow',
                          linewidth=3)
ax1.add_patch(bert_box)
ax1.text(6, 3.5, 'BERT = Encoder部分のみ', ha='center', va='center', fontsize=16, fontweight='bold', color='red')

# 矢印
arrow1 = FancyArrowPatch((6, 6), (6, 5), 
                         arrowstyle='->', 
                         mutation_scale=30, 
                         linewidth=3, 
                         color='red')
ax1.add_patch(arrow1)
ax1.text(6.5, 5.5, '使用', ha='left', va='center', fontsize=12, color='red', fontweight='bold')

# この研究
research_box = FancyBboxPatch((4, 0.5), 4, 1, 
                              boxstyle="round,pad=0.2", 
                              edgecolor='purple', 
                              facecolor='lavender',
                              linewidth=2)
ax1.add_patch(research_box)
ax1.text(6, 1, 'この研究\nBERTベース', ha='center', va='center', fontsize=12, fontweight='bold')

arrow2 = FancyArrowPatch((6, 2), (6, 1.5), 
                         arrowstyle='->', 
                         mutation_scale=20, 
                         linewidth=2, 
                         color='purple')
ax1.add_patch(arrow2)

ax1.set_title('TransformerとBERTの関係', fontsize=18, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/Users/hashimototaiga/Desktop/transformer_bert_relation.png', dpi=300, bbox_inches='tight')
print("図1を保存しました: /Users/hashimototaiga/Desktop/transformer_bert_relation.png")

# 図2: モデル構造図
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 12))
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 13)
ax2.axis('off')

y_positions = [12, 10.5, 9, 7.5, 6, 4.5, 3, 1.5]
box_width = 8
box_height = 1.0

# 中央のx座標
center_x = 5

# 1. 入力PSDデータ
input_box = FancyBboxPatch((center_x - box_width/2, y_positions[0]-box_height/2), box_width, box_height, 
                           boxstyle="round,pad=0.2", 
                           edgecolor='blue', 
                           facecolor='lightblue',
                           linewidth=2)
ax2.add_patch(input_box)
ax2.text(center_x, y_positions[0], '入力PSDデータ\n(3000ポイント)', ha='center', va='center', fontsize=11, fontweight='bold')

# 矢印1
arrow1 = FancyArrowPatch((center_x, y_positions[0]-box_height/2), (center_x, y_positions[1]+box_height/2), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax2.add_patch(arrow1)

# 2. 埋め込み層
embed_box = FancyBboxPatch((center_x - box_width/2, y_positions[1]-box_height/2), box_width, box_height, 
                           boxstyle="round,pad=0.2", 
                           edgecolor='green', 
                           facecolor='lightgreen',
                           linewidth=2)
ax2.add_patch(embed_box)
ax2.text(center_x, y_positions[1], '埋め込み層\n(各ポイント → 64次元)', ha='center', va='center', fontsize=11, fontweight='bold')

# 矢印2
arrow2 = FancyArrowPatch((center_x, y_positions[1]-box_height/2), (center_x, y_positions[2]+box_height/2), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax2.add_patch(arrow2)

# 3. マスクトークン
mask_box = FancyBboxPatch((center_x - box_width/2, y_positions[2]-box_height/2), box_width, box_height, 
                          boxstyle="round,pad=0.2", 
                          edgecolor='orange', 
                          facecolor='lightyellow',
                          linewidth=2)
ax2.add_patch(mask_box)
ax2.text(center_x, y_positions[2], 'マスクトークンで置き換え', ha='center', va='center', fontsize=11, fontweight='bold')

# 矢印3
arrow3 = FancyArrowPatch((center_x, y_positions[2]-box_height/2), (center_x, y_positions[3]+box_height/2), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax2.add_patch(arrow3)

# 4. 位置埋め込み
pos_box = FancyBboxPatch((center_x - box_width/2, y_positions[3]-box_height/2), box_width, box_height, 
                         boxstyle="round,pad=0.2", 
                         edgecolor='purple', 
                         facecolor='lavender',
                         linewidth=2)
ax2.add_patch(pos_box)
ax2.text(center_x, y_positions[3], '位置埋め込み追加', ha='center', va='center', fontsize=11, fontweight='bold')

# 矢印4
arrow4 = FancyArrowPatch((center_x, y_positions[3]-box_height/2), (center_x, y_positions[4]+box_height/2), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax2.add_patch(arrow4)

# 5. CLSトークン
cls_box = FancyBboxPatch((center_x - box_width/2, y_positions[4]-box_height/2), box_width, box_height, 
                         boxstyle="round,pad=0.2", 
                         edgecolor='red', 
                         facecolor='mistyrose',
                         linewidth=2)
ax2.add_patch(cls_box)
ax2.text(center_x, y_positions[4], 'CLSトークン追加\n→ 3001トークン', ha='center', va='center', fontsize=11, fontweight='bold')

# 矢印5
arrow5 = FancyArrowPatch((center_x, y_positions[4]-box_height/2), (center_x, y_positions[5]+box_height/2), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax2.add_patch(arrow5)

# 6. Transformer Encoder
transformer_box = FancyBboxPatch((center_x - box_width/2, y_positions[5]-box_height/2), box_width, box_height, 
                                 boxstyle="round,pad=0.2", 
                                 edgecolor='darkblue', 
                                 facecolor='lightcyan',
                                 linewidth=3)
ax2.add_patch(transformer_box)
ax2.text(center_x, y_positions[5], 'Transformer Encoder\n(2層)', ha='center', va='center', fontsize=11, fontweight='bold')

# Transformer Encoderの詳細（横に並べる）
detail_y = y_positions[5] - 0.8
detail_height = 0.5
detail_width = 2.3
detail_spacing = 0.3

detail_x_start = center_x - (detail_width * 3 + detail_spacing * 2) / 2

detail1 = FancyBboxPatch((detail_x_start, detail_y), detail_width, detail_height, 
                         boxstyle="round,pad=0.1", 
                         edgecolor='darkblue', 
                         facecolor='white',
                         linewidth=1.5)
ax2.add_patch(detail1)
ax2.text(detail_x_start + detail_width/2, detail_y + detail_height/2, 'Multi-Head\nAttention', ha='center', va='center', fontsize=9)

detail2 = FancyBboxPatch((detail_x_start + detail_width + detail_spacing, detail_y), detail_width, detail_height, 
                         boxstyle="round,pad=0.1", 
                         edgecolor='darkblue', 
                         facecolor='white',
                         linewidth=1.5)
ax2.add_patch(detail2)
ax2.text(detail_x_start + detail_width + detail_spacing + detail_width/2, detail_y + detail_height/2, 'Feedforward\nNetwork', ha='center', va='center', fontsize=9)

detail3 = FancyBboxPatch((detail_x_start + (detail_width + detail_spacing) * 2, detail_y), detail_width, detail_height, 
                         boxstyle="round,pad=0.1", 
                         edgecolor='darkblue', 
                         facecolor='white',
                         linewidth=1.5)
ax2.add_patch(detail3)
ax2.text(detail_x_start + (detail_width + detail_spacing) * 2 + detail_width/2, detail_y + detail_height/2, 'Layer\nNormalization', ha='center', va='center', fontsize=9)

# 矢印6
arrow6 = FancyArrowPatch((center_x, y_positions[5]-box_height/2), (center_x, y_positions[6]+box_height/2), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax2.add_patch(arrow6)

# 7. 出力ヘッド
output_box = FancyBboxPatch((center_x - box_width/2, y_positions[6]-box_height/2), box_width, box_height, 
                            boxstyle="round,pad=0.2", 
                            edgecolor='darkgreen', 
                            facecolor='lightgreen',
                            linewidth=2)
ax2.add_patch(output_box)
ax2.text(center_x, y_positions[6], '出力ヘッド\n(各位置の値を予測)', ha='center', va='center', fontsize=11, fontweight='bold')

# 矢印7
arrow7 = FancyArrowPatch((center_x, y_positions[6]-box_height/2), (center_x, y_positions[7]+box_height/2), 
                         arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax2.add_patch(arrow7)

# 8. 出力
final_box = FancyBboxPatch((center_x - box_width/2, y_positions[7]-box_height/2), box_width, box_height, 
                           boxstyle="round,pad=0.2", 
                           edgecolor='darkred', 
                           facecolor='lightcoral',
                           linewidth=2)
ax2.add_patch(final_box)
ax2.text(center_x, y_positions[7], '出力: 予測されたPSDデータ\n(3000ポイント)', ha='center', va='center', fontsize=11, fontweight='bold')

ax2.set_title('BERTベースのTransformerモデルの構造', fontsize=18, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/Users/hashimototaiga/Desktop/bert_model_architecture.png', dpi=300, bbox_inches='tight')
print("図2を保存しました: /Users/hashimototaiga/Desktop/bert_model_architecture.png")

# テキストファイルに文章を保存
text_content = """1. BERTの基本概念

BERT（Bidirectional Encoder Representations from Transformers）は、Googleが2018年に発表した自己教師あり学習モデルです。TransformerのEncoder部分のみを使用し、双方向に文脈を読み取ることができるのが特徴です。従来のモデルが左から右（または右から左）にしか読めなかったのに対し、BERTは前後の文脈を同時に見ることができます。これにより、文脈をより深く理解し、マスクされた部分を正確に予測できるようになります。自己教師あり学習により、ラベルなしデータで学習できるため、大量のデータを効率的に活用できます。


3. この研究での使用理由

この研究では、BERTベースのTransformerモデルを使用することで、PSDデータの3000ポイント全体の関係を捉えることができます。双方向性により、マスクされた位置の前後のポイントを同時に見て、より正確にマスク位置を予測できます。長距離依存関係を捉えられるため、離れたポイント間の関係も学習できます。自己教師あり学習により、ラベルなしデータで学習できるため、ノイズ検知のための大量のラベル付きデータを準備する必要がありません。Multi-Head Attentionにより、異なる観点から各ポイントの重要度を学習し、ノイズ区間を効果的に検知できます。
"""

with open('/Users/hashimototaiga/Desktop/bert_explanation_text.txt', 'w', encoding='utf-8') as f:
    f.write(text_content)

print("テキストファイルを保存しました: /Users/hashimototaiga/Desktop/bert_explanation_text.txt")
print("\n=== 完成 ===")
print("1. TransformerとBERTの関係図: transformer_bert_relation.png")
print("2. モデル構造図: bert_model_architecture.png")
print("3. 説明文: bert_explanation_text.txt")

plt.show()

