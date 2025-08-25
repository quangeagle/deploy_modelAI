import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Tạo figure
plt.figure(figsize=(18, 12))
ax = plt.gca()
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis('off')

# Màu sắc
colors = {
    'input': '#90EE90',
    'gru': '#87CEEB',
    'attention': '#FFD700',
    'pooling': '#FFA500',
    'fc': '#DDA0DD',
    'output': '#FF6B6B'
}

# Hàm vẽ mũi tên chuẩn đẹp
def draw_arrow(x1, y1, x2, y2, color='black'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2))

# 1. INPUT SEQUENCE - Căn giữa
input_x, input_y = 2, 6
for i in range(10):
    box = FancyBboxPatch((input_x + i*0.4, input_y-0.3), 0.35, 0.6,
                         boxstyle="round,pad=0.02", facecolor=colors['input'],
                         edgecolor='black')
    ax.add_patch(box)
    plt.text(input_x + i*0.4 + 0.175, input_y, f'X{i+1}',
             ha='center', va='center', fontsize=9, weight='bold')

plt.text(input_x + 2, input_y + 0.8, 'INPUT SEQUENCE\n(10 weeks)',
         ha='center', va='center', fontsize=11, weight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# 2. BIDIRECTIONAL GRU - Căn giữa
gru_x, gru_y = 7, 6
gru_width, gru_height = 3.5, 2.2
gru_box = FancyBboxPatch((gru_x, gru_y), gru_width, gru_height,
                         boxstyle="round,pad=0.1", facecolor=colors['gru'],
                         edgecolor='black', linewidth=2)
ax.add_patch(gru_box)

# Forward và Backward GRU labels
plt.text(gru_x+1, gru_y+1.1, 'FORWARD\nGRU', color='blue', fontsize=9,
         ha='center', va='center', bbox=dict(facecolor='white', edgecolor='blue', linewidth=1))
plt.text(gru_x+2.5, gru_y+1.1, 'BACKWARD\nGRU', color='red', fontsize=9,
         ha='center', va='center', bbox=dict(facecolor='white', edgecolor='red', linewidth=1))

plt.text(gru_x+gru_width/2, gru_y+gru_height+0.3, 'BIDIRECTIONAL GRU\n(128 + 128 = 256 features)',
         ha='center', va='center', fontsize=11, weight='bold')

# 3. LAYER NORM - Căn giữa
ln_x, ln_y = 11.5, 6
ln_box = FancyBboxPatch((ln_x, ln_y), 1.2, 1.2,
                        boxstyle="round,pad=0.05", facecolor='lightgray',
                        edgecolor='black')
ax.add_patch(ln_box)
plt.text(ln_x+0.6, ln_y+0.6, 'LAYER\nNORM', ha='center', va='center', fontsize=9, weight='bold')

# 4. MULTI-HEAD ATTENTION - Căn giữa
attn_x, attn_y = 14, 6
attn_box = FancyBboxPatch((attn_x, attn_y), 3, 1.8,
                          boxstyle="round,pad=0.1", facecolor=colors['attention'],
                          edgecolor='black', linewidth=2)
ax.add_patch(attn_box)

# Attention heads
for i in range(8):
    head_x = attn_x + 0.2 + i * 0.3
    head_y = attn_y + 0.3
    head = plt.Circle((head_x, head_y), 0.08, color='white', ec='black', linewidth=1)
    ax.add_patch(head)
    plt.text(head_x, head_y, f'H{i+1}', ha='center', va='center', fontsize=7, weight='bold')

plt.text(attn_x+1.5, attn_y+2.1, 'MULTI-HEAD ATTENTION (8 heads)',
         ha='center', va='center', fontsize=11, weight='bold')

# 5. DUAL POOLING - Căn giữa
pool_x, pool_y = 18, 6
avg_pool = FancyBboxPatch((pool_x, pool_y), 0.9, 0.7,
                          boxstyle="round,pad=0.05", facecolor=colors['pooling'],
                          edgecolor='black')
ax.add_patch(avg_pool)
plt.text(pool_x+0.45, pool_y+0.35, 'AVG\nPOOL', ha='center', va='center', fontsize=8, weight='bold')

max_pool = FancyBboxPatch((pool_x+1.1, pool_y), 0.9, 0.7,
                          boxstyle="round,pad=0.05", facecolor=colors['pooling'],
                          edgecolor='black')
ax.add_patch(max_pool)
plt.text(pool_x+1.55, pool_y+0.35, 'MAX\nPOOL', ha='center', va='center', fontsize=8, weight='bold')

plt.text(pool_x+1, pool_y-0.5, 'DUAL POOLING (256 + 256 = 512)',
         ha='center', va='center', fontsize=10, weight='bold')

# 6. FULLY CONNECTED LAYERS - Căn giữa
fc_x, fc_y = 18.5, 3
fc1 = FancyBboxPatch((fc_x, fc_y+2), 1.5, 0.8,
                     boxstyle="round,pad=0.05", facecolor=colors['fc'], edgecolor='black')
ax.add_patch(fc1)
plt.text(fc_x+0.75, fc_y+2.4, 'FC1\n256→128', ha='center', va='center', fontsize=8, weight='bold')

fc2 = FancyBboxPatch((fc_x, fc_y+1), 1.5, 0.8,
                     boxstyle="round,pad=0.05", facecolor=colors['fc'], edgecolor='black')
ax.add_patch(fc2)
plt.text(fc_x+0.75, fc_y+1.4, 'FC2\n128→64', ha='center', va='center', fontsize=8, weight='bold')

fc3 = FancyBboxPatch((fc_x, fc_y), 1.5, 0.8,
                     boxstyle="round,pad=0.05", facecolor=colors['fc'], edgecolor='black')
ax.add_patch(fc3)
plt.text(fc_x+0.75, fc_y+0.4, 'FC3\n64→1', ha='center', va='center', fontsize=8, weight='bold')

# Dropout indicators
plt.text(fc_x+1.6, fc_y+2.4, '×', fontsize=14, color='red', weight='bold')
plt.text(fc_x+1.6, fc_y+1.4, '×', fontsize=14, color='red', weight='bold')

plt.text(fc_x+0.75, fc_y+3, 'FULLY CONNECTED LAYERS\n(ReLU + Dropout)',
         ha='center', va='center', fontsize=10, weight='bold')

# 7. OUTPUT - Căn giữa
output_x, output_y = 21, 1.5
output_circle = plt.Circle((output_x, output_y), 0.7, color=colors['output'], ec='black', linewidth=2)
ax.add_patch(output_circle)
plt.text(output_x, output_y, 'SALES\nPREDICTION', ha='center', va='center', fontsize=9, weight='bold', color='white')

# Mũi tên kết nối (được căn chỉnh thẳng và đẹp)
draw_arrow(input_x+4, input_y, gru_x-0.2, gru_y+1.1)
draw_arrow(gru_x+gru_width, gru_y+1.1, ln_x, ln_y+0.6)
draw_arrow(ln_x+1.2, ln_y+0.6, attn_x, attn_y+0.9)
draw_arrow(attn_x+3, attn_y+0.9, pool_x, pool_y+0.35)
draw_arrow(pool_x+1, pool_y, fc_x+0.75, fc_y+2.8)
draw_arrow(fc_x+1.5, fc_y+0.4, output_x-0.7, output_y)

# Tiêu đề - Căn giữa
plt.text(10, 11, 'IMPROVED GRU SALES PREDICTION MODEL',
         ha='center', va='center', fontsize=18, weight='bold',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
plt.text(10, 10.5, 'Features: Bidirectional GRU + Multi-head Attention + Dual Pooling',
         ha='center', va='center', fontsize=13, style='italic')

# Legend - Căn trái dưới
legend_x, legend_y = 0.5, 1.5
legend_items = [
    ('Input Sequence', colors['input']),
    ('Bidirectional GRU', colors['gru']),
    ('Multi-head Attention', colors['attention']),
    ('Dual Pooling', colors['pooling']),
    ('Fully Connected', colors['fc']),
    ('Output Prediction', colors['output'])
]

for i, (label, color) in enumerate(legend_items):
    legend_box = FancyBboxPatch((legend_x, legend_y - i*0.4), 0.4, 0.3,
                               boxstyle="round,pad=0.02", 
                               facecolor=color, 
                               edgecolor='black', linewidth=1)
    ax.add_patch(legend_box)
    plt.text(legend_x + 0.5, legend_y - i*0.4 + 0.15, label, 
             ha='left', va='center', fontsize=10)

plt.text(legend_x, legend_y + 0.4, 'LEGEND', 
         ha='left', va='center', fontsize=12, weight='bold')

# Thông tin kỹ thuật - Căn phải dưới
tech_x, tech_y = 14, 1.5
tech_info = [
    'Model Architecture:',
    '• Hidden Size: 128',
    '• Num Layers: 2', 
    '• Dropout: 0.2',
    '• Attention Heads: 8',
    '• Bidirectional: True'
]

for i, info in enumerate(tech_info):
    if i == 0:
        plt.text(tech_x, tech_y - i*0.3, info, 
                ha='left', va='center', fontsize=11, weight='bold')
    else:
        plt.text(tech_x, tech_y - i*0.3, info, 
                ha='left', va='center', fontsize=9)

# Lưu hình ảnh
plt.tight_layout()
plt.savefig('gru_architecture_diagram.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('gru_architecture_diagram.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("✅ Đã tạo thành công hình ảnh kiến trúc GRU với căn chỉnh đẹp!")
print("📁 Files đã lưu:")
print("   • gru_architecture_diagram.png (PNG format)")
print("   • gru_architecture_diagram.pdf (PDF format)")

plt.show()
