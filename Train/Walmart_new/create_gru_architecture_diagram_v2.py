import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, ConnectionPatch
import numpy as np

# Tạo figure với kích thước phù hợp
fig, ax = plt.subplots(1, 1, figsize=(20, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis('off')

# Màu sắc đẹp và nhất quán
colors = {
    'input': '#90EE90',      # Xanh lá nhạt
    'gru': '#87CEEB',        # Xanh dương nhạt
    'attention': '#FFD700',   # Vàng
    'pooling': '#FFA500',     # Cam
    'fc': '#DDA0DD',         # Tím nhạt
    'output': '#FF6B6B',     # Đỏ
    'layer_norm': '#D3D3D3'  # Xám nhạt
}

# Hàm vẽ mũi tên đẹp
def draw_connection_arrow(start_x, start_y, end_x, end_y, color='black', width=2):
    arrow = patches.FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        arrowstyle='->', mutation_scale=20, 
        color=color, linewidth=width
    )
    ax.add_patch(arrow)

# 1. INPUT SEQUENCE (10 tuần)
input_x, input_y = 1, 6
input_width, input_height = 0.4, 0.8

plt.text(input_x + 2, input_y + 1.5, 'INPUT SEQUENCE\n(10 weeks)', 
         ha='center', va='center', fontsize=12, weight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))

for i in range(10):
    box = FancyBboxPatch(
        (input_x + i*0.4, input_y - 0.4), input_width, input_height,
        boxstyle="round,pad=0.02", 
        facecolor=colors['input'], 
        edgecolor='black', linewidth=1
    )
    ax.add_patch(box)
    plt.text(input_x + i*0.4 + 0.2, input_y, f'X{i+1}', 
             ha='center', va='center', fontsize=9, weight='bold')

# 2. BIDIRECTIONAL GRU
gru_x, gru_y = 6, 6
gru_width, gru_height = 4, 2.5

# Container chính của GRU
gru_container = FancyBboxPatch(
    (gru_x, gru_y - 1.25), gru_width, gru_height,
    boxstyle="round,pad=0.1", 
    facecolor=colors['gru'], 
    edgecolor='black', linewidth=2
)
ax.add_patch(gru_container)

# Forward GRU
forward_box = FancyBboxPatch(
    (gru_x + 0.3, gru_y - 0.8), 1.5, 1,
    boxstyle="round,pad=0.05", 
    facecolor='white', 
    edgecolor='blue', linewidth=2
)
ax.add_patch(forward_box)
plt.text(gru_x + 1.05, gru_y - 0.3, 'FORWARD\nGRU', 
         ha='center', va='center', fontsize=10, weight='bold', color='blue')

# Backward GRU
backward_box = FancyBboxPatch(
    (gru_x + 2.2, gru_y - 0.8), 1.5, 1,
    boxstyle="round,pad=0.05", 
    facecolor='white', 
    edgecolor='red', linewidth=2
)
ax.add_patch(backward_box)
plt.text(gru_x + 2.95, gru_y - 0.3, 'BACKWARD\nGRU', 
         ha='center', va='center', fontsize=10, weight='bold', color='red')

# Mũi tên bidirectional
for i in range(3):
    # Forward arrows (blue)
    arrow_y = gru_y - 0.6 + i*0.3
    draw_connection_arrow(gru_x + 1.05, arrow_y, gru_x + 1.8, arrow_y, 'blue', 2)
    # Backward arrows (red)
    draw_connection_arrow(gru_x + 2.95, arrow_y, gru_x + 2.2, arrow_y, 'red', 2)

plt.text(gru_x + 2, gru_y + 1.5, 'BIDIRECTIONAL GRU\n(128 + 128 = 256 features)', 
         ha='center', va='center', fontsize=12, weight='bold')

# 3. LAYER NORMALIZATION
ln_x, ln_y = 11, 6
ln_box = FancyBboxPatch(
    (ln_x, ln_y - 0.6), 1.2, 1.2,
    boxstyle="round,pad=0.05", 
    facecolor=colors['layer_norm'], 
    edgecolor='black', linewidth=1
)
ax.add_patch(ln_box)
plt.text(ln_x + 0.6, ln_y, 'LAYER\nNORM', 
         ha='center', va='center', fontsize=10, weight='bold')

# 4. MULTI-HEAD ATTENTION
attn_x, attn_y = 13.5, 6
attn_width, attn_height = 3.5, 2

attn_container = FancyBboxPatch(
    (attn_x, attn_y - 1), attn_width, attn_height,
    boxstyle="round,pad=0.1", 
    facecolor=colors['attention'], 
    edgecolor='black', linewidth=2
)
ax.add_patch(attn_container)

# 8 Attention heads
for i in range(8):
    head_x = attn_x + 0.3 + i * 0.35
    head_y = attn_y - 0.3
    head_circle = Circle((head_x, head_y), 0.12, 
                        facecolor='white', 
                        edgecolor='black', linewidth=1)
    ax.add_patch(head_circle)
    plt.text(head_x, head_y, f'H{i+1}', 
             ha='center', va='center', fontsize=8, weight='bold')
    
    # Attention arrows
    draw_connection_arrow(head_x, head_y - 0.15, head_x, head_y - 0.4, 'black', 1)

plt.text(attn_x + attn_width/2, attn_y + 1.3, 'MULTI-HEAD ATTENTION\n(8 heads)', 
         ha='center', va='center', fontsize=12, weight='bold')

# 5. DUAL POOLING
pool_x, pool_y = 18, 6.5

# Average Pooling
avg_pool = FancyBboxPatch(
    (pool_x, pool_y + 0.5), 1, 0.8,
    boxstyle="round,pad=0.05", 
    facecolor=colors['pooling'], 
    edgecolor='black', linewidth=1
)
ax.add_patch(avg_pool)
plt.text(pool_x + 0.5, pool_y + 0.9, 'AVG\nPOOL', 
         ha='center', va='center', fontsize=9, weight='bold')

# Max Pooling
max_pool = FancyBboxPatch(
    (pool_x, pool_y - 0.5), 1, 0.8,
    boxstyle="round,pad=0.05", 
    facecolor=colors['pooling'], 
    edgecolor='black', linewidth=1
)
ax.add_patch(max_pool)
plt.text(pool_x + 0.5, pool_y - 0.1, 'MAX\nPOOL', 
         ha='center', va='center', fontsize=9, weight='bold')

# Combine arrow pointing down
draw_connection_arrow(pool_x + 0.5, pool_y + 0.4, pool_x + 0.5, pool_y - 0.4, 'black', 2)

plt.text(pool_x + 0.5, pool_y - 1.5, 'DUAL POOLING\n(256 + 256 = 512)', 
         ha='center', va='center', fontsize=11, weight='bold')

# 6. FULLY CONNECTED LAYERS
fc_x, fc_y = 18.2, 3.5
fc_width, fc_height = 1.6, 0.8
fc_gap = 0.3

# FC1
fc1 = FancyBboxPatch(
    (fc_x, fc_y + 2*(fc_height + fc_gap)), fc_width, fc_height,
    boxstyle="round,pad=0.05", 
    facecolor=colors['fc'], 
    edgecolor='black', linewidth=1
)
ax.add_patch(fc1)
plt.text(fc_x + fc_width/2, fc_y + 2*(fc_height + fc_gap) + fc_height/2, 'FC1\n512→128', 
         ha='center', va='center', fontsize=9, weight='bold')

# FC2
fc2 = FancyBboxPatch(
    (fc_x, fc_y + (fc_height + fc_gap)), fc_width, fc_height,
    boxstyle="round,pad=0.05", 
    facecolor=colors['fc'], 
    edgecolor='black', linewidth=1
)
ax.add_patch(fc2)
plt.text(fc_x + fc_width/2, fc_y + (fc_height + fc_gap) + fc_height/2, 'FC2\n128→64', 
         ha='center', va='center', fontsize=9, weight='bold')

# FC3
fc3 = FancyBboxPatch(
    (fc_x, fc_y), fc_width, fc_height,
    boxstyle="round,pad=0.05", 
    facecolor=colors['fc'], 
    edgecolor='black', linewidth=1
)
ax.add_patch(fc3)
plt.text(fc_x + fc_width/2, fc_y + fc_height/2, 'FC3\n64→1', 
         ha='center', va='center', fontsize=9, weight='bold')

# Dropout indicators
plt.text(fc_x + fc_width + 0.1, fc_y + 2*(fc_height + fc_gap) + fc_height/2, '×', 
         fontsize=16, color='red', weight='bold')
plt.text(fc_x + fc_width + 0.1, fc_y + (fc_height + fc_gap) + fc_height/2, '×', 
         fontsize=16, color='red', weight='bold')

plt.text(fc_x + fc_width/2, fc_y + 3*(fc_height + fc_gap) + 0.3, 
         'FULLY CONNECTED LAYERS\n(ReLU + Dropout)', 
         ha='center', va='center', fontsize=11, weight='bold')

# 7. OUTPUT PREDICTION
output_x, output_y = 22, 3.9
output_radius = 0.9

output_circle = Circle((output_x, output_y), output_radius, 
                      facecolor=colors['output'], 
                      edgecolor='black', linewidth=3)
ax.add_patch(output_circle)
plt.text(output_x, output_y, 'SALES\nPREDICTION', 
         ha='center', va='center', fontsize=11, weight='bold', color='white')

# 8. MŨI TÊN KẾT NỐI CHÍNH
# Input → GRU
draw_connection_arrow(input_x + 4, input_y, gru_x - 0.2, gru_y, 'black', 3)

# GRU → Layer Norm
draw_connection_arrow(gru_x + gru_width, gru_y, ln_x, ln_y, 'black', 3)

# Layer Norm → Attention
draw_connection_arrow(ln_x + 1.2, ln_y, attn_x, attn_y, 'black', 3)

# Attention → Pooling
draw_connection_arrow(attn_x + attn_width, attn_y, pool_x, pool_y, 'black', 3)

# Pooling → FC
draw_connection_arrow(pool_x + 0.5, pool_y - 1.5, fc_x + fc_width/2, fc_y + 3*(fc_height + fc_gap), 'black', 3)

# FC → Output
draw_connection_arrow(fc_x + fc_width, fc_y + fc_height/2, output_x - output_radius, output_y, 'black', 3)

# 9. TIÊU ĐỀ VÀ THÔNG TIN
plt.text(10, 11, 'IMPROVED GRU SALES PREDICTION MODEL', 
         ha='center', va='center', fontsize=20, weight='bold',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))

plt.text(10, 10.3, 'Features: Bidirectional GRU + Multi-head Attention + Dual Pooling', 
         ha='center', va='center', fontsize=14, style='italic')

# 10. LEGEND
legend_x, legend_y = 0.5, 2.5
legend_items = [
    ('Input Sequence', colors['input']),
    ('Bidirectional GRU', colors['gru']),
    ('Layer Normalization', colors['layer_norm']),
    ('Multi-head Attention', colors['attention']),
    ('Dual Pooling', colors['pooling']),
    ('Fully Connected', colors['fc']),
    ('Output Prediction', colors['output'])
]

plt.text(legend_x, legend_y + 0.5, 'LEGEND', 
         ha='left', va='center', fontsize=14, weight='bold')

for i, (label, color) in enumerate(legend_items):
    legend_box = FancyBboxPatch(
        (legend_x, legend_y - i*0.4), 0.4, 0.3,
        boxstyle="round,pad=0.02", 
        facecolor=color, 
        edgecolor='black', linewidth=1
    )
    ax.add_patch(legend_box)
    plt.text(legend_x + 0.5, legend_y - i*0.4 + 0.15, label, 
             ha='left', va='center', fontsize=11)

# 11. THÔNG TIN KỸ THUẬT
tech_x, tech_y = 13, 2.5
tech_info = [
    'Model Architecture:',
    '• Hidden Size: 128',
    '• Num Layers: 2', 
    '• Dropout: 0.2',
    '• Attention Heads: 8',
    '• Bidirectional: True',
    '• Lookback: 10 weeks'
]

for i, info in enumerate(tech_info):
    if i == 0:
        plt.text(tech_x, tech_y - i*0.3, info, 
                ha='left', va='center', fontsize=12, weight='bold')
    else:
        plt.text(tech_x, tech_y - i*0.3, info, 
                ha='left', va='center', fontsize=10)

# 12. LƯU FILE
plt.tight_layout()
plt.savefig('gru_architecture_diagram_v2.png', 
            dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('gru_architecture_diagram_v2.pdf', 
            bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("✅ Đã tạo thành công hình ảnh kiến trúc GRU hoàn chỉnh!")
print("📁 Files đã lưu:")
print("   • gru_architecture_diagram_v2.png (PNG format)")
print("   • gru_architecture_diagram_v2.pdf (PDF format)")
print("\n🎯 Đặc điểm của hình ảnh mới:")
print("   • Layout cân đối và đẹp mắt")
print("   • Tất cả thành phần đầy đủ")
print("   • Mũi tên kết nối rõ ràng")
print("   • Màu sắc nhất quán")
print("   • Legend và thông tin kỹ thuật đầy đủ")

plt.show()
