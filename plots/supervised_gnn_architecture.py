"""Horizontal architecture diagram for the Supervised GNN Classifier.

Style: colored background regions, grid/matrix visualizations for tensors,
compact component boxes, horizontal data flow (left -> right).

Key architectural detail: edge features [E, 5] bypass the input projection
and enter the GAT blocks at the attention score computation step via
edge_proj(edge_attr). Only node features [N, 40] go through input_proj.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

fig, ax = plt.subplots(figsize=(24, 8))
ax.set_xlim(0, 24)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')

# ── Colour palette ──────────────────────────────────────────────────
# Background regions
BG_GREEN  = '#E8F5E9'
BG_ORANGE = '#FFF8E1'
BG_PURPLE = '#EDE7F6'

# Component fills
C_INPUT  = '#C8E6C9'
C_PROJ   = '#BBDEFB'
C_GAT    = '#FFE0B2'
C_POOL   = '#E1BEE7'
C_CLS    = '#FFCDD2'
C_OUT    = '#FFF9C4'

# Grid colours
G_NODE   = '#FF9800'
G_EDGE   = '#42A5F5'
G_EMBED  = '#FF9800'
G_GRAPH  = '#AB47BC'

# Borders / arrows
C_ARROW  = '#546E7A'
C_BORDER = '#37474F'
C_GAT_BORDER = '#EF6C00'
C_RESIDUAL = '#E65100'

CY = 4.0  # centre y for main flow


# ── Helper functions ────────────────────────────────────────────────

def draw_box(x, y, w, h, text, color, fontsize=10, bold=False,
             border=C_BORDER, lw=1.5, ls='-'):
    """Rounded box with centred text."""
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.1',
        facecolor=color, edgecolor=border, linewidth=lw,
        linestyle=ls, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, zorder=3)


def draw_grid(x, y, w, h, nr, nc, color, seed=42):
    """Stylised matrix grid with per-cell colour variation."""
    rng = np.random.RandomState(seed)
    base = np.array(plt.matplotlib.colors.to_rgb(color))
    cw, ch = w / nc, h / nr
    for r in range(nr):
        for c in range(nc):
            b = 0.50 + 0.50 * rng.random()
            cc = np.clip(base * b + (1 - b) * 0.15, 0, 1)
            cell = Rectangle((x + c * cw, y + r * ch), cw, ch,
                              facecolor=cc, edgecolor='white',
                              linewidth=0.4, zorder=2)
            ax.add_patch(cell)
    border = Rectangle((x, y), w, h, facecolor='none',
                        edgecolor=C_BORDER, linewidth=1.0, zorder=3)
    ax.add_patch(border)


def harrow(x1, x2, y, color=C_ARROW, lw=2.0):
    """Horizontal arrow."""
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=4)


def carrow(x1, y1, x2, y2, rad=0.2, color=C_ARROW, lw=1.5):
    """Curved arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle=f'arc3,rad={rad}'),
                zorder=4)


def txt(x, y, s, fontsize=9, bold=False, color='black',
        ha='center', va='center'):
    """Text label."""
    ax.text(x, y, s, ha=ha, va=va, fontsize=fontsize,
            fontweight='bold' if bold else 'normal', color=color, zorder=5)


def dim(x, y, s):
    """Dimension annotation (italic, grey)."""
    ax.text(x, y, s, ha='center', va='center', fontsize=8,
            color='#757575', fontstyle='italic', zorder=5)


# ═══════════════════════════════════════════════════════════════════
# Background regions
# ═══════════════════════════════════════════════════════════════════
regions = [
    (0.15, 6.05, BG_GREEN,  'Graph Construction'),
    (6.35, 9.15, BG_ORANGE, 'GAT Backbone'),
    (15.65, 8.15, BG_PURPLE, 'Readout & Prediction'),
]
for rx, rw, rc, rl in regions:
    bg = FancyBboxPatch(
        (rx, 0.35), rw, 7.0, boxstyle='round,pad=0.15',
        facecolor=rc, edgecolor='none', alpha=0.45, zorder=0)
    ax.add_patch(bg)
    txt(rx + rw / 2, 7.55, rl, fontsize=12, bold=True, color='#555')


# ═══════════════════════════════════════════════════════════════════
# 1. SMILES Input
# ═══════════════════════════════════════════════════════════════════
draw_box(0.6, CY - 0.55, 2.0, 1.1, 'SMILES\nInput', C_INPUT,
         fontsize=11, bold=True)
harrow(2.6, 3.15, CY)
txt(2.88, CY + 0.25, 'RDKit', fontsize=7.5, color='#666')


# ═══════════════════════════════════════════════════════════════════
# 2. Node & Edge Feature Grids
# ═══════════════════════════════════════════════════════════════════
# Node features (upper) — these go to Input Projection
nf_x, nf_w, nf_h = 3.4, 1.6, 1.5
nf_y = CY + 0.25
draw_grid(nf_x, nf_y, nf_w, nf_h, 8, 5, G_NODE, seed=42)
txt(nf_x + nf_w / 2, nf_y + nf_h + 0.22, 'Node Features',
    fontsize=8.5, bold=True)
dim(nf_x + nf_w / 2, nf_y - 0.22, '[N, 40]')

# Edge features (lower) — these bypass Input Projection, go to GAT
ef_x, ef_w, ef_h = 3.6, 0.9, 1.1
ef_y = CY - 1.6
draw_grid(ef_x, ef_y, ef_w, ef_h, 6, 3, G_EDGE, seed=17)
txt(ef_x + ef_w / 2, ef_y - 0.25, 'Edge Features',
    fontsize=8.5, bold=True)
dim(ef_x + ef_w / 2, ef_y - 0.48, '[E, 5]')

# Node features → Input Projection (curved arrow, upper path)
carrow(nf_x + nf_w + 0.08, nf_y + nf_h / 2,
       6.5, CY + 0.1, rad=-0.15)


# ═══════════════════════════════════════════════════════════════════
# 3. Input Projection  (node features only)
# ═══════════════════════════════════════════════════════════════════
ip_x, ip_w, ip_h = 6.6, 2.0, 1.2
draw_box(ip_x, CY - ip_h / 2, ip_w, ip_h,
         'Input Projection\nLinear(40 → 256)\n(node features only)', C_PROJ,
         fontsize=8.5, bold=True)
dim(ip_x + ip_w / 2, CY - ip_h / 2 - 0.22, '[N, 256]')
harrow(ip_x + ip_w, ip_x + ip_w + 0.45, CY)


# ═══════════════════════════════════════════════════════════════════
# 4. ×3 GAT Blocks  (dashed box with expanded internal sub-boxes)
# ═══════════════════════════════════════════════════════════════════
gat_x = 9.15
gat_w = 4.6
inner_w = 3.8
inner_x = gat_x + (gat_w - inner_w) / 2

# Sub-box definitions: (label, height)
sub_boxes = [
    ('Multi-Head Projection\nLinear(256 → 1024) → reshape [N, 4, 256]', 0.7),
    ('Attention Scores\n'
     'att_src[src] + att_dst[dst] + edge_proj(edge_attr)\n'
     '→ LeakyReLU(0.2) → edge softmax', 0.95),
    ('Message Aggregation\n'
     'weighted sum over source nodes\n'
     '→ mean over 4 heads + bias → [N, 256]', 0.8),
    ('LayerNorm → ELU → Dropout(0.1)', 0.5),
]
inner_gap = 0.12
n_steps = len(sub_boxes)
total_inner = sum(h for _, h in sub_boxes) + (n_steps - 1) * inner_gap
top_pad = 0.75
bot_pad = 0.65
gat_h = top_pad + total_inner + bot_pad
gat_y = CY - gat_h / 2

# Outer dashed box
gat_box = FancyBboxPatch(
    (gat_x, gat_y), gat_w, gat_h, boxstyle='round,pad=0.12',
    facecolor='#FFF8E1', edgecolor=C_GAT_BORDER,
    linewidth=2, linestyle='--', zorder=2)
ax.add_patch(gat_box)
txt(gat_x + gat_w / 2, gat_y + gat_h - 0.3,
    '×3 GAT Blocks', fontsize=10.5, bold=True, color=C_RESIDUAL)

# Draw inner sub-boxes (stacked top → bottom)
iy = gat_y + gat_h - top_pad
res_first_mid = None
res_last_mid = None
attn_box_left = None
attn_box_mid_y = None

for i, (step_text, step_h) in enumerate(sub_boxes):
    draw_box(inner_x, iy - step_h, inner_w, step_h, step_text,
             C_GAT, fontsize=7.5, border='#BF360C', lw=1)
    mid_y = iy - step_h / 2
    if i == 0:
        res_first_mid = mid_y
    if i == n_steps - 1:
        res_last_mid = mid_y
    if i == 1:  # attention scores box — edge features enter here
        attn_box_left = inner_x
        attn_box_mid_y = mid_y
    if i < n_steps - 1:
        arr_top = iy - step_h - 0.02
        arr_bot = arr_top - inner_gap + 0.04
        ax.annotate('', xy=(gat_x + gat_w / 2, arr_bot),
                    xytext=(gat_x + gat_w / 2, arr_top),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1),
                    zorder=4)
    iy -= step_h + inner_gap

# Residual arc (right side)
res_x = inner_x + inner_w + 0.05
ax.annotate(
    '', xy=(res_x, res_last_mid),
    xytext=(res_x, res_first_mid),
    arrowprops=dict(arrowstyle='->', color=C_RESIDUAL, lw=1.3,
                    connectionstyle='arc3,rad=-0.35'),
    zorder=4)
txt(res_x + 0.08, (res_first_mid + res_last_mid) / 2 + 0.25,
    '+ res.', fontsize=7, bold=True, color=C_RESIDUAL)

dim(gat_x + gat_w / 2, gat_y - 0.22, '[N, 256]')


# ── Edge features bypass arrow ────────────────────────────────────
# Edge features [E, 5] skip the Input Projection and enter the GAT
# block at the Attention Scores step via edge_proj(edge_attr)
carrow(ef_x + ef_w + 0.08, ef_y + ef_h / 2,
       gat_x + 0.05, attn_box_mid_y,
       rad=0.35, color=G_EDGE, lw=2.0)
# Label on the bypass arrow
txt(6.8, ef_y + ef_h / 2 - 0.4,
    'edge_attr [E, 5]', fontsize=7.5, bold=True, color='#1565C0')

# Arrow from GAT block to next component
harrow(gat_x + gat_w, gat_x + gat_w + 0.45, CY)


# ═══════════════════════════════════════════════════════════════════
# 5. Node Embedding Grid
# ═══════════════════════════════════════════════════════════════════
ne_x, ne_w, ne_h = 14.35, 1.1, 1.6
ne_y = CY - ne_h / 2
draw_grid(ne_x, ne_y, ne_w, ne_h, 8, 5, G_EMBED, seed=55)
txt(ne_x + ne_w / 2, ne_y + ne_h + 0.22, 'Node\nEmbedding',
    fontsize=8.5, bold=True)
dim(ne_x + ne_w / 2, ne_y - 0.22, '[N, 256]')

harrow(ne_x + ne_w + 0.08, ne_x + ne_w + 0.55, CY)


# ═══════════════════════════════════════════════════════════════════
# 6. Mean Pooling
# ═══════════════════════════════════════════════════════════════════
mp_x, mp_w, mp_h = 16.1, 1.9, 1.1
draw_box(mp_x, CY - mp_h / 2, mp_w, mp_h,
         'Mean\nPooling', C_POOL, fontsize=10, bold=True)
harrow(mp_x + mp_w, mp_x + mp_w + 0.35, CY)


# ═══════════════════════════════════════════════════════════════════
# 7. Graph Embedding Bar
# ═══════════════════════════════════════════════════════════════════
ge_x, ge_w, ge_h = 18.45, 1.5, 0.45
ge_y = CY - ge_h / 2
draw_grid(ge_x, ge_y, ge_w, ge_h, 1, 8, G_GRAPH, seed=77)
txt(ge_x + ge_w / 2, ge_y + ge_h + 0.3, 'Graph\nEmbedding',
    fontsize=8.5, bold=True)
dim(ge_x + ge_w / 2, ge_y - 0.25, '[B, 256]')

harrow(ge_x + ge_w + 0.08, ge_x + ge_w + 0.45, CY)


# ═══════════════════════════════════════════════════════════════════
# 8. Classification Head
# ═══════════════════════════════════════════════════════════════════
ch_x, ch_w, ch_h = 20.5, 1.8, 1.5
draw_box(ch_x, CY - ch_h / 2, ch_w, ch_h,
         'Classification\nHead\n\nLinear(256→128)\nReLU\nLinear(128→1)',
         C_CLS, fontsize=8, bold=True)
dim(ch_x + ch_w / 2, CY - ch_h / 2 - 0.22, '[B, 1]')

harrow(ch_x + ch_w, ch_x + ch_w + 0.3, CY)


# ═══════════════════════════════════════════════════════════════════
# 9. Output
# ═══════════════════════════════════════════════════════════════════
out_x, out_w, out_h = 22.7, 1.1, 0.9
draw_box(out_x, CY - out_h / 2, out_w, out_h,
         'σ →\nP(PKS)', C_OUT, fontsize=9.5, bold=True)
dim(out_x + out_w / 2, CY - out_h / 2 - 0.22, 'scalar ∈ [0, 1]')


# ═══════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════
plt.tight_layout()
plt.savefig(
    '/Users/yashchainani/Desktop/PythonProjects/ContrastiveGNNs/'
    'plots/supervised_gnn_architecture.png',
    dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(
    '/Users/yashchainani/Desktop/PythonProjects/ContrastiveGNNs/'
    'plots/supervised_gnn_architecture.pdf',
    bbox_inches='tight', facecolor='white')
plt.close()
print("Saved to plots/supervised_gnn_architecture.png and .pdf")
