"""Bar chart of SupCon GNN AUPRC on validation and test sets."""

import matplotlib.pyplot as plt
import numpy as np

splits = ['Validation', 'Test']
auprc = [0.96, 0.88]
colors = ['#1565C0', '#E65100']

fig, ax = plt.subplots(figsize=(6, 5))

bars = ax.bar(splits, auprc, width=0.5, color=colors, edgecolor='black', linewidth=0.8)

# Value labels on bars
for bar, val in zip(bars, auprc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('AUPRC', fontsize=13)
ax.set_title('SupCon GNN — AUPRC by Split', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.08)
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('plots/gnn_auprc_val_test.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('plots/gnn_auprc_val_test.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved to plots/gnn_auprc_val_test.png and .pdf")
