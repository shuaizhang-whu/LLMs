"""FR-SVD weight size and speedup, sized for an ACL single column.

All values are the corrected LLaMA-7B measurements already used by the
10.7 manuscript. Percentages refer to the compression setting; model weight
size is measured in GiB, and speedup is relative to the uncompressed model.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COMPRESSION_RATIO = [0, 20, 40, 60, 80]
WEIGHT_GIB = [12.55, 10.48, 8.15, 5.73, 3.63]
SPEEDUP = [1.00, 1.17, 1.48, 2.10, 3.28]
ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9,
    'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5, 'axes.linewidth': .7,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
fig, ax = plt.subplots(figsize=(3.35, 2.40))
fig.subplots_adjust(left=.15, right=.85, bottom=.20, top=.85)
ax2 = ax.twinx()
x = list(range(5))
bar_colors = ['#D9EAF7', '#BBD7EE', '#8FBAD9', '#5E95BE', '#2F6F9F']
line_color = '#BA352C'
bars = ax.bar(x, WEIGHT_GIB, width=.59, color=bar_colors,
              edgecolor='#345266', linewidth=.55, zorder=3)
ax2.plot(x, SPEEDUP, color=line_color, marker='o', markersize=4.2,
         markerfacecolor='white', markeredgewidth=1.15, linewidth=1.4, zorder=4)
for i, (bar, value) in enumerate(zip(bars, WEIGHT_GIB)):
    ax.annotate(f'{value:.2f}', (i, value), xytext=(0, 3),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=8.2, color='#203A4C')
for i, value in enumerate(SPEEDUP):
    ax2.annotate(f'{value:.2f}', (i, value), xytext=(0, 5),
                 textcoords='offset points', ha='center', va='bottom',
                 fontsize=8.3, color=line_color, fontweight='bold')
ax.set_xticks(x, [str(v) for v in COMPRESSION_RATIO])
ax.set_xlim(-.6, 4.6)
ax.set_ylim(0, 14.8)
ax.set_yticks([0, 4, 8, 12])
ax2.set_ylim(0, 3.85)
ax2.set_yticks([0, 1, 2, 3])
ax.set_xlabel('Compression ratio (%)', labelpad=3)
ax.set_ylabel('Weight size (GiB)', labelpad=3)
ax2.set_ylabel('Speedup ($\\times$)', labelpad=3, color=line_color)
ax2.tick_params(axis='y', colors=line_color)
ax.grid(axis='y', color='#DADADA', linewidth=.5, linestyle='--', zorder=0)
for a in (ax, ax2):
    a.tick_params(direction='in', length=3, width=.65)
    a.spines['top'].set_visible(False)
ax.legend(handles=[Patch(facecolor=bar_colors[2], edgecolor='#345266', label='Weight size'),
    Line2D([], [], color=line_color, marker='o', markerfacecolor='white', markersize=4,
           linewidth=1.2, label='Speedup')],
    loc='lower center', bbox_to_anchor=(.5, 1.04), ncol=2,
    frameon=False, columnspacing=1.2, handlelength=1.5, handletextpad=.5, borderaxespad=0)
fig.savefig(ROOT/'figs/frsvd_weight_speedup.pdf', metadata={'Title':'FR-SVD model weight size and speedup','Author':'Anonymous'})
fig.savefig('/private/tmp/frsvd_weight_speedup_coling.png', dpi=300)
plt.close(fig)
