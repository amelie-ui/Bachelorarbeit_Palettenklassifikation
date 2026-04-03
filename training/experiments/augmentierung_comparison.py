"""
Visualisierungen für den Augmentierungsvergleich.
Liest Ergebnisse direkt aus augmentation_comparison.json.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PATHS

# ── Daten aus JSON laden ───────────────────────────────────────────────────────
json_path = PATHS['metrics'] / 'augmentation_comparison.json'
with open(json_path, 'r') as f:
    results = json.load(f)

ref = next(r for r in results if r['experiment'] == 'keine_augmentation')
REF_ACC  = ref['best_val_acc']
REF_LOSS = ref['best_val_loss']

print(f'Geladen: {json_path} ({len(results)} Experimente)')
print(f'Referenz: Val Accuracy = {REF_ACC:.4f}, Val Loss = {REF_LOSS:.4f}')

LABELS = {
    'keine_augmentation': 'Keine Aug. (Referenz)',
    'nur_flip':           'Flip',
    'nur_rotation':       'Rotation',
    'nur_brightness':     'Brightness',
    'nur_contrast':       'Contrast',
    'nur_zoom':           'Zoom',
    'alle_kombiniert':    'Alle kombiniert',
}

# Farben: räumlich = rot-orange, pixelweise = blau, referenz = grau, kombiniert = schwarz
COLORS = {
    'keine_augmentation': '#888888',
    'nur_flip':           '#E07B39',
    'nur_rotation':       '#C0392B',
    'nur_brightness':     '#2980B9',
    'nur_contrast':       '#1A5276',
    'nur_zoom':           '#E8A838',
    'alle_kombiniert':    '#1C1C1C',
}

# sortiert nach val_acc absteigend
sorted_results = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)

# ══════════════════════════════════════════════════════════════════════
# Abbildung 1: Horizontales Balkendiagramm — Val Accuracy
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

names  = [LABELS[r['experiment']] for r in sorted_results]
accs   = [r['best_val_acc'] for r in sorted_results]
colors = [COLORS[r['experiment']] for r in sorted_results]

bars = ax.barh(names, accs, color=colors, height=0.55, edgecolor='white', linewidth=0.5)

# Referenzlinie
ax.axvline(REF_ACC, color='#888888', linestyle='--', linewidth=1.2, label=f'Referenz (keine Aug.) = {REF_ACC:.3f}')

# Werte an Balken
for bar, val in zip(bars, accs):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', ha='left', fontsize=9)

ax.set_xlabel('Val Accuracy', fontsize=11)
ax.set_title('Vergleich: Val Accuracy je Augmentierungskonfiguration', fontsize=11, pad=12)
ax.set_xlim(0.35, 1.05)
ax.legend(fontsize=9)
ax.invert_yaxis()

# Legende räumlich/pixelweise
#patch_spatial = mpatches.Patch(color='#C0392B', label='Räumliche Transformation')
#patch_pixel   = mpatches.Patch(color='#2980B9', label='Pixelweise Transformation')
#patch_ref     = mpatches.Patch(color='#888888', label='Referenz / Kombiniert')
#ax.legend(handles=[patch_spatial, patch_pixel, patch_ref], fontsize=8, loc='lower right')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(str(PATHS['plots'] / 'aug_fig1_bar_val_acc.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Gespeichert: aug_fig1_bar_val_acc.png')


# ══════════════════════════════════════════════════════════════════════
# Abbildung 2: Delta zur Referenz — Abfall in Prozentpunkten
# ══════════════════════════════════════════════════════════════════════
# Nur Transformationen, nicht die Referenz selbst
trans_results = [r for r in results if r['experiment'] != 'keine_augmentation']
trans_sorted  = sorted(trans_results, key=lambda x: x['best_val_acc'] - REF_ACC)

fig, ax = plt.subplots(figsize=(9, 5))

names  = [LABELS[r['experiment']] for r in trans_sorted]
deltas = [r['best_val_acc'] - REF_ACC for r in trans_sorted]
colors = [COLORS[r['experiment']] for r in trans_sorted]

bars = ax.barh(names, deltas, color=colors, height=0.55, edgecolor='white', linewidth=0.5)
ax.axvline(0, color='#888888', linestyle='--', linewidth=1.2)

for bar, val in zip(bars, deltas):
    offset = -0.01 if val < 0 else 0.005
    ha = 'right' if val < 0 else 'left'
    ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f}', va='center', ha=ha, fontsize=9)

ax.set_xlabel('Delta Val Accuracy gegenüber Referenz (keine Augmentierung)', fontsize=10)
ax.set_title('Abbildung 2 — Abfall der Val Accuracy gegenüber Referenz', fontsize=11, pad=12)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

patch_spatial = mpatches.Patch(color='#C0392B', label='Räumliche Transformation')
patch_pixel   = mpatches.Patch(color='#2980B9', label='Pixelweise Transformation')
patch_comb    = mpatches.Patch(color='#1C1C1C', label='Alle kombiniert')
ax.legend(handles=[patch_spatial, patch_pixel, patch_comb], fontsize=8)

plt.tight_layout()
plt.savefig(str(PATHS['plots'] / 'aug_fig2_delta_acc.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Gespeichert: aug_fig2_delta_acc.png')


# ══════════════════════════════════════════════════════════════════════
# Abbildung 3: Doppelplot — Val Accuracy + Val Loss nebeneinander
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

sorted_by_acc  = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)
sorted_by_loss = sorted(results, key=lambda x: x['best_val_loss'])

# Linker Plot: Val Accuracy
names_acc  = [LABELS[r['experiment']] for r in sorted_by_acc]
accs       = [r['best_val_acc'] for r in sorted_by_acc]
cols_acc   = [COLORS[r['experiment']] for r in sorted_by_acc]
ax1.barh(names_acc, accs, color=cols_acc, height=0.55, edgecolor='white')
ax1.axvline(REF_ACC, color='#888888', linestyle='--', linewidth=1.2)
for i, (val, name) in enumerate(zip(accs, names_acc)):
    ax1.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=8.5)
ax1.set_xlabel('Beste Val Accuracy')
ax1.set_title('Val Accuracy', fontsize=11)
ax1.set_xlim(0.35, 1.05)
ax1.invert_yaxis()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Rechter Plot: Val Loss
names_loss = [LABELS[r['experiment']] for r in sorted_by_loss]
losses     = [r['best_val_loss'] for r in sorted_by_loss]
cols_loss  = [COLORS[r['experiment']] for r in sorted_by_loss]
ax2.barh(names_loss, losses, color=cols_loss, height=0.55, edgecolor='white')
ax2.axvline(REF_LOSS, color='#888888', linestyle='--', linewidth=1.2)
for i, (val, name) in enumerate(zip(losses, names_loss)):
    ax2.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8.5)
ax2.set_xlabel('Beste Val Loss')
ax2.set_title('Val Loss', fontsize=11)
ax2.invert_yaxis()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

patch_spatial = mpatches.Patch(color='#C0392B', label='Räumlich')
patch_pixel   = mpatches.Patch(color='#2980B9', label='Pixelweise')
patch_ref     = mpatches.Patch(color='#888888', label='Referenz')
patch_comb    = mpatches.Patch(color='#1C1C1C', label='Kombiniert')
fig.legend(handles=[patch_spatial, patch_pixel, patch_ref, patch_comb],
           fontsize=8, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))

fig.suptitle('Abbildung 3 — Val Accuracy und Val Loss je Augmentierungskonfiguration', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(str(PATHS['plots'] / 'aug_fig3_doppelplot.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Gespeichert: aug_fig3_doppelplot.png')


# ══════════════════════════════════════════════════════════════════════
# Abbildung 4: Scatter — Val Accuracy vs. Val Loss
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))

for r in results:
    ax.scatter(r['best_val_loss'], r['best_val_acc'],
               color=COLORS[r['experiment']], s=120, zorder=3,
               edgecolors='white', linewidths=0.8)
    offset_x = 0.01
    offset_y = 0.008
    ax.annotate(LABELS[r['experiment']],
                xy=(r['best_val_loss'], r['best_val_acc']),
                xytext=(r['best_val_loss'] + offset_x, r['best_val_acc'] + offset_y),
                fontsize=8.5)

ax.axhline(REF_ACC,  color='#888888', linestyle='--', linewidth=1.0, alpha=0.6)
ax.axvline(REF_LOSS, color='#888888', linestyle='--', linewidth=1.0, alpha=0.6)

ax.set_xlabel('Beste Val Loss', fontsize=11)
ax.set_ylabel('Beste Val Accuracy', fontsize=11)
ax.set_title('Abbildung 4 — Val Accuracy vs. Val Loss je Konfiguration', fontsize=11, pad=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(str(PATHS['plots'] / 'aug_fig4_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Gespeichert: aug_fig4_scatter.png')

print('\nAlle Abbildungen gespeichert.')