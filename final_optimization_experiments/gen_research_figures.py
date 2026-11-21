"""
Generate specialized research visualizations:
1. Reliability diagrams for v1 and v2 (Calibration Analysis)
2. 2x2 Factorial Interaction Plot: Root-Norm x Bilateral Relational Modeling
3. Efficiency Frontier: Accuracy vs. Inference Latency (ms) and Parameter Count
4. Per-Class Precision, Recall, F1 grouped comparative chart (C0 vs v1 vs v2)
"""
import sys, os, json
sys.path.insert(0, r'D:\Documents\Autism_Project\.codex-deps')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = r'D:\Documents\Autism_Project'
FOE = os.path.join(ROOT, 'final_optimization_experiments')

# Load comprehensive eval data
with open(os.path.join(FOE, 'comprehensive_research_evaluation.json')) as f:
    eval_data = json.load(f)

classes = ['as', 'bs', 'ce', 'dr', 'mfs', 'ms', 'sac', 'sq', 'tr', 'tw']

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Reliability Diagrams: v1 vs v2
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (m_key, title, col) in enumerate([
    ('RC_BiGCN_v1', 'RC-BiGCN v1 (Proposed Base)', '#F97316'),
    ('RC_BiGCN_v2', 'RC-BiGCN v2 (Optimized Multi-Stream)', '#10B981')
]):
    ax = axes[idx]
    cal = eval_data['calibration_data'][m_key]
    
    bin_centers = np.array(cal['bin_centers'])
    bin_accs = np.array(cal['bin_accs'])
    bin_confs = np.array(cal['bin_confs'])
    bin_counts = np.array(cal['bin_counts'])
    ece = cal['ece']
    
    # Bar plot of empirical accuracy vs bin confidence
    width = 0.08
    ax.bar(bin_centers, bin_accs, width=width, color=col, alpha=0.65, edgecolor='#1E293B', label='Outputs (Empirical Accuracy)')
    
    # Overconfidence gap
    gap = np.maximum(0, bin_confs - bin_accs)
    ax.bar(bin_centers, gap, bottom=bin_accs, width=width, color='#EF4444', alpha=0.35, hatch='//', edgecolor='#EF4444', label='Calibration Gap')
    
    # Ideal calibration line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.8, label='Perfect Calibration ($y = x$)')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Confidence (Predicted Probability)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Empirical Accuracy', fontsize=11, fontweight='bold')
    ax.set_title(f'{title}\nECE = {ece:.4f} ({ece*100:.2f}%)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=9.5)
    
    # Add count tags
    for bc, ba, cnt in zip(bin_centers, bin_accs, bin_counts):
        if cnt > 0:
            ax.text(bc, 0.02, f'n={cnt}', ha='center', va='bottom', fontsize=7.5, rotation=90, color='#334155')

plt.suptitle('Reliability Diagrams (Model Calibration Analysis)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FOE, 'fig9_reliability_diagrams_v1_v2.png'), dpi=160, bbox_inches='tight')
plt.close()
print("Saved fig9_reliability_diagrams_v1_v2.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Factorial Interaction Plot: Root-Norm x Bilateral Relational Modeling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(9, 6))

# Data points:
# Without Root-Norm:
#   Non-Bilateral (A0 Fixed Topo STGT): 57.13%
#   Bilateral (A8 Dynamic Soft Rel): 65.94%
# With Root-Norm:
#   Non-Bilateral (C0 Fixed Graph + MS-TCN): 68.70%
#   Bilateral (RC-BiGCN v1 / v2): 75.33% / 76.35%

x = [0, 1]
x_labels = ['Raw Coordinates\n(No Root-Centric Norm)', 'Root-Centric Normalized\n(Mid-Hip Centered + Torso Scale)']

y_non_bilateral = [57.13, 68.70]
y_bilateral_v1  = [65.94, 75.33]
y_bilateral_v2  = [65.94, 76.35]

ax.plot(x, y_non_bilateral, 'o-', color='#64748B', lw=2.5, ms=8, label='Non-Bilateral (Fixed Graph Topology)')
ax.plot(x, y_bilateral_v1, 's-', color='#F97316', lw=2.5, ms=8, label='Bilateral Relational (RC-BiGCN v1)')
ax.plot(x, y_bilateral_v2, '^--', color='#10B981', lw=2.5, ms=8, label='Bilateral Multi-Stream (RC-BiGCN v2)')

# Annotate points
ax.text(0, 57.13 - 1.2, '57.13%\n(A0)', ha='center', va='top', fontsize=9, fontweight='bold', color='#475569')
ax.text(1, 68.70 - 1.2, '68.70%\n(C0)', ha='center', va='top', fontsize=9, fontweight='bold', color='#475569')
ax.text(0, 65.94 + 0.8, '65.94%\n(A8)', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#C2410C')
ax.text(1, 75.33 - 1.4, '75.33%\n(v1)', ha='center', va='top', fontsize=9, fontweight='bold', color='#C2410C')
ax.text(1, 76.35 + 0.8, '76.35%\n(v2)', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#047857')

# Marginal gain arrows
ax.annotate('+11.57 pp (Root-Norm Gain)', xy=(0.5, 62.91), xytext=(0.5, 60.5),
            ha='center', fontsize=9, fontweight='bold', color='#334155',
            arrowprops=dict(arrowstyle='->', color='#334155', lw=1.2))

ax.annotate('+8.81 pp (Bilateral Gain without Norm)', xy=(0, 61.5), xytext=(-0.15, 61.5),
            ha='right', fontsize=8.5, fontweight='bold', color='#C2410C',
            arrowprops=dict(arrowstyle='->', color='#C2410C', lw=1.2))

ax.annotate('+7.65 pp (Bilateral Gain with Norm)', xy=(1, 72.5), xytext=(1.15, 72.5),
            ha='left', fontsize=8.5, fontweight='bold', color='#047857',
            arrowprops=dict(arrowstyle='->', color='#047857', lw=1.2))

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
ax.set_ylabel('Pooled OOF Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_ylim(52, 80)
ax.set_title('Factorial Interaction Analysis:\nRoot-Centric Normalization vs. Bilateral Relational Modeling', fontsize=13, fontweight='bold')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(FOE, 'fig10_interaction_analysis.png'), dpi=160, bbox_inches='tight')
plt.close()
print("Saved fig10_interaction_analysis.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Efficiency Frontier: Accuracy vs. Inference Latency & Parameter Count
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

eff = eval_data['efficiency_results']
models_eff = [
    ('C0_FixedGraph', 'C0: Fixed Graph (Ablation)', 68.70, eff['C0_FixedGraph']['latency_ms'], eff['C0_FixedGraph']['params'], '#64748B'),
    ('C3_RelConditioned', 'C3: Rel-Conditioned (Ablation)', 73.87, eff['C3_RelConditioned']['latency_ms'], eff['C3_RelConditioned']['params'], '#3B82F6'),
    ('RC_BiGCN_v1', 'RC-BiGCN v1 (Proposed)', 75.33, eff['RC_BiGCN_v1']['latency_ms'], eff['RC_BiGCN_v1']['params'], '#F97316'),
    ('RC_BiGCN_v2', 'RC-BiGCN v2 (Optimized)', 76.35, eff['RC_BiGCN_v2']['latency_ms'], eff['RC_BiGCN_v2']['params'], '#10B981'),
]

# Panel 1: Accuracy vs Latency
for key, lbl, acc, lat, par, col in models_eff:
    ax1.scatter(lat, acc, s=par/600, color=col, alpha=0.85, edgecolors='#1E293B', lw=1.5, label=f'{lbl} ({par:,} params)')
    ax1.text(lat, acc + 0.35, f'{acc:.2f}%\n({lat:.2f} ms)', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax1.set_xlabel('Inference Latency per Clip (ms, CPU) [Lower is Better]', fontsize=10.5, fontweight='bold')
ax1.set_ylabel('Pooled OOF Accuracy (%) [Higher is Better]', fontsize=10.5, fontweight='bold')
ax1.set_ylim(67, 78.5)
ax1.set_xlim(3.5, 9.5)
ax1.set_title('Accuracy vs. Inference Latency (Bubble size $\\propto$ Parameter count)', fontsize=11.5, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right', fontsize=8.5)

# Panel 2: Accuracy vs Parameter Count
for key, lbl, acc, lat, par, col in models_eff:
    ax2.scatter(par / 1000, acc, s=lat*25, color=col, alpha=0.85, edgecolors='#1E293B', lw=1.5, label=f'{lbl} ({lat:.1f} ms)')
    ax2.text(par / 1000, acc + 0.35, f'{acc:.2f}%\n({par:,})', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax2.set_xlabel('Parameter Count (kilo-parameters, kParams) [Lower is Better]', fontsize=10.5, fontweight='bold')
ax2.set_ylabel('Pooled OOF Accuracy (%) [Higher is Better]', fontsize=10.5, fontweight='bold')
ax2.set_ylim(67, 78.5)
ax2.set_xlim(40, 280)
ax2.set_title('Accuracy vs. Model Size (Bubble size $\\propto$ Latency)', fontsize=11.5, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='lower right', fontsize=8.5)

plt.suptitle('Computational Efficiency & Accuracy Tradeoff', fontsize=13.5, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FOE, 'fig11_efficiency_tradeoff.png'), dpi=160, bbox_inches='tight')
plt.close()
print("Saved fig11_efficiency_tradeoff.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Per-Class Precision, Recall, and F1: C0 vs v1 vs v2
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

x = np.arange(len(classes))
w = 0.26

p_c0 = [eval_data['per_class_results']['C0_FixedGraph']['classes'][c]['precision'] for c in classes]
p_v1 = [eval_data['per_class_results']['RC_BiGCN_v1']['classes'][c]['precision'] for c in classes]
p_v2 = [eval_data['per_class_results']['RC_BiGCN_v2']['classes'][c]['precision'] for c in classes]

r_c0 = [eval_data['per_class_results']['C0_FixedGraph']['classes'][c]['recall'] for c in classes]
r_v1 = [eval_data['per_class_results']['RC_BiGCN_v1']['classes'][c]['recall'] for c in classes]
r_v2 = [eval_data['per_class_results']['RC_BiGCN_v2']['classes'][c]['recall'] for c in classes]

f_c0 = [eval_data['per_class_results']['C0_FixedGraph']['classes'][c]['f1'] for c in classes]
f_v1 = [eval_data['per_class_results']['RC_BiGCN_v1']['classes'][c]['f1'] for c in classes]
f_v2 = [eval_data['per_class_results']['RC_BiGCN_v2']['classes'][c]['f1'] for c in classes]

# Precision
axes[0].bar(x - w, p_c0, width=w, color='#64748B', alpha=0.85, label='C0: Non-Bilateral Fixed Graph (Macro P = 0.686)')
axes[0].bar(x,     p_v1, width=w, color='#F97316', alpha=0.85, label='RC-BiGCN v1 (Macro P = 0.742)')
axes[0].bar(x + w, p_v2, width=w, color='#10B981', alpha=0.85, label='RC-BiGCN v2 (Macro P = 0.765)')
axes[0].set_ylabel('Precision', fontsize=10.5, fontweight='bold')
axes[0].set_ylim(0, 1.05)
axes[0].set_title('Per-Class Precision Comparison', fontsize=11, fontweight='bold')
axes[0].grid(axis='y', linestyle=':', alpha=0.6)
axes[0].legend(loc='lower right', fontsize=8.5)

# Recall
axes[1].bar(x - w, r_c0, width=w, color='#64748B', alpha=0.85, label='C0: Non-Bilateral Fixed Graph (Macro R = 0.675)')
axes[1].bar(x,     r_v1, width=w, color='#F97316', alpha=0.85, label='RC-BiGCN v1 (Macro R = 0.739)')
axes[1].bar(x + w, r_v2, width=w, color='#10B981', alpha=0.85, label='RC-BiGCN v2 (Macro R = 0.767)')
axes[1].set_ylabel('Recall', fontsize=10.5, fontweight='bold')
axes[1].set_ylim(0, 1.05)
axes[1].set_title('Per-Class Recall Comparison', fontsize=11, fontweight='bold')
axes[1].grid(axis='y', linestyle=':', alpha=0.6)
axes[1].legend(loc='lower right', fontsize=8.5)

# F1
axes[2].bar(x - w, f_c0, width=w, color='#64748B', alpha=0.85, label='C0: Non-Bilateral Fixed Graph (Macro F1 = 0.671)')
axes[2].bar(x,     f_v1, width=w, color='#F97316', alpha=0.85, label='RC-BiGCN v1 (Macro F1 = 0.736)')
axes[2].bar(x + w, f_v2, width=w, color='#10B981', alpha=0.85, label='RC-BiGCN v2 (Macro F1 = 0.760)')
axes[2].set_ylabel('F1 Score', fontsize=10.5, fontweight='bold')
axes[2].set_ylim(0, 1.05)
axes[2].set_title('Per-Class F1 Score Comparison', fontsize=11, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels([c.upper() for c in classes], fontsize=11, fontweight='bold')
axes[2].set_xlabel('Activity Class Code', fontsize=11, fontweight='bold')
axes[2].grid(axis='y', linestyle=':', alpha=0.6)
axes[2].legend(loc='lower right', fontsize=8.5)

plt.suptitle('Detailed Per-Class Performance Decomposition Across Evolution Stages', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FOE, 'fig12_perclass_precision_recall_f1.png'), dpi=160, bbox_inches='tight')
plt.close()
print("Saved fig12_perclass_precision_recall_f1.png")

print("\nAll specialized research figures generated successfully.")
