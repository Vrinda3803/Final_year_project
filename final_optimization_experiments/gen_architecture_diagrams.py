"""
Generate Publication-Quality System Architecture and Bilateral Design Diagrams
in high resolution (300 DPI) matching the user's exact aesthetic style.
"""
import sys, os
sys.path.insert(0, r'D:\Documents\Autism_Project\.codex-deps')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, ConnectionPatch

ROOT = r'D:\Documents\Autism_Project'
FOE = os.path.join(ROOT, 'final_optimization_experiments')
FIG_DIR = os.path.join(ROOT, 'Figures') if os.path.exists(os.path.join(ROOT, 'Figures')) else FOE

# Ensure output directory exists
os.makedirs(FOE, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DIAGRAM 1: End-to-End System Architecture Diagram (RC-BiGCN)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def draw_system_architecture():
    fig = plt.figure(figsize=(18, 9), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    # Color Palette matching User Ref 1
    c_blue_light   = '#E0ECFB'
    c_blue_border  = '#3B82F6'
    c_green_light  = '#DCFCE7'
    c_green_border = '#22C55E'
    c_purple_light = '#F3E8FF'
    c_purple_border= '#A855F7'
    c_yellow_light = '#FEF9C3'
    c_yellow_border= '#EAB308'
    c_orange_light = '#FFEDD5'
    c_orange_border= '#F97316'
    c_card_bg      = '#FAF5FF'

    # ── Top Pipeline (5 Major Stages) ──────────────────────────────────────
    top_stages = [
        ("1. Dataset (MMASD)", "OpenPose 2D Skeletons\n1,374 Clips, 32 Cohorts", c_purple_light, c_purple_border, 3, 36, 16, 10),
        ("2. Input Representation", "Multi-Stream Features\n9 Channels ($x,y,v,a,b,q$)\n$T=64, V=25$", c_blue_light, c_blue_border, 23, 36, 16, 10),
        ("3. Preprocessing", "Root-Centric Normalization\nMid-Hip Centered\nTorso-Scaled Invariance", c_green_light, c_green_border, 43, 36, 16, 10),
        ("4. RC-BiGCN Model", "Reliability-Conditioned\nBilateral Graph Conv\n+ MS-TCN Backbone", c_orange_light, c_orange_border, 63, 36, 16, 10),
        ("5. Output Prediction", "10 Physical Therapy Classes\nMacro-F1: 0.7601\nAccuracy: 76.35%", c_yellow_light, c_yellow_border, 83, 36, 15, 10)
    ]

    for title, desc, bg, border, x, y, w, h in top_stages:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                             fc=bg, ec=border, lw=1.8, zorder=2)
        ax.add_patch(box)
        # Header banner inside box
        ax.plot([x, x+w], [y+h-2.8, y+h-2.8], color=border, lw=1.2, zorder=3)
        ax.text(x+w/2, y+h-1.4, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1E293B', zorder=4)
        ax.text(x+w/2, y+(h-2.8)/2, desc, ha='center', va='center', fontsize=8, color='#334155', zorder=4)

    # Top Connecting Arrows
    for i in range(len(top_stages)-1):
        x_start = top_stages[i][4] + top_stages[i][6] + 0.6
        x_end = top_stages[i+1][4] - 0.6
        y_mid = 41
        ax.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid),
                    arrowprops=dict(arrowstyle="-|>", color='#475569', lw=2.0, mutation_scale=14), zorder=5)

    # ── Downward Connecting Line from Box 4 to Detailed Architecture ──────
    ax.annotate('', xy=(50, 31.5), xytext=(71, 35.5),
                arrowprops=dict(arrowstyle="-|>", color=c_orange_border, lw=2.2, linestyle='--', mutation_scale=15), zorder=5)

    # ── Bottom Large Container: Detailed RC-BiGCN System Architecture ──────
    bottom_box = FancyBboxPatch((2, 2), 96, 29, boxstyle="round,pad=0.8,rounding_size=2.0",
                                fc=c_card_bg, ec='#C084FC', lw=2.0, linestyle='-', zorder=1)
    ax.add_patch(bottom_box)
    ax.text(6, 29.2, "Detailed Proposed Architecture: Reliability-Conditioned Bilateral Graph Convolutional Network (RC-BiGCN)",
            fontsize=11.5, fontweight='bold', color='#6B21A8')

    # Component A: Skeleton Input & Root Normalization
    box_a = FancyBboxPatch((4, 4.5), 14, 22.5, boxstyle="round,pad=0.4,rounding_size=1.0",
                           fc='#FFFFFF', ec='#94A3B8', lw=1.5, zorder=2)
    ax.add_patch(box_a)
    ax.text(11, 25.2, "Input Stream", ha='center', fontsize=9.5, fontweight='bold', color='#1E293B')
    # Mini skeleton sketch
    ax.plot([11, 11], [19.5, 23], color='#3B82F6', lw=2.5) # spine
    ax.plot([9, 13], [22, 22], color='#3B82F6', lw=2.0)   # shoulders
    ax.plot([9, 7.5], [22, 19], color='#3B82F6', lw=2.0)  # L-arm
    ax.plot([13, 14.5], [22, 19], color='#3B82F6', lw=2.0)# R-arm
    ax.plot([9.5, 8.5], [19.5, 15.5], color='#3B82F6', lw=2.0) # L-leg
    ax.plot([12.5, 13.5], [19.5, 15.5], color='#3B82F6', lw=2.0) # R-leg
    ax.scatter([11, 11, 9, 13, 7.5, 14.5, 9.5, 12.5, 8.5, 13.5],
               [23.5, 21.5, 22, 22, 19, 19, 19.5, 19.5, 15.5, 15.5], color='#EF4444', s=24, zorder=5)
    ax.text(11, 13.5, "9 Feature Channels:\n• Coords $(x, y)$\n• Velocity $(\\dot{x}, \\dot{y})$\n• Acceleration $(\\ddot{x}, \\ddot{y})$\n• Bone Vector $(\\tilde{x}, \\tilde{y})$\n• Confidence $q$",
            ha='center', fontsize=7.2, color='#334155', linespacing=1.3)
    ax.text(11, 6.0, "Root Centered at Mid-Hip\nTorso Scale Normalized", ha='center', fontsize=7.2, fontweight='bold', color='#047857')

    ax.annotate('', xy=(19.5, 16), xytext=(18.2, 16),
                arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.8, mutation_scale=12))

    # Component B: Graph Topology Construction (Fixed Anatomical + Dynamic Bilateral)
    box_b = FancyBboxPatch((20.5, 4.5), 18, 22.5, boxstyle="round,pad=0.4,rounding_size=1.0",
                           fc='#FFFFFF', ec='#3B82F6', lw=1.5, zorder=2)
    ax.add_patch(box_b)
    ax.text(29.5, 25.2, "Dual Graph Topology", ha='center', fontsize=9.5, fontweight='bold', color='#1E293B')
    
    # Sub-box B1: Fixed Anatomical
    b1 = FancyBboxPatch((21.5, 15.2), 16, 8.5, boxstyle="round,pad=0.3,rounding_size=0.8", fc='#F8FAFC', ec='#64748B', lw=1.2)
    ax.add_patch(b1)
    ax.text(29.5, 22.2, "Fixed Anatomical Graph", ha='center', fontsize=8, fontweight='bold', color='#475569')
    ax.text(29.5, 17.5, "24 Natural Kinematic Edges\nDegree Normalized $\\hat{A} = D^{-\\frac{1}{2}}AD^{-\\frac{1}{2}}$\nAdjacent Limb Propagations", ha='center', fontsize=7.0, color='#64748B')

    # Sub-box B2: Dynamic Bilateral
    b2 = FancyBboxPatch((21.5, 5.5), 16, 8.5, boxstyle="round,pad=0.3,rounding_size=0.8", fc='#FEF2F2', ec='#EF4444', lw=1.2)
    ax.add_patch(b2)
    ax.text(29.5, 12.5, "11 Bilateral Pair Edges", ha='center', fontsize=8, fontweight='bold', color='#DC2626')
    ax.text(29.5, 8.0, "Homologous Left-Right Limbs\nWrists, Elbows, Knees, Feet\nCross-Midline Symmetry Edges", ha='center', fontsize=7.0, color='#991B1B')

    ax.annotate('', xy=(40, 16), xytext=(38.7, 16),
                arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.8, mutation_scale=12))

    # Component C: Spatial GCN Block with Reliability-Conditioned Bilateral Conv
    box_c = FancyBboxPatch((41, 4.5), 23, 22.5, boxstyle="round,pad=0.4,rounding_size=1.0",
                           fc='#FFFFFF', ec='#F97316', lw=1.5, zorder=2)
    ax.add_patch(box_c)
    ax.text(52.5, 25.2, "RC-Bilateral Conv Block", ha='center', fontsize=9.5, fontweight='bold', color='#C2410C')
    
    c_sub = FancyBboxPatch((42, 6.0), 21, 17.8, boxstyle="round,pad=0.3,rounding_size=0.8", fc='#FFF7ED', ec='#FB923C', lw=1.2)
    ax.add_patch(c_sub)
    ax.text(52.5, 22.0, "Dynamic Bilateral Edge MLP:", ha='center', fontsize=8, fontweight='bold', color='#9A3412')
    ax.text(52.5, 19.5, "$w_{uv}^{(t)} = \\sigma(\\text{MLP}[\\mathbf{h}_u, \\mathbf{h}_v, q_u, q_v, |\\mathbf{h}_u-\\mathbf{h}_v|, \\mathbf{h}_u \\odot \\mathbf{h}_v])$",
            ha='center', fontsize=7.4, fontweight='bold', color='#EA580C')
    ax.text(52.5, 16.5, "Reliability Conditioning:\n$q_u, q_v \\in [0, 1]$ directly gate message flow.\nDownweights noisy/occluded joints.", ha='center', fontsize=7.2, color='#7C2D12')
    ax.text(52.5, 12.0, "Bilateral Message Passing:\n$\\mathbf{h}_u \\leftarrow \\mathbf{h}_u + \\sum_v w_{uv}^{(t)} \\cdot \\mathbf{h}_v$\nResidual Skip + LayerNorm + SiLU", ha='center', fontsize=7.2, color='#334155')
    ax.text(52.5, 7.8, "Joint Mean-Pooling ($V=25 \\rightarrow 1$)", ha='center', fontsize=7.5, fontweight='bold', color='#0284C7')

    ax.annotate('', xy=(65.5, 16), xytext=(64.2, 16),
                arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.8, mutation_scale=12))

    # Component D: Temporal Backbone (MS-TCN)
    box_d = FancyBboxPatch((66.5, 4.5), 16, 22.5, boxstyle="round,pad=0.4,rounding_size=1.0",
                           fc='#FFFFFF', ec='#10B981', lw=1.5, zorder=2)
    ax.add_patch(box_d)
    ax.text(74.5, 25.2, "MS-TCN Temporal Backbone", ha='center', fontsize=9.2, fontweight='bold', color='#047857')

    # Dilated branches
    d_branches = [("Branch 1: Conv1D (k=3, d=1)", 19.8),
                  ("Branch 2: Conv1D (k=5, d=2)", 15.5),
                  ("Branch 3: Conv1D (k=7, d=3)", 11.2)]
    for lbl, y_pos in d_branches:
        p_box = FancyBboxPatch((67.5, y_pos), 14, 3.2, boxstyle="round,pad=0.2,rounding_size=0.5", fc='#ECFDF5', ec='#34D399', lw=1.0)
        ax.add_patch(p_box)
        ax.text(74.5, y_pos+1.6, lbl, ha='center', va='center', fontsize=7.2, fontweight='bold', color='#065F46')

    ax.text(74.5, 8.2, "Concat Parallel Branches\n+ Residual Projection\nGlobal Temporal Avg Pool", ha='center', fontsize=7.2, color='#1E293B')

    ax.annotate('', xy=(84, 16), xytext=(82.7, 16),
                arrowprops=dict(arrowstyle="-|>", color='#475569', lw=1.8, mutation_scale=12))

    # Component E: Classification Head
    box_e = FancyBboxPatch((85, 4.5), 11.5, 22.5, boxstyle="round,pad=0.4,rounding_size=1.0",
                           fc='#FFFFFF', ec='#6366F1', lw=1.5, zorder=2)
    ax.add_patch(box_e)
    ax.text(90.75, 25.2, "Classification", ha='center', fontsize=9.2, fontweight='bold', color='#4338CA')
    
    e_layers = [
        ("Linear Projection", 20.0, '#EEF2FF'),
        ("LayerNorm + SiLU", 15.8, '#EEF2FF'),
        ("Dropout (p=0.3)", 11.6, '#EEF2FF'),
        ("10 Activity Classes\nSoftmax", 6.0, '#FEF08A')
    ]
    for lbl, y_pos, b_col in e_layers:
        h_h = 4.2 if "Softmax" in lbl else 3.2
        l_box = FancyBboxPatch((86, y_pos), 9.5, h_h, boxstyle="round,pad=0.2,rounding_size=0.5", fc=b_col, ec='#818CF8', lw=1.0)
        ax.add_patch(l_box)
        ax.text(90.75, y_pos + h_h/2, lbl, ha='center', va='center', fontsize=7.2, fontweight='bold', color='#1E293B')

    plt.tight_layout()
    out_path = os.path.join(FOE, 'fig_system_architecture.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"System architecture diagram saved: {out_path}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DIAGRAM 2: Bilateral System Design & Reliability Conditioning Diagram
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def draw_bilateral_system_design():
    fig = plt.figure(figsize=(18, 9), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    c_blue_bg    = '#EFF6FF'
    c_blue_border= '#2563EB'
    c_green_bg   = '#F0FDF4'
    c_green_border= '#16A34A'
    c_purple_bg  = '#FAF5FF'
    c_purple_border='#9333EA'
    c_amber_bg   = '#FFFBEB'

    # Title Banner
    ax.text(50, 47.5, "Bilateral System Design: Dynamic & Reliability-Conditioned Graph Learning",
            ha='center', va='center', fontsize=14, fontweight='bold', color='#1E293B')

    # ── Left Input Box (2D Skeleton Sequence + Reliability) ─────────────────
    in_box = FancyBboxPatch((2, 10), 16, 34, boxstyle="round,pad=0.6,rounding_size=1.5",
                            fc='#FFFFFF', ec='#64748B', lw=1.8, zorder=2)
    ax.add_patch(in_box)
    ax.text(10, 41.5, "2D Skeleton Input\n(MMASD Dataset)", ha='center', fontsize=10.5, fontweight='bold', color='#0F172A')

    # Draw Skeleton with confidence colormap
    ax.plot([10, 10], [28, 34], color='#0284C7', lw=3.0) # spine
    ax.plot([7, 13], [32.5, 32.5], color='#0284C7', lw=2.5) # shoulders
    ax.plot([7, 5.5], [32.5, 27], color='#16A34A', lw=2.5) # L-arm (high rel)
    ax.plot([13, 14.5], [32.5, 27], color='#EA580C', lw=2.5, linestyle='--') # R-arm (occluded/low rel)
    ax.plot([8, 6.5], [28, 20], color='#16A34A', lw=2.5) # L-leg
    ax.plot([12, 13.5], [28, 20], color='#16A34A', lw=2.5) # R-leg
    # Joint markers
    joints = [(10, 34.5), (10, 31), (7, 32.5), (13, 32.5), (5.5, 27), (14.5, 27), (8, 28), (12, 28), (6.5, 20), (13.5, 20)]
    colors_j = ['#22C55E', '#22C55E', '#22C55E', '#F59E0B', '#22C55E', '#EF4444', '#22C55E', '#22C55E', '#22C55E', '#22C55E']
    for (jx, jy), jc in zip(joints, colors_j):
        ax.scatter(jx, jy, color=jc, s=45, zorder=5, edgecolor='#0F172A')

    ax.text(10, 18.0, "Shape: $T \\times V \\times 2$\n($T=64, V=25$)", ha='center', fontsize=8.5, fontweight='bold', color='#334155')
    
    # Confidence Grid Representation
    ax.text(10, 14.5, "Joint Confidence Scores $Q$:", ha='center', fontsize=8.0, fontweight='bold', color='#0F172A')
    for row in range(2):
        for col in range(5):
            val_col = '#86EFAC' if (row*5+col) not in [3, 5] else '#FCA5A5'
            rect = patches.Rectangle((4.5 + col*2.2, 11.2 - row*1.8), 1.8, 1.4, fc=val_col, ec='#475569', lw=0.8)
            ax.add_patch(rect)

    # ── Pathway 1 (Top): Dynamic Bilateral Graph Learning (DBGL) ───────────
    dbgl_box = FancyBboxPatch((23, 26.5), 44, 18, boxstyle="round,pad=0.6,rounding_size=1.5",
                              fc=c_blue_bg, ec=c_blue_border, lw=2.0, zorder=2)
    ax.add_patch(dbgl_box)
    ax.text(25, 42.5, "Dynamic Bilateral Graph Learning (DBGL)", fontsize=11, fontweight='bold', color='#1D4ED8')
    ax.text(62, 42.5, "Captures Motor Coordination Across Midline", fontsize=8, fontstyle='italic', color='#1E40AF')

    # Step 1: 11 Bilateral Pairs definition
    p1 = FancyBboxPatch((24.5, 28.5), 12, 12.5, boxstyle="round,pad=0.3,rounding_size=0.8", fc='#FFFFFF', ec='#3B82F6', lw=1.2)
    ax.add_patch(p1)
    ax.text(30.5, 39.2, "11 Bilateral Pairs ($\\mathcal{B}$)", ha='center', fontsize=8.5, fontweight='bold', color='#1E293B')
    ax.text(30.5, 33.5, "• Wrists (4, 7)\n• Elbows (3, 6)\n• Shoulders (2, 5)\n• Knees (9, 12)\n• Ankles (10, 13)\n• Feet & Eyes",
            ha='center', fontsize=7.2, color='#334155', linespacing=1.3)

    # Step 2: Dynamic Edge Weight calculation
    p2 = FancyBboxPatch((38.5, 28.5), 14.5, 12.5, boxstyle="round,pad=0.3,rounding_size=0.8", fc='#FFFFFF', ec='#3B82F6', lw=1.2)
    ax.add_patch(p2)
    ax.text(45.75, 39.2, "Time-Varying Relations", ha='center', fontsize=8.5, fontweight='bold', color='#1E293B')
    ax.text(45.75, 34.0, "$w_{uv}^{(t)} = \\sigma(\\text{MLP}[\\mathbf{h}_u, \\mathbf{h}_v])$", ha='center', fontsize=8.0, fontweight='bold', color='#2563EB')
    ax.text(45.75, 30.5, "Sample-adaptive edge weights\nvary across repetitions\n(In-phase vs. Alternating)", ha='center', fontsize=7.0, color='#475569')

    # Step 3: Bilateral Features F_bil^d
    p3 = FancyBboxPatch((55, 28.5), 10.5, 12.5, boxstyle="round,pad=0.3,rounding_size=0.8", fc='#FFFFFF', ec='#3B82F6', lw=1.2)
    ax.add_patch(p3)
    ax.text(60.25, 39.2, "Bilateral Feats", ha='center', fontsize=8.5, fontweight='bold', color='#1E293B')
    ax.text(60.25, 34.0, "$\\mathbf{F}_{\\text{bil}}^{\\text{dyn}}$", ha='center', fontsize=11, fontweight='bold', color='#1D4ED8')
    ax.text(60.25, 30.5, "Aggregated\nL-R Limb\nDynamics", ha='center', fontsize=7.2, color='#475569')

    # Connect internal DBGL steps
    ax.annotate('', xy=(38.0, 34.5), xytext=(36.7, 34.5), arrowprops=dict(arrowstyle="-|>", color='#2563EB', lw=1.5))
    ax.annotate('', xy=(54.5, 34.5), xytext=(53.2, 34.5), arrowprops=dict(arrowstyle="-|>", color='#2563EB', lw=1.5))

    # ── Pathway 2 (Bottom): Reliability-Conditioned Bilateral Graph Learning (RCBGL)
    rcbgl_box = FancyBboxPatch((23, 6.0), 44, 18, boxstyle="round,pad=0.6,rounding_size=1.5",
                               fc=c_green_bg, ec=c_green_border, lw=2.0, zorder=2)
    ax.add_patch(rcbgl_box)
    ax.text(25, 22.0, "Reliability-Conditioned Bilateral Graph Learning (RCBGL)", fontsize=11, fontweight='bold', color='#15803D')
    ax.text(62, 22.0, "Attenuates Noise & Occlusion Gaps", fontsize=8, fontstyle='italic', color='#166534')

    # Step 1: Joint Confidence Injection
    q1 = FancyBboxPatch((24.5, 8.0), 12, 12.5, boxstyle="round,pad=0.3,rounding_size=0.8", fc='#FFFFFF', ec='#22C55E', lw=1.2)
    ax.add_patch(q1)
    ax.text(30.5, 18.7, "Confidence Conditioning", ha='center', fontsize=8.2, fontweight='bold', color='#1E293B')
    ax.text(30.5, 14.0, "Direct Inputs $(q_u, q_v)$:\n• $q \\to 1$: High Trust\n• $q \\to 0$: Occluded\nDirectly modulates edge MLP",
            ha='center', fontsize=7.2, color='#334155', linespacing=1.3)

    # Step 2: Reliability-Conditioned Message Passing
    q2 = FancyBboxPatch((38.5, 8.0), 14.5, 12.5, boxstyle="round,pad=0.3,rounding_size=0.8", fc='#FFFFFF', ec='#22C55E', lw=1.2)
    ax.add_patch(q2)
    ax.text(45.75, 18.7, "Gated Message Passing", ha='center', fontsize=8.2, fontweight='bold', color='#1E293B')
    ax.text(45.75, 14.2, "$w_{uv} \\leftarrow w_{uv} \\cdot \\min(q_u, q_v)$", ha='center', fontsize=8.0, fontweight='bold', color='#15803D')
    ax.text(45.75, 10.5, "Suppresses corrupt joints\nPrevents noise propagation\nacross body midline", ha='center', fontsize=7.0, color='#475569')

    # Step 3: Reliability-Aware Features F_bil^rel
    q3 = FancyBboxPatch((55, 8.0), 10.5, 12.5, boxstyle="round,pad=0.3,rounding_size=0.8", fc='#FFFFFF', ec='#22C55E', lw=1.2)
    ax.add_patch(q3)
    ax.text(60.25, 18.7, "Reliable Feats", ha='center', fontsize=8.2, fontweight='bold', color='#1E293B')
    ax.text(60.25, 13.8, "$\\mathbf{F}_{\\text{bil}}^{\\text{rel}}$", ha='center', fontsize=11, fontweight='bold', color='#15803D')
    ax.text(60.25, 10.2, "Occlusion-\nRobust\nKinematics", ha='center', fontsize=7.2, color='#475569')

    # Connect internal RCBGL steps
    ax.annotate('', xy=(38.0, 14.0), xytext=(36.7, 14.0), arrowprops=dict(arrowstyle="-|>", color='#16A34A', lw=1.5))
    ax.annotate('', xy=(54.5, 14.0), xytext=(53.2, 14.0), arrowprops=dict(arrowstyle="-|>", color='#16A34A', lw=1.5))

    # Input to Pathway Connections
    ax.annotate('', xy=(22.5, 35.5), xytext=(18.2, 28), arrowprops=dict(arrowstyle="-|>", color='#2563EB', lw=1.8))
    ax.annotate('', xy=(22.5, 15.0), xytext=(18.2, 26), arrowprops=dict(arrowstyle="-|>", color='#16A34A', lw=1.8))

    # ── Fusion Box: Asymmetry & Co-Activation Interaction ──────────────────
    fuse_box = FancyBboxPatch((70.5, 16.5), 14, 18, boxstyle="round,pad=0.5,rounding_size=1.2",
                              fc=c_purple_bg, ec=c_purple_border, lw=1.8, zorder=2)
    ax.add_patch(fuse_box)
    ax.text(77.5, 32.2, "Cross-Limb Fusion", ha='center', fontsize=10, fontweight='bold', color='#7E22CE')
    ax.text(77.5, 27.5, "Asymmetry Term:\n$|\\mathbf{h}_u - \\mathbf{h}_v|$\n\nCo-Activation Term:\n$\\mathbf{h}_u \\odot \\mathbf{h}_v$",
            ha='center', fontsize=7.8, fontweight='bold', color='#6B21A8')
    ax.text(77.5, 19.5, "$\\mathbf{F}_{\\text{fused}} = [\\mathbf{F}_{\\text{anat}}, \\mathbf{F}_{\\text{bil}}]$",
            ha='center', fontsize=8.2, fontweight='bold', color='#4C1D95')

    # Pathway outputs to Fusion Box
    ax.annotate('', xy=(70.0, 28), xytext=(66.0, 34), arrowprops=dict(arrowstyle="-|>", color='#2563EB', lw=1.8))
    ax.annotate('', xy=(70.0, 23), xytext=(66.0, 15), arrowprops=dict(arrowstyle="-|>", color='#16A34A', lw=1.8))

    # ── Output Classifier Head ─────────────────────────────────────────────
    head_box = FancyBboxPatch((87.5, 14), 10.5, 23, boxstyle="round,pad=0.5,rounding_size=1.2",
                             fc='#FEFCE8', ec='#CA8A04', lw=1.8, zorder=2)
    ax.add_patch(head_box)
    ax.text(92.75, 34.5, "Classifier Head", ha='center', fontsize=9.5, fontweight='bold', color='#854D0E')
    
    cls_layers = [
        ("MS-TCN Blocks\n(k=3, 5, 7)", 28.5),
        ("Temporal Pooling", 24.2),
        ("FC + LayerNorm", 20.0),
        ("Dropout (0.3)", 15.8)
    ]
    for lbl, y_pos in cls_layers:
        b_sub = FancyBboxPatch((88.5, y_pos), 8.5, 3.4, boxstyle="round,pad=0.2,rounding_size=0.4", fc='#FFFFFF', ec='#EAB308', lw=1.0)
        ax.add_patch(b_sub)
        ax.text(92.75, y_pos+1.7, lbl, ha='center', va='center', fontsize=6.8, fontweight='bold', color='#1E293B')

    # Final Activity Prediction Label
    pred_box = FancyBboxPatch((86.5, 4.5), 12.5, 7.5, boxstyle="round,pad=0.3,rounding_size=0.6", fc='#DCFCE7', ec='#16A34A', lw=1.5)
    ax.add_patch(pred_box)
    ax.text(92.75, 9.8, "Predicted Activity", ha='center', fontsize=8.5, fontweight='bold', color='#15803D')
    ax.text(92.75, 6.5, "Arm Swings, Squats,\nMarching, Twists, etc.", ha='center', fontsize=7.2, color='#166534')

    ax.annotate('', xy=(87.0, 25.5), xytext=(85.0, 25.5), arrowprops=dict(arrowstyle="-|>", color='#7E22CE', lw=2.0))
    ax.annotate('', xy=(92.75, 12.2), xytext=(92.75, 13.8), arrowprops=dict(arrowstyle="-|>", color='#16A34A', lw=1.8))

    # ── Bottom Advantages Banner (matching user's Ref 2) ───────────────────
    adv_box = FancyBboxPatch((4, 2.0), 80, 5.0, boxstyle="round,pad=0.3,rounding_size=0.8",
                             fc='#F3E8FF', ec='#C084FC', lw=1.4)
    ax.add_patch(adv_box)
    ax.text(12, 4.5, "Key Contributions:", fontsize=8.5, fontweight='bold', color='#6B21A8')
    ax.text(26, 4.5, "✔ Dynamic Bilateral Learning captures left-right coordination (+3.35 pp gain)", fontsize=7.8, color='#4A044E')
    ax.text(58, 4.5, "✔ Reliability-Conditioned Gating mitigates OpenPose tracking occlusion noise", fontsize=7.8, color='#4A044E')

    plt.tight_layout()
    out_path = os.path.join(FOE, 'fig_bilateral_system_design.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bilateral system design diagram saved: {out_path}")

if __name__ == '__main__':
    draw_system_architecture()
    draw_bilateral_system_design()
    print("Both diagrams generated in high resolution.")
