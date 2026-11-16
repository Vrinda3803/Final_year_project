"""
Generates high-resolution SVG publication figures for Stage 2 PR-STGT research contributions.
"""
from __future__ import annotations
import sys, json, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import numpy as np

CLASSES = ["as", "bs", "ce", "dr", "mfs", "ms", "sac", "sq", "tr", "tw"]


def create_ablation_bar_chart(summary_data: dict, out_path: Path):
    """
    Generates fig3_ablation_comparison.svg:
    Grouped bar chart showing Mean Fold Accuracy and Macro-F1 across E0 to E5.
    """
    experiments = [
        ("E0: Vanilla STGT", 37.86, 29.11, 10.81, 7.91),
        ("E1: RA-STGT v1", 46.47, 41.37, 7.84, 13.28),
    ]
    
    # Add E2-E5 from summary data if present
    name_map = {
        "e2_topology_only": ("E2: PR-GAT Topology", 5.2, 8.4),
        "e3_phase_only": ("E3: Phase Dynamics", 6.1, 7.5),
        "e4_margin_only": ("E4: Margin Loss", 5.8, 8.1),
        "e5_full_pr_stgt": ("E5: Full PR-STGT", 4.9, 6.2)
    }
    
    for k, (disp_name, default_std_acc, default_std_f1) in name_map.items():
        if k in summary_data:
            s = summary_data[k]
            acc = s["mean_fold_acc"] * 100
            f1 = s["mean_fold_macro_f1"] * 100
            s_acc = s.get("std_fold_acc", default_std_acc / 100) * 100
            s_f1 = s.get("std_fold_macro_f1", default_std_f1 / 100) * 100
            experiments.append((disp_name, acc, f1, s_acc, s_f1))
            
    width = 900
    height = 500
    margin = {"top": 60, "right": 50, "bottom": 80, "left": 80}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    
    n_groups = len(experiments)
    group_width = plot_w / n_groups
    bar_width = group_width * 0.35
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">',
        f'<text x="{width/2}" y="36" text-anchor="middle" font-size="20" font-weight="bold" fill="#1e293b">Stage 2 Controlled Ablation: Accuracy &amp; Macro-F1 Progression (E0 to E5)</text>',
        
        # Grid lines
        '<g stroke="#e2e8f0" stroke-width="1">',
    ]
    
    # Y-axis ticks (0% to 70%)
    for pct in range(0, 80, 10):
        y_pos = margin["top"] + plot_h - (pct / 70.0) * plot_h
        svg.append(f'<line x1="{margin["left"]}" y1="{y_pos}" x2="{margin["left"] + plot_w}" y2="{y_pos}" stroke-dasharray="4,4" />')
        svg.append(f'<text x="{margin["left"] - 12}" y="{y_pos + 4}" text-anchor="end" font-size="12" fill="#64748b">{pct}%</text>')
    svg.append('</g>')
    
    # Axes
    svg.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" stroke="#334155" stroke-width="2" />')
    svg.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="#334155" stroke-width="2" />')
    
    # Legend
    leg_x = margin["left"] + plot_w - 240
    leg_y = margin["top"] + 10
    svg.append(f'<rect x="{leg_x}" y="{leg_y}" width="230" height="40" fill="#f8fafc" stroke="#cbd5e1" rx="4" />')
    svg.append(f'<rect x="{leg_x + 15}" y="{leg_y + 14}" width="16" height="12" fill="#2563eb" rx="2" />')
    svg.append(f'<text x="{leg_x + 38}" y="{leg_y + 24}" font-size="12" fill="#1e293b" font-weight="600">Mean Accuracy (%)</text>')
    svg.append(f'<rect x="{leg_x + 130}" y="{leg_y + 14}" width="16" height="12" fill="#059669" rx="2" />')
    svg.append(f'<text x="{leg_x + 153}" y="{leg_y + 24}" font-size="12" fill="#1e293b" font-weight="600">Macro-F1 (%)</text>')
    
    # Draw bars
    for i, (name, acc, f1, s_acc, s_f1) in enumerate(experiments):
        gx = margin["left"] + i * group_width
        cx = gx + group_width / 2
        
        # Accuracy bar
        b1_x = cx - bar_width - 2
        b1_h = (acc / 70.0) * plot_h
        b1_y = margin["top"] + plot_h - b1_h
        svg.append(f'<rect x="{b1_x}" y="{b1_y}" width="{bar_width}" height="{b1_h}" fill="#3b82f6" rx="3" />')
        svg.append(f'<text x="{b1_x + bar_width/2}" y="{b1_y - 8}" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e40af">{acc:.1f}%</text>')
        
        # Macro-F1 bar
        b2_x = cx + 2
        b2_h = (f1 / 70.0) * plot_h
        b2_y = margin["top"] + plot_h - b2_h
        svg.append(f'<rect x="{b2_x}" y="{b2_y}" width="{bar_width}" height="{b2_h}" fill="#10b981" rx="3" />')
        svg.append(f'<text x="{b2_x + bar_width/2}" y="{b2_y - 8}" text-anchor="middle" font-size="11" font-weight="bold" fill="#065f46">{f1:.1f}%</text>')
        
        # X label
        svg.append(f'<text x="{cx}" y="{margin["top"] + plot_h + 24}" text-anchor="middle" font-size="12" font-weight="600" fill="#334155">{name}</text>')
        
    svg.append('</svg>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"Generated {out_path}")


def create_confusion_resolution_chart(stats_path: Path, out_path: Path):
    """
    Generates fig4_confusion_comparison.svg:
    Visualizes the direct reduction in confusable-pair error counts.
    """
    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    dr = data["confusable_pair_resolution"]["drumming_as_clapping"]
    ms = data["confusable_pair_resolution"]["maracas_shaking_as_forward"]
    
    width = 750
    height = 420
    margin = {"top": 60, "right": 50, "bottom": 70, "left": 80}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">',
        f'<text x="{width/2}" y="36" text-anchor="middle" font-size="18" font-weight="bold" fill="#1e293b">Confusable-Pair Error Reduction: RA-STGT v1 vs PR-STGT</text>',
    ]
    
    # Pairs to show
    pairs = [
        ("Drumming -> Clapping", dr["e1_count"], dr["e5_count"], dr["relative_reduction_pct"]),
        ("Maracas Shaking -> Forward", ms["e1_count"], ms["e5_count"], ms["relative_reduction_pct"])
    ]
    
    max_val = 160
    # Grid lines
    svg.append('<g stroke="#e2e8f0" stroke-width="1">')
    for tick in range(0, max_val + 1, 20):
        y_pos = margin["top"] + plot_h - (tick / float(max_val)) * plot_h
        svg.append(f'<line x1="{margin["left"]}" y1="{y_pos}" x2="{margin["left"] + plot_w}" y2="{y_pos}" stroke-dasharray="4,4" />')
        svg.append(f'<text x="{margin["left"] - 12}" y="{y_pos + 4}" text-anchor="end" font-size="12" fill="#64748b">{tick}</text>')
    svg.append('</g>')
    
    svg.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" stroke="#334155" stroke-width="2" />')
    svg.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="#334155" stroke-width="2" />')
    
    group_w = plot_w / len(pairs)
    bar_w = 60
    
    for i, (name, c_e1, c_e5, red_pct) in enumerate(pairs):
        gx = margin["left"] + i * group_w
        cx = gx + group_w / 2
        
        # E1 bar
        b1_x = cx - bar_w - 8
        b1_h = (c_e1 / float(max_val)) * plot_h
        b1_y = margin["top"] + plot_h - b1_h
        svg.append(f'<rect x="{b1_x}" y="{b1_y}" width="{bar_w}" height="{b1_h}" fill="#ef4444" rx="4" />')
        svg.append(f'<text x="{b1_x + bar_w/2}" y="{b1_y - 8}" text-anchor="middle" font-size="12" font-weight="bold" fill="#b91c1c">{c_e1} errors</text>')
        
        # E5 bar
        b2_x = cx + 8
        b2_h = (c_e5 / float(max_val)) * plot_h
        b2_y = margin["top"] + plot_h - b2_h
        svg.append(f'<rect x="{b2_x}" y="{b2_y}" width="{bar_w}" height="{b2_h}" fill="#10b981" rx="4" />')
        svg.append(f'<text x="{b2_x + bar_w/2}" y="{b2_y - 8}" text-anchor="middle" font-size="12" font-weight="bold" fill="#047857">{c_e5} errors</text>')
        
        # Label and reduction pill
        svg.append(f'<text x="{cx}" y="{margin["top"] + plot_h + 24}" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">{name}</text>')
        svg.append(f'<rect x="{cx - 50}" y="{margin["top"] + plot_h + 34}" width="100" height="22" fill="#dcfce7" rx="11" />')
        svg.append(f'<text x="{cx}" y="{margin["top"] + plot_h + 49}" text-anchor="middle" font-size="11" font-weight="bold" fill="#15803d">-{red_pct:.1f}% Reduction</text>')
        
    svg.append('</svg>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"Generated {out_path}")
