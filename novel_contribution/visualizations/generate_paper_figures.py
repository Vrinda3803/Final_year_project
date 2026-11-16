"""
Generates publication-ready figures for RA-STGT:
1. Reliability Diagram (ECE before vs after calibration)
2. Risk-Coverage Selective Prediction Curve
3. Robustness Curves under 0% to 40% Missingness
4. Per-Class F1 Radar/Bar Comparison
"""
from __future__ import annotations
import sys, os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import numpy as np

def generate_svg_reliability_diagram(uncal_ece: float, cal_ece: float, out_svg: str):
    """Generates an SVG visualization of the ECE calibration improvement."""
    svg = f"""<svg width="600" height="350" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="300" y="30" font-family="Arial, sans-serif" font-size="16" font-weight="bold" text-anchor="middle" fill="#333333">
    Predictive Calibration: Uncalibrated vs Calibrated ECE
  </text>
  
  <!-- Subtitle -->
  <text x="300" y="55" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#666666">
    Temperature scaling fitted strictly on held-out validation subjects
  </text>
  
  <!-- Bars -->
  <!-- Bar 1: Uncalibrated -->
  <rect x="140" y="100" width="100" height="{uncal_ece * 800}" fill="#e74c3c" rx="4" />
  <text x="190" y="90" font-family="Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle" fill="#e74c3c">
    ECE: {uncal_ece:.4f}
  </text>
  <text x="190" y="{115 + uncal_ece * 800}" font-family="Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle" fill="#333333">
    Uncalibrated
  </text>
  <text x="190" y="{135 + uncal_ece * 800}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle" fill="#777777">
    (Overconfident)
  </text>

  <!-- Bar 2: Calibrated -->
  <rect x="360" y="{100 + (uncal_ece - cal_ece) * 800}" width="100" height="{cal_ece * 800}" fill="#27ae60" rx="4" />
  <text x="410" y="{90 + (uncal_ece - cal_ece) * 800}" font-family="Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle" fill="#27ae60">
    ECE: {cal_ece:.4f}
  </text>
  <text x="410" y="{115 + uncal_ece * 800}" font-family="Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle" fill="#333333">
    Calibrated (RA-STGT)
  </text>
  <text x="410" y="{135 + uncal_ece * 800}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle" fill="#777777">
    (Error reduced by {(1 - cal_ece/uncal_ece)*100:.1f}%)
  </text>
  
  <!-- Clinical Takeaway note -->
  <rect x="50" y="270" width="500" height="50" fill="#f8f9fa" stroke="#e2e8f0" rx="6" />
  <text x="300" y="292" font-family="Arial, sans-serif" font-size="11" text-anchor="middle" fill="#2d3748">
    Clinical Benefit: Calibrated probabilities reflect true accuracy, enabling dependable
  </text>
  <text x="300" y="308" font-family="Arial, sans-serif" font-size="11" text-anchor="middle" fill="#2d3748">
    abstention when observations are degraded by occlusions or severe tracking jitter.
  </text>
</svg>"""
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Saved calibration figure to: {out_svg}")


def generate_svg_selective_prediction(curve: list, out_svg: str):
    """Generates an SVG visualization of the Selective Prediction Risk-Coverage curve."""
    pts = []
    # Plot coverage (x: 80 to 520) vs Accuracy (y: 240 to 80)
    for pt in curve:
        cov = pt["coverage"]
        acc = pt["selective_accuracy"]
        x = 80 + cov * 440
        y = 250 - (acc - 0.5) * 340  # Map 50%-100% to 250-80
        pts.append((x, y, cov, acc, pt["threshold"]))
        
    path_data = " ".join([f"{'M' if i==0 else 'L'} {p[0]:.1f},{p[1]:.1f}" for i, p in enumerate(pts)])
    
    svg = f"""<svg width="600" height="350" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="300" y="30" font-family="Arial, sans-serif" font-size="16" font-weight="bold" text-anchor="middle" fill="#333333">
    Selective Prediction: Accuracy vs Coverage Trade-off
  </text>
  <text x="300" y="52" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#666666">
    Accuracy increases as the model selectively abstains on low-confidence occluded clips
  </text>

  <!-- Axes -->
  <line x1="80" y1="260" x2="540" y2="260" stroke="#cbd5e0" stroke-width="2" />
  <line x1="80" y1="260" x2="80" y2="70" stroke="#cbd5e0" stroke-width="2" />
  
  <!-- Axis labels -->
  <text x="310" y="295" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#4a5568">Coverage (% of sessions classified)</text>
  <text x="35" y="165" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#4a5568" transform="rotate(-90 35,165)">Selective Accuracy</text>
  
  <!-- Curve -->
  <path d="{path_data}" fill="none" stroke="#2b6cb0" stroke-width="3.5" />
"""
    for p in pts[::2]:
        svg += f"""  <circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="5" fill="#3182ce" />
  <text x="{p[0]:.1f}" y="{p[1]-10:.1f}" font-family="Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle" fill="#2b6cb0">{p[3]*100:.1f}%</text>
"""
    svg += """</svg>"""
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Saved selective prediction figure to: {out_svg}")
