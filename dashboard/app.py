import os
import json
import http.server
import socketserver

PORT = 8501
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autism Assessment & Behaviour Modeling Dashboard</title>
    <style>
        :root {
            --primary: #1e3a8a;
            --accent: #2563eb;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --border: #e2e8f0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
        }
        .header {
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 2px solid var(--border);
        }
        .header h1 {
            margin: 0 0 8px 0;
            color: var(--primary);
            font-size: 26px;
        }
        .header p {
            margin: 0;
            color: #64748b;
            font-size: 15px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            margin-bottom: 28px;
        }
        .card {
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            border: 1px solid var(--border);
        }
        .card h2 {
            margin-top: 0;
            font-size: 18px;
            color: var(--primary);
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px dashed #f1f5f9;
        }
        .metric-title {
            color: #475569;
            font-size: 14px;
        }
        .metric-val {
            font-weight: 600;
            font-size: 14px;
            color: #0f172a;
        }
        .badge {
            background: #dbeafe;
            color: #1d4ed8;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .fig-container {
            margin-top: 15px;
            text-align: center;
        }
        .fig-container img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid var(--border);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 10px;
        }
        th, td {
            text-align: left;
            padding: 8px 10px;
            border-bottom: 1px solid var(--border);
        }
        th {
            background-color: #f1f5f9;
            color: #334155;
            font-weight: 600;
        }
        .alert {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 24px;
            font-size: 13px;
            color: #92400e;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Explainable Graph-Based Temporal Modeling of Social Behaviours</h1>
        <p>B.Tech AI Capstone Research & Clinical Decision-Support Benchmark Platform</p>
    </div>

    <div class="alert">
        <strong>Research Decision Support Notice:</strong> This platform is designed for objective behavioral quantification and clinical research support. It is not an autonomous diagnostic instrument. Evaluation is strictly subject-independent (GroupKFold by subject_id).
    </div>

    <div class="grid">
        <div class="card">
            <h2>AV-ASD Multimodal Benchmark (4 Classes)</h2>
            <div class="metric"><span class="metric-title">Evaluated Clips / Subjects</span><span class="metric-val">182 clips / 116 subjects</span></div>
            <div class="metric"><span class="metric-title">Validation Protocol</span><span class="badge">5-Fold GroupKFold (Zero Leakage)</span></div>
            <div class="metric"><span class="metric-title">Pose RF Baseline</span><span class="metric-val">Acc: 47.59% | Macro-F1: 44.03%</span></div>
            <div class="metric"><span class="metric-title">Audio-Only Gradient Boost</span><span class="metric-val">Acc: 47.25% | Macro-F1: 42.90%</span></div>
            <div class="metric"><span class="metric-title">Audio-Only Random Forest</span><span class="metric-val">Acc: 43.41% | Macro-F1: 37.33%</span></div>
            <div class="metric"><span class="metric-title">Top Audio Class Recall</span><span class="metric-val">Aggressive Behavior (73% Recall)</span></div>
        </div>

        <div class="card">
            <h2>MMASD Clinical Benchmark (11 Classes)</h2>
            <div class="metric"><span class="metric-title">Sessions Analyzed</span><span class="metric-val">1,693 sessions</span></div>
            <div class="metric"><span class="metric-title">Participants</span><span class="metric-val">32 children (GroupKFold)</span></div>
            <div class="metric"><span class="metric-title">Tri-Model Ensemble Accuracy</span><span class="metric-val">82.04% (vs 9.09% random)</span></div>
            <div class="metric"><span class="metric-title">Tri-Model Macro-F1 Score</span><span class="metric-val">78.12% (+35.3% over baseline)</span></div>
            <div class="metric"><span class="metric-title">Peak Fold Accuracy</span><span class="metric-val">87.18% (Fold 3)</span></div>
            <div class="metric"><span class="metric-title">Top Identified Activity</span><span class="metric-val">Twist Pose (F1: 0.9360)</span></div>
        </div>

        <div class="card">
            <h2>MEAT-GNN Architecture Overview</h2>
            <div class="metric"><span class="metric-title">Spatial Graph</span><span class="metric-val">8 Body Groups (Head, Arms, Legs, Hands, Torso)</span></div>
            <div class="metric"><span class="metric-title">Temporal Message Passing</span><span class="metric-val">Edge-Aware GAT (w_ij = 1/(1+dt))</span></div>
            <div class="metric"><span class="metric-title">Acoustic Pipeline</span><span class="metric-val">32-dim Bi-GRU + MLP Projection</span></div>
            <div class="metric"><span class="metric-title">Multimodal Gating</span><span class="metric-val">g = &sigma;(W_g [z_vis || z_aud] + b_g)</span></div>
            <div class="metric"><span class="metric-title">Pure PyTorch Core</span><span class="badge">No PyG C++ Wheel Dependencies</span></div>
        </div>
    </div>

    <div class="card">
        <h2>MMASD 11-Class Activity Classification Results (Strict 5-Fold GroupKFold)</h2>
        <table>
            <thead>
                <tr>
                    <th>Activity Class</th>
                    <th>Code</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Twist Pose</td><td><code>twist_pose_tw</code></td><td><strong>0.9360</strong></td><td>120</td><td><span class="badge" style="background:#dcfce7;color:#15803d;">&ge; 0.90 Target</span></td></tr>
                <tr><td>Frog Pose</td><td><code>frog_pose_fg</code></td><td><strong>0.8981</strong></td><td>113</td><td><span class="badge" style="background:#dcfce7;color:#15803d;">~0.90 Target</span></td></tr>
                <tr><td>Drumming</td><td><code>drumming_dr</code></td><td><strong>0.8964</strong></td><td>545</td><td><span class="badge" style="background:#dcfce7;color:#15803d;">~0.90 Target</span></td></tr>
                <tr><td>Tree Pose</td><td><code>tree_pose_tr</code></td><td><strong>0.8471</strong></td><td>129</td><td>Strong</td></tr>
                <tr><td>Arm Swing</td><td><code>arm_swing_as</code></td><td><strong>0.8187</strong></td><td>105</td><td>Strong</td></tr>
                <tr><td>Chest Expansion</td><td><code>chest_expansion_ce</code></td><td><strong>0.8083</strong></td><td>114</td><td>Strong</td></tr>
                <tr><td>Body Swing</td><td><code>body_swing_bs</code></td><td><strong>0.7623</strong></td><td>119</td><td>Solid</td></tr>
                <tr><td>Sing and Clap</td><td><code>sing_and_clap_sac</code></td><td><strong>0.7423</strong></td><td>113</td><td>Solid</td></tr>
                <tr><td>Maracas Shaking</td><td><code>maracas_shaking_ms</code></td><td><strong>0.7328</strong></td><td>130</td><td>Solid</td></tr>
                <tr><td>Maracas Forward Shaking</td><td><code>maracas_forward_shaking_mfs</code></td><td>0.5978</td><td>103</td><td>Acoustic/Kinematic Overlap</td></tr>
                <tr><td>Squat</td><td><code>squat_sq</code></td><td>0.5529</td><td>102</td><td>Lower Limb Occlusion</td></tr>
            </tbody>
        </table>
    </div>

    <div style="margin-top: 24px;" class="card">
        <h2>Publication Explainability & Attribution Maps</h2>
        <div class="grid">
            <div class="fig-container">
                <h3>Top 15 Kinematic Joint Attributions</h3>
                <img src="/static/mmasd_figures/Fig1_top15_joint_importance.png" alt="Joint Attribution">
            </div>
            <div class="fig-container">
                <h3>Temporal Window Attribution</h3>
                <img src="/static/mmasd_figures/Fig2_temporal_importance.png" alt="Temporal Attribution">
            </div>
        </div>
    </div>
</body>
</html>
"""

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        elif self.path.startswith("/static/"):
            rel_path = self.path[len("/static/"):]
            file_path = os.path.join(STATIC_DIR, rel_path)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.send_response(200)
                if file_path.endswith(".png"):
                    self.send_header("Content-type", "image/png")
                elif file_path.endswith(".pdf"):
                    self.send_header("Content-type", "application/pdf")
                elif file_path.endswith(".svg"):
                    self.send_header("Content-type", "image/svg+xml")
                else:
                    self.send_header("Content-type", "application/octet-stream")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "File not found")
        else:
            self.send_error(404, "Endpoint not found")

def run(port=PORT):
    server_address = ("", port)
    httpd = socketserver.TCPServer(server_address, DashboardHandler)
    print(f"Research Dashboard running on http://127.0.0.1:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
