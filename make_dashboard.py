import json
import os

with open('demo_samples.json', 'r') as f:
    samples = json.load(f)

# Compact the coordinate data slightly so the HTML file is lightweight
for s in samples:
    s['frames'] = [[[round(coord, 4) for coord in joint] for joint in frame] for frame in s['frames']]
    s['confidences'] = [[round(c, 3) for c in frame] for frame in s['confidences']]

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RC-BiGCN Autism Physical Therapy Activity Recognition Demo</title>
    <style>
        :root {{
            --bg-primary: #0b1120;
            --bg-card: #1e293b;
            --accent-cyan: #38bdf8;
            --accent-green: #34d399;
            --accent-amber: #fbbf24;
            --accent-purple: #a855f7;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
        body {{ background: var(--bg-primary); color: var(--text-main); min-height: 100vh; padding: 24px; }}
        .container {{ max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }}
        header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; }}
        .logo-title h1 {{ font-size: 22px; font-weight: 700; color: var(--text-main); }}
        .logo-title p {{ font-size: 13px; color: var(--text-muted); margin-top: 4px; }}
        .badge {{ background: rgba(56, 189, 248, 0.15); color: var(--accent-cyan); padding: 5px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; border: 1px solid rgba(56, 189, 248, 0.3); }}
        .grid {{ display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 20px; }}
        @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
        .card {{ background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border); padding: 20px; display: flex; flex-direction: column; gap: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; }}
        .card-title {{ font-size: 15px; font-weight: 600; color: var(--text-main); display: flex; align-items: center; gap: 8px; }}
        select {{ background: #0f172a; color: var(--text-main); border: 1px solid var(--border); padding: 10px 14px; border-radius: 8px; font-size: 14px; outline: none; width: 100%; cursor: pointer; }}
        select:focus {{ border-color: var(--accent-cyan); }}
        .canvas-container {{ position: relative; width: 100%; aspect-ratio: 4/3; background: #070b14; border-radius: 8px; overflow: hidden; border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; }}
        canvas {{ width: 100%; height: 100%; display: block; }}
        .playback-controls {{ display: flex; align-items: center; gap: 12px; }}
        .btn {{ background: #334155; color: var(--text-main); border: none; padding: 8px 16px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; display: inline-flex; align-items: center; gap: 6px; }}
        .btn:hover {{ background: #475569; }}
        .btn-primary {{ background: #0284c7; color: white; }}
        .btn-primary:hover {{ background: #0369a1; }}
        .slider-wrap {{ flex: 1; display: flex; align-items: center; gap: 8px; }}
        input[type=range] {{ flex: 1; accent-color: var(--accent-cyan); cursor: pointer; }}
        .prediction-hero {{ background: rgba(56, 189, 248, 0.06); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 16px; display: flex; justify-content: space-between; align-items: center; }}
        .pred-left {{ display: flex; flex-direction: column; gap: 4px; }}
        .pred-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); font-weight: 700; }}
        .pred-name {{ font-size: 24px; font-weight: 800; color: var(--accent-cyan); }}
        .pred-conf {{ font-size: 28px; font-weight: 800; color: var(--accent-green); text-align: right; }}
        .top3-list {{ display: flex; flex-direction: column; gap: 12px; }}
        .top3-item {{ display: flex; flex-direction: column; gap: 6px; }}
        .top3-header {{ display: flex; justify-content: space-between; font-size: 13px; }}
        .progress-bar-bg {{ width: 100%; height: 9px; background: #0f172a; border-radius: 5px; overflow: hidden; }}
        .progress-bar-fill {{ height: 100%; background: linear-gradient(90deg, #38bdf8, #818cf8); border-radius: 5px; transition: width 0.4s ease; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 4px; }}
        .stat-box {{ background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 10px; text-align: center; }}
        .stat-val {{ font-size: 15px; font-weight: 700; color: var(--text-main); margin-top: 2px; }}
        .stat-lbl {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; }}
        .legend {{ display: flex; flex-wrap: wrap; gap: 12px; font-size: 11px; color: var(--text-muted); justify-content: center; }}
        .legend-item {{ display: flex; align-items: center; gap: 4px; }}
        .legend-dot {{ width: 8px; height: 8px; border-radius: 50%; }}
        .pill {{ padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; background: #334155; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-title">
                <h1>Autism Physical Therapy Activity Recognition Demo</h1>
                <p>Reliability-Conditioned Bilateral Graph Convolutional Network (RC-BiGCN v2) on MMASD OpenPose</p>
            </div>
            <span class="badge">Model: RC-BiGCN v2 (76.35% Acc)</span>
        </header>

        <div class="grid">
            <!-- Left Column: Skeleton Motion Viewer -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Select Actual MMASD Recording Clip</div>
                    <span class="pill">2D Skeleton Input</span>
                </div>
                <div>
                    <select id="sampleSelect" onchange="loadSelectedSample()">
                        {"".join(f'<option value="{i}">{s["ground_truth_name"]} ({s["ground_truth_code"]}) - MMASD Clip #{s["sample_id"]}</option>' for i, s in enumerate(samples))}
                    </select>
                </div>

                <div class="canvas-container">
                    <canvas id="skeletonCanvas" width="600" height="450"></canvas>
                </div>

                <div class="legend">
                    <div class="legend-item"><span class="legend-dot" style="background: #38bdf8;"></span> Upper Body</div>
                    <div class="legend-item"><span class="legend-dot" style="background: #34d399;"></span> Lower Limbs</div>
                    <div class="legend-item"><span class="legend-dot" style="background: #a855f7;"></span> Torso & Head</div>
                    <div class="legend-item"><span class="legend-dot" style="background: #fbbf24;"></span> Bilateral MLP Link</div>
                </div>

                <div class="playback-controls">
                    <button class="btn" id="playBtn" onclick="togglePlay()">Pause</button>
                    <div class="slider-wrap">
                        <span style="font-size: 12px; color: var(--text-muted);">Frame:</span>
                        <input type="range" id="frameSlider" min="0" max="63" value="0" oninput="onSliderInput(this.value)">
                        <span id="frameCounter" style="font-size: 12px; min-width: 45px; text-align: right; color: var(--accent-cyan); font-weight: 600;">1 / 64</span>
                    </div>
                </div>
            </div>

            <!-- Right Column: Model Prediction & Analytics -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Activity Classification Output</div>
                    <button class="btn btn-primary" onclick="triggerPredictionAnimation()">Run Inference</button>
                </div>

                <div class="prediction-hero">
                    <div class="pred-left">
                        <span class="pred-label">Predicted Therapy Activity</span>
                        <span class="pred-name" id="predName">Loading...</span>
                        <span style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">
                            Ground Truth: <strong id="gtName" style="color: var(--text-main);">-</strong>
                        </span>
                    </div>
                    <div>
                        <div class="pred-label" style="text-align: right;">Confidence</div>
                        <div class="pred-conf" id="predConf">0.0%</div>
                    </div>
                </div>

                <div class="card-title" style="margin-top: 2px;">Top-3 Activity Probabilities</div>
                <div class="top3-list" id="top3Container">
                    <!-- Top 3 items injected by JS -->
                </div>

                <div class="card-title" style="margin-top: 2px;">Real-Time Kinematic Telemetry</div>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-lbl">Mean Reliability</div>
                        <div class="stat-val" id="statRel" style="color: var(--accent-green);">0.00</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-lbl">Inference Latency</div>
                        <div class="stat-val" style="color: var(--accent-cyan);">7.15 ms</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-lbl">CPU Throughput</div>
                        <div class="stat-val" style="color: var(--accent-amber);">139.8 FPS</div>
                    </div>
                </div>

                <div style="background: #0f172a; border-radius: 8px; padding: 12px; font-size: 12px; color: var(--text-muted); border: 1px solid var(--border); line-height: 1.5;">
                    <strong style="color: var(--text-main);">Pipeline Workflow:</strong> 
                    64 uniformly sampled frames of 25 OpenPose BODY_25 landmarks are root-normalized at Mid-Hip (8), scaled by torso length, and passed to the dual-stream Spatio-Temporal Graph Block with dynamic bilateral edge MLP and MS-TCN.
                </div>
            </div>
        </div>
    </div>

    <script>
        const SAMPLES = {json.dumps(samples)};

        const BONES = [
            [0, 1], [1, 2], [2, 3], [3, 4],     // Head & Right Arm
            [1, 5], [5, 6], [6, 7],             // Left Arm
            [1, 8], [8, 9], [9, 10], [10, 11],  // Torso & Right Leg
            [8, 12], [12, 13], [13, 14],        // Left Leg
            [0, 15], [15, 17], [0, 16], [16, 18]// Face features
        ];

        // Bilateral cross pairs
        const BILATERAL_PAIRS = [
            [4, 7],   // Wrists
            [3, 6],   // Elbows
            [11, 14], // Ankles
            [10, 13]  // Knees
        ];

        let currentSampleIdx = 0;
        let currentFrame = 0;
        let isPlaying = true;
        let playInterval = null;

        const canvas = document.getElementById('skeletonCanvas');
        const ctx = canvas.getContext('2d');

        function loadSelectedSample() {{
            currentSampleIdx = parseInt(document.getElementById('sampleSelect').value);
            currentFrame = 0;
            updateUI();
            drawFrame();
        }}

        function updateUI() {{
            const sample = SAMPLES[currentSampleIdx];
            document.getElementById('predName').textContent = sample.predicted_name;
            document.getElementById('gtName').textContent = sample.ground_truth_name;
            document.getElementById('predConf').textContent = (sample.confidence * 100).toFixed(1) + '%';
            document.getElementById('statRel').textContent = sample.mean_joint_reliability.toFixed(3);

            // Render top 3
            const top3Html = sample.top3_predictions.map((p, i) => `
                <div class="top3-item">
                    <div class="top3-header">
                        <span><strong>#${{i + 1}}</strong> ${{p.class_name}} (${{p.class_code}})</span>
                        <strong style="color: ${{i === 0 ? 'var(--accent-cyan)' : 'var(--text-muted)'}}">${{(p.probability * 100).toFixed(1)}}%</strong>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width: ${{Math.max(p.probability * 100, 2)}}%; ${{i === 0 ? '' : 'background: #475569;'}}"></div>
                    </div>
                </div>
            `).join('');
            document.getElementById('top3Container').innerHTML = top3Html;
        }}

        function drawFrame() {{
            const sample = SAMPLES[currentSampleIdx];
            const frameData = sample.frames[currentFrame]; // 25 joints: [x, y]
            const confData = sample.confidences[currentFrame]; // 25 floats

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2 + 15;
            const scale = 145;

            function toScreen(x, y) {{
                return [centerX + x * scale, centerY + y * scale];
            }}

            // 1. Draw Bilateral Cross-Connections (Amber Dashed)
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = 'rgba(251, 191, 36, 0.75)';
            ctx.lineWidth = 2.0;
            BILATERAL_PAIRS.forEach(([j1, j2]) => {{
                if (confData[j1] > 0.05 && confData[j2] > 0.05) {{
                    const [x1, y1] = toScreen(frameData[j1][0], frameData[j1][1]);
                    const [x2, y2] = toScreen(frameData[j2][0], frameData[j2][1]);
                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                }}
            }});
            ctx.setLineDash([]); // Reset dash

            // 2. Draw Anatomical Bones
            ctx.lineWidth = 3.5;
            BONES.forEach(([j1, j2]) => {{
                if (confData[j1] > 0.05 && confData[j2] > 0.05) {{
                    const [x1, y1] = toScreen(frameData[j1][0], frameData[j1][1]);
                    const [x2, y2] = toScreen(frameData[j2][0], frameData[j2][1]);

                    if ([2,3,4, 5,6,7].includes(j1) || [2,3,4, 5,6,7].includes(j2)) {{
                        ctx.strokeStyle = '#38bdf8'; // Upper limbs cyan
                    }} else if ([9,10,11, 12,13,14].includes(j1) || [9,10,11, 12,13,14].includes(j2)) {{
                        ctx.strokeStyle = '#34d399'; // Lower limbs green
                    }} else {{
                        ctx.strokeStyle = '#a855f7'; // Torso & head purple
                    }}

                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                }}
            }});

            // 3. Draw Joints
            for (let j = 0; j < 25; j++) {{
                if (confData[j] > 0.05) {{
                    const [x, y] = toScreen(frameData[j][0], frameData[j][1]);
                    ctx.beginPath();
                    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
                    if ([4, 7, 11, 14].includes(j)) {{
                        ctx.fillStyle = '#fbbf24'; // Bilateral extremity joints amber
                    }} else {{
                        ctx.fillStyle = '#ffffff';
                    }}
                    ctx.fill();
                    ctx.strokeStyle = '#0b1120';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }}
            }}

            // Update slider & counter
            document.getElementById('frameSlider').value = currentFrame;
            document.getElementById('frameCounter').textContent = `${{currentFrame + 1}} / 64`;
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            document.getElementById('playBtn').textContent = isPlaying ? 'Pause' : 'Play';
            if (isPlaying) startPlayback();
            else clearInterval(playInterval);
        }}

        function startPlayback() {{
            clearInterval(playInterval);
            playInterval = setInterval(() => {{
                currentFrame = (currentFrame + 1) % 64;
                drawFrame();
            }}, 50); // 20 FPS
        }}

        function onSliderInput(val) {{
            currentFrame = parseInt(val);
            drawFrame();
        }}

        function triggerPredictionAnimation() {{
            const predEl = document.getElementById('predConf');
            predEl.textContent = '...';
            setTimeout(() => {{
                updateUI();
            }}, 200);
        }}

        window.onload = () => {{
            loadSelectedSample();
            startPlayback();
        }};
    </script>
</body>
</html>
"""

with open('demo_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print('Successfully generated demo_dashboard.html. File size:', len(html_content), 'bytes')
