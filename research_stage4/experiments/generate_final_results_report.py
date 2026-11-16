"""Generate the canonical Stage 4 Markdown and standalone HTML results reports."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S4 = ROOT / "research_stage4"

NAMES = {
    "a0_fixed_topo_repro": ("A0", "Fixed bilateral topology (E2 reproduction)"),
    "a1_dynamic_no_reliability": ("A1", "Dynamic bilateral; no multiplicative gate"),
    "a2_reliability_no_dynamic": ("A2", "Fixed topology with reliability only"),
    "a3_rcdb_full_main": ("A3", "Full RCDB-STGT (dynamic + reliability gate + smoothing)"),
    "a4_rcdb_no_temp_smooth": ("A4", "RCDB-STGT without temporal smoothing"),
    "a5_rcdb_with_phase": ("A5", "RCDB-STGT with phase stream"),
    "a6_rcdb_wrist_only": ("A6", "RCDB-STGT with wrists only"),
    "a7_rcdb_multi_pair": ("A7", "RCDB-STGT with five bilateral pairs"),
    "a8_rcdb_input_rel_only": ("A8", "Configured identically to A1; not an independent ablation"),
    "a9_rcdb_with_margin": ("A9", "RCDB-STGT with confusable-class margin loss"),
}

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def pct(x):
    return f"{x * 100:.2f}%"

def generate():
    master = read_json(S4 / "metrics" / "stage4_ablation_master_summary.json")
    stats_path = S4 / "statistics" / "stage4_statistical_significance_results.json"
    rob_path = S4 / "robustness" / "stage4_robustness_benchmark_results.json"
    stats = read_json(stats_path) if stats_path.exists() else {}
    rob = read_json(rob_path) if rob_path.exists() else {}

    rows = []
    for key, (code, desc) in NAMES.items():
        d = master.get(key)
        if d:
            rows.append((code, desc, pct(d["mean_fold_acc"]), f"±{pct(d['std_fold_acc'])}",
                         f"{d['pooled_oof_macro_f1']:.4f}", f"{d.get('mean_ece', float('nan')):.4f}"))
        else:
            rows.append((code, desc, "Pending", "—", "—", "—"))

    md = [
        "# Stage 4 Final Results: RCDB-STGT",
        "",
        "**Status:** A0–A9 training and automated post-processing completed. **A8 is not an independent experiment**: inspection of the implementation shows A1 and A8 both instantiate the model with `use_reliability_gate=False`, and reliability is supplied to the edge learner in both configurations. Their matching scores should not be interpreted as an independent replication.",
        "",
        "## Ablation results",
        "",
        "| ID | Configuration | Mean fold accuracy | Fold SD | Pooled OOF macro-F1 | Mean ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append("| " + " | ".join(row) + " |")
    md += ["", "A0–A9 use the same five cohort-grouped folds and seed 42. Accuracy and standard deviation are computed across folds; pooled macro-F1 uses the combined out-of-fold predictions. ECE is the mean fold expected calibration error.", "",
           "## Selected statistical comparisons", "",
           "These are the comparisons present in the saved statistical output; they are not an exhaustive pairwise test of every ablation. Fold-level tests use only five observations and should be interpreted cautiously. Clip-level tests use the pooled held-out predictions.", "",
           "| Comparison | Mean accuracy difference | Fold t-test p | McNemar p | Bootstrap 95% CI for accuracy difference |",
           "|---|---:|---:|---:|---:|"]
    for k, d in stats.items():
        ref, test = d["reference"], d["test"]
        label = f"{NAMES[ref][0]} → {NAMES[test][0]}"
        ci = d["clip_level"]["bootstrap_95ci"]
        lo, hi = ci["ci_95_lower"] * 100, ci["ci_95_upper"] * 100
        md.append(f"| {label} | {d['delta_mean_acc']*100:+.2f} pp | {d['fold_level']['p_val_t']:.4f} | {d['clip_level']['mcnemar']['p_value']:.3g} | [{lo:+.2f}, {hi:+.2f}] pp |")

    md += ["", "## Sensor corruption benchmark", "", "The robustness benchmark covers A0, A1, and A3. These are zero-shot perturbations applied at evaluation time.", "",
           "| Model | Condition | Accuracy | Macro-F1 |", "|---|---|---:|---:|"]
    conditions = [("clean", "Clean"), ("random_20", "Random 20% joint dropout"), ("random_40", "Random 40% joint dropout"),
                  ("wrist_occlusion", "Both wrists occluded"), ("ankle_occlusion", "Both ankles occluded"), ("consecutive_15", "15-frame blackout")]
    for model in ("a0_fixed_topo_repro", "a1_dynamic_no_reliability", "a3_rcdb_full_main"):
        for cond, label in conditions:
            d = rob.get(model, {}).get(cond)
            if d:
                md.append(f"| {NAMES[model][0]} | {label} | {pct(d['mean_acc'])} | {d['mean_macro_f1']:.4f} |")
    md += ["", "## Figures", "", "Publication figures are in `research_stage4/visualizations/`: `fig1_ablation_matrix_benchmark.png`, `fig2_robustness_stress_comparison.png`, `fig3_dynamic_bilateral_heatmap_drumming.png`, `fig3_dynamic_bilateral_heatmap_tree_pose.png`, and `fig4_confusion_matrices_comparison.png`.", "",
           "## Interpretation notes", "", "- The highest mean accuracy among these runs is A1/A8 at 65.92%; A8 duplicates A1 in the current implementation, so this is one configuration, not two independent confirmations.",
           "- A3 is the full reliability-gated RCDB-STGT model and has 60.40% mean fold accuracy and 0.5947 pooled OOF macro-F1.",
           "- A5–A9 results are specific to this dataset and grouped-fold protocol. They do not establish clinical effectiveness or participant-independent generalization.",
           "- Check implementation and split assumptions before making claims from these results. In particular, cohort IDs are not verified participant IDs.", ""]
    md_path = S4 / "STAGE4_FINAL_RESULTS.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    ablation_path = S4 / "ABLATION_RESULTS.md"
    ablation = ablation_path.read_text(encoding="utf-8")
    ablation = ablation.replace(
        "**Status:** A0 Confirmed ✅ | A1–A9 In Progress / To be populated after experiment completion",
        "**Status:** A0–A9 complete | Consolidated final table below; A8 duplicates A1 in the current implementation"
    )
    section_start = "<!-- FINAL_STAGE4_TABLE_START -->"
    section_end = "<!-- FINAL_STAGE4_TABLE_END -->"
    current_table = [section_start, "## Final Consolidated Results (A0–A9)", "",
                     "All ten run IDs have saved fold summaries. A8 is configured identically to A1 in the current model code and is therefore not an independent ablation.", "",
                     "| ID | Configuration | Mean fold accuracy | Fold SD | Pooled OOF macro-F1 | Mean ECE |", "|---|---|---:|---:|---:|---:|"]
    current_table.extend("| " + " | ".join(r) + " |" for r in rows)
    current_table += ["", "See [STAGE4_FINAL_RESULTS.md](STAGE4_FINAL_RESULTS.md) for selected statistical comparisons, robustness results, and limitations. The standalone report is `reports/STAGE4_FINAL_RESULTS.html`.", section_end]
    generated = "\n".join(current_table)
    if section_start in ablation and section_end in ablation:
        start = ablation.index(section_start)
        end = ablation.index(section_end, start) + len(section_end)
        ablation = ablation[:start] + generated + ablation[end:]
    else:
        first_hr = ablation.find("---")
        ablation = ablation[:first_hr + 3] + "\n\n" + generated + "\n" + ablation[first_hr + 3:]
    ablation = ablation.replace(
        "### 3.6 Remaining Ablations (A5–A9)",
        "### 3.6 Historical Pending-Run Placeholder (Superseded)"
    )
    ablation = ablation.replace(
        "> [!IMPORTANT]\n> A5 is currently running. A6–A9 will follow sequentially.",
        "> [!NOTE]\n> This historical placeholder is superseded by the final consolidated results table at the top of this document."
    )
    ablation_path.write_text(ablation, encoding="utf-8")

    css = "body{font:16px/1.55 system-ui,Segoe UI,sans-serif;max-width:1200px;margin:36px auto;padding:0 22px;color:#172033}h1,h2{color:#14345b}table{border-collapse:collapse;width:100%;margin:16px 0 28px;font-size:14px}th,td{border:1px solid #ccd4df;padding:9px;text-align:left}th{background:#edf3fa}figure{margin:26px 0}img{max-width:100%;height:auto;border:1px solid #d5dbe4}figcaption{color:#53657a}.notice{padding:14px;background:#fff7df;border-left:4px solid #c28b1b}.muted{color:#526173}"
    body = ["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Stage 4 Final Results - RCDB-STGT</title>", f"<style>{css}</style><body>",
            "<h1>Stage 4 Final Results: RCDB-STGT</h1>",
            "<p>Final report of the completed A0–A9 experiments on the local MMASD OpenPose activity dataset.</p>",
            "<p class='notice'><strong>Configuration note:</strong> A8 is not an independent ablation in the current code. A1 and A8 both set <code>use_reliability_gate=False</code>, and both edge learners receive joint reliability inputs. Their identical scores reflect the same configuration.</p>",
            "<h2>Ablation results</h2><table><thead><tr><th>ID</th><th>Configuration</th><th>Mean fold accuracy</th><th>Fold SD</th><th>Pooled OOF macro-F1</th><th>Mean ECE</th></tr></thead><tbody>"]
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>")
    body += ["</tbody></table><p class='muted'>All runs use five cohort-grouped folds and seed 42. Cohort identifiers are not verified participant identifiers.</p>",
             "<h2>Selected statistical comparisons</h2><p>These are the comparisons included in the saved test output, not all possible pairs. Fold-level tests have only five observations and should be interpreted cautiously.</p>",
             "<table><thead><tr><th>Comparison</th><th>Mean accuracy difference</th><th>Fold t-test p</th><th>McNemar p</th><th>Bootstrap 95% CI (pp)</th></tr></thead><tbody>"]
    for k, d in stats.items():
        ref, test = d["reference"], d["test"]
        ci = d["clip_level"]["bootstrap_95ci"]
        body.append(f"<tr><td>{NAMES[ref][0]} &rarr; {NAMES[test][0]}</td><td>{d['delta_mean_acc']*100:+.2f} pp</td><td>{d['fold_level']['p_val_t']:.4f}</td><td>{d['clip_level']['mcnemar']['p_value']:.3g}</td><td>[{ci['ci_95_lower']*100:+.2f}, {ci['ci_95_upper']*100:+.2f}]</td></tr>")
    body += ["</tbody></table><h2>Sensor corruption benchmark</h2><p>Zero-shot perturbations were evaluated for A0, A1, and A3.</p><table><thead><tr><th>Model</th><th>Condition</th><th>Accuracy</th><th>Macro-F1</th></tr></thead><tbody>"]
    for model in ("a0_fixed_topo_repro", "a1_dynamic_no_reliability", "a3_rcdb_full_main"):
        for cond, label in conditions:
            d = rob.get(model, {}).get(cond)
            if d:
                body.append(f"<tr><td>{NAMES[model][0]}</td><td>{label}</td><td>{pct(d['mean_acc'])}</td><td>{d['mean_macro_f1']:.4f}</td></tr>")
    body += ["</tbody></table><h2>Figures</h2>"]
    figures = [("fig1_ablation_matrix_benchmark.png", "Ablation benchmark"), ("fig2_robustness_stress_comparison.png", "Sensor corruption benchmark"),
               ("fig3_dynamic_bilateral_heatmap_drumming.png", "Dynamic bilateral weights during drumming"),
               ("fig3_dynamic_bilateral_heatmap_tree_pose.png", "Dynamic bilateral weights during tree pose"),
               ("fig4_confusion_matrices_comparison.png", "Confusion matrices")]
    for filename, caption in figures:
        p = S4 / "visualizations" / filename
        if p.exists():
            data = base64.b64encode(p.read_bytes()).decode("ascii")
            body.append(f"<figure><img alt='{html.escape(caption)}' src='data:image/png;base64,{data}'><figcaption>{html.escape(caption)}</figcaption></figure>")
    body += ["<h2>Scope and limitations</h2><p>These results describe activity classification on this processed dataset and cohort-grouped protocol. They do not establish clinical effectiveness or generalization to verified unseen participants. A8 duplicates A1 and should not be counted as a separate confirmation.</p>",
             "<p>Machine-readable results, out-of-fold predictions, robustness data, statistical outputs, and publication figures are retained in the Stage 4 project folders.</p></body></html>"]
    report_dir = S4 / "reports"
    report_dir.mkdir(exist_ok=True)
    html_path = report_dir / "STAGE4_FINAL_RESULTS.html"
    html_path.write_text("\n".join(body), encoding="utf-8")

    # Add a prominent pointer to the canonical results in the existing detailed report.
    report = S4 / "STAGE4_FINAL_RESEARCH_REPORT.md"
    if report.exists():
        text = report.read_text(encoding="utf-8")
        marker = "<!-- CANONICAL_FINAL_RESULTS -->"
        notice = marker + "\n\n> **Updated results:** All A0–A9 runs are complete. See [the consolidated Stage 4 results](STAGE4_FINAL_RESULTS.md) and [the standalone HTML report](reports/STAGE4_FINAL_RESULTS.html). A8 duplicates A1 in the current implementation and must not be treated as an independent ablation.\n\n<!-- /CANONICAL_FINAL_RESULTS -->"
        if marker in text:
            begin = text.index(marker)
            close = text.index("<!-- /CANONICAL_FINAL_RESULTS -->", begin) + len("<!-- /CANONICAL_FINAL_RESULTS -->")
            text = text[:begin] + notice + text[close:]
        else:
            insert_at = text.find("---", text.find("---") + 3)
            text = text[:insert_at + 3] + "\n\n" + notice + text[insert_at + 3:]
        # Replace the old partial A0-A9 table with the machine-derived final table.
        h_start, h_end = "### 3.2 Stage 4 Ablation Results", "### 3.3 A0 Detailed Per-Class Analysis"
        if h_start in text and h_end in text:
            i, j = text.index(h_start), text.index(h_end)
            report_table = [h_start, "", "| Experiment | Model Description | Mean Fold Accuracy | Fold SD | Pooled OOF Macro-F1 | Mean ECE | Status |",
                            "|---|---|---:|---:|---:|---:|---|"]
            for row, key in zip(rows, NAMES):
                code, desc, acc, sd, f1, ece = row
                report_table.append(f"| {code} | {desc} | {acc} | {sd} | {f1} | {ece} | Complete |")
            report_table += ["", "A8 is configured identically to A1 in the current implementation and is not an independent ablation. See `STAGE4_FINAL_RESULTS.md` for statistical comparisons, robustness results, and interpretation notes.", ""]
            text = text[:i] + "\n".join(report_table) + text[j:]
        text = text.replace("A3 results pending.", "A3 completed at 60.40% mean fold accuracy and 0.5947 pooled OOF macro-F1.")
        text = text.replace("🔄 A0 filled, A1–A9 pending", "✅ A0–A9 complete; canonical table updated")
        text = text.replace("⏳ Pending experiment completion", "✅ Results saved; see canonical Stage 4 results")
        report.write_text(text, encoding="utf-8")
    print(md_path)
    print(html_path)

if __name__ == "__main__":
    generate()
