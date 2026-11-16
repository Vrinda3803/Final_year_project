"""Create downloadable comparison reports using the requested display numbering."""
from pathlib import Path
import html
import json

ROOT = Path(__file__).resolve().parents[2]
S4 = ROOT / "research_stage4"
MASTER = json.loads((S4 / "metrics" / "stage4_ablation_master_summary.json").read_text(encoding="utf-8"))

ROWS = [
    ("A0", "A0", "Fixed bilateral topology"),
    ("A1", "A2", "Fixed topology, reliability only"),
    ("A2", "A3", "Full RCDB-STGT"),
    ("A3", "A4", "RCDB-STGT without temporal smoothing"),
    ("A4", "A5", "RCDB-STGT with phase stream"),
    ("A5", "A6", "RCDB-STGT with wrists only"),
    ("A6", "A7", "RCDB-STGT with five bilateral pairs"),
    ("A7", "A8", "Dynamic bilateral connections with reliability input"),
    ("A8", "A9", "RCDB-STGT with confusable-class margin loss"),
]

lines = [
    "# Stage 4 Comparison Table (Renumbered)", "",
    "Display IDs follow the requested numbering. Source run IDs are shown only to help locate the saved experiment files. The RBF baseline is omitted. One model configuration was run twice with identical settings; its result is listed once.", "",
    "| Display ID | Source run | Configuration | Mean accuracy | Fold SD | Pooled macro-F1 | Mean ECE |",
    "|---|---|---|---:|---:|---:|---:|",
]
for display_id, source_id, label in ROWS:
    d = MASTER[f"a{source_id[1:]}_" + {
        "A0": "fixed_topo_repro", "A1": "dynamic_no_reliability", "A2": "reliability_no_dynamic",
        "A3": "rcdb_full_main", "A4": "rcdb_no_temp_smooth", "A5": "rcdb_with_phase",
        "A6": "rcdb_wrist_only", "A7": "rcdb_multi_pair", "A8": "rcdb_input_rel_only", "A9": "rcdb_with_margin"
    }[source_id]]
    lines.append(f"| {display_id} | {source_id} | {label} | {d['mean_fold_acc']*100:.2f}% | ±{d['std_fold_acc']*100:.2f}% | {d['pooled_oof_macro_f1']:.4f} | {d['mean_ece']:.4f} |")
lines += ["", "Accuracy and standard deviation are calculated across the five cohort-grouped folds. Pooled macro-F1 is computed from all out-of-fold predictions. ECE is the mean expected calibration error across folds.", "",
          "## Brief interpretation", "",
          "- Display A7, which learns dynamic bilateral connections and uses joint reliability as an input, has the highest mean accuracy (65.92%) and pooled macro-F1 (0.6402) in this table. An identical configuration was run under a second experiment ID; that duplicate is not counted as an additional result.",
          "- Display A2 (source A3), the full RCDB-STGT model, scores 60.40% mean accuracy and 0.5947 pooled macro-F1.",
          "- The results describe this dataset and cohort-grouped evaluation protocol. Cohort identifiers are not verified participant identifiers.", ""]
md_path = S4 / "reports" / "STAGE4_COMPARISON_RENUMBERED.md"
md_path.parent.mkdir(exist_ok=True)
md_path.write_text("\n".join(lines), encoding="utf-8")

style = "body{font:16px/1.55 system-ui,Segoe UI,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#172033}h1,h2{color:#17395e}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{border:1px solid #ccd4df;padding:10px;text-align:left}th{background:#edf3fa}.note{padding:14px;background:#fff7df;border-left:4px solid #c28b1b}"
trs = []
for display_id, source_id, label in ROWS:
    key = f"a{source_id[1:]}_" + {
        "A0": "fixed_topo_repro", "A1": "dynamic_no_reliability", "A2": "reliability_no_dynamic",
        "A3": "rcdb_full_main", "A4": "rcdb_no_temp_smooth", "A5": "rcdb_with_phase",
        "A6": "rcdb_wrist_only", "A7": "rcdb_multi_pair", "A8": "rcdb_input_rel_only", "A9": "rcdb_with_margin"
    }[source_id]
    d = MASTER[key]
    cells = [display_id, source_id, label, f"{d['mean_fold_acc']*100:.2f}%", f"±{d['std_fold_acc']*100:.2f}%",
             f"{d['pooled_oof_macro_f1']:.4f}", f"{d['mean_ece']:.4f}"]
    trs.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>")
html_doc = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Stage 4 Renumbered Comparison</title><style>""" + style + """</style><body><h1>Stage 4 Comparison Table</h1><p>Renumbered display IDs, with source run IDs included to help locate saved experiment files. The RBF baseline is omitted.</p><p class="note"><strong>Best-performing configuration:</strong> Display A7 learns dynamic bilateral connections and uses joint reliability as an input. It has the highest mean accuracy (65.92%) and pooled macro-F1 (0.6402) in this table. An identical configuration was run under a second experiment ID; that duplicate is listed once and is not an independent result.</p><table><thead><tr><th>Display ID</th><th>Source run</th><th>Configuration</th><th>Mean accuracy</th><th>Fold SD</th><th>Pooled macro-F1</th><th>Mean ECE</th></tr></thead><tbody>""" + "".join(trs) + """</tbody></table><h2>Brief interpretation</h2><ul><li>Display A2, the full RCDB-STGT model, scores 60.40% mean accuracy and 0.5947 pooled macro-F1.</li><li>Results apply to this dataset and cohort-grouped protocol; cohort identifiers are not verified participant identifiers.</li></ul><p>Mean accuracy and standard deviation are across five folds. Pooled macro-F1 uses all out-of-fold predictions. ECE is mean expected calibration error across folds.</p></body></html>"""
html_path = S4 / "reports" / "STAGE4_COMPARISON_RENUMBERED.html"
html_path.write_text(html_doc, encoding="utf-8")
print(html_path)
print(md_path)
