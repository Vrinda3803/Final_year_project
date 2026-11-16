from pathlib import Path

OUT = Path(__file__).resolve().parent / "reference_symbols"
OUT.mkdir(parents=True, exist_ok=True)

N = "#253D54"
B = "#4D91C1"
T = "#4E9C92"
O = "#E39B56"
R = "#D96F82"
G = "#62A879"
P = "#9276BA"
L = "#9EB6C8"

icons = {
    "01_dataset_files": f'''<g stroke="{N}" fill="#fff"><path d="M24 31h45l13 13v53H24z"/><path d="M34 22h45l13 13v53"/><path d="M44 13h45l13 13v53"/></g><path d="M59 40c-7 0-7 7-7 11s-2 6-6 7c4 1 6 3 6 7s0 11 7 11m22-36c7 0 7 7 7 11s2 6 6 7c-4 1-6 3-6 7s0 11-7 11" fill="none" stroke="{B}" stroke-width="3" stroke-linecap="round"/><path d="M67 48h8m-8 9h8m-8 9h8" stroke="{O}" stroke-width="3"/>''',
    "02_skeleton_pose": f'''<g stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><g stroke="{B}"><path d="M64 29v25m0-14-17 13m17-13 17 13m-17 1-13 23m13-23 13 23m-13 0-9 23m9-23 9 23"/></g><g fill="#fff" stroke="{N}"><circle cx="64" cy="20" r="7"/><circle cx="64" cy="29" r="4"/><circle cx="64" cy="54" r="4"/><circle cx="47" cy="53" r="4"/><circle cx="81" cy="53" r="4"/><circle cx="51" cy="77" r="4"/><circle cx="77" cy="77" r="4"/><circle cx="55" cy="100" r="4"/><circle cx="73" cy="100" r="4"/></g><circle cx="64" cy="20" r="3" fill="{R}" stroke="none"/><circle cx="47" cy="53" r="2.5" fill="{G}" stroke="none"/><circle cx="81" cy="53" r="2.5" fill="{O}" stroke="none"/></g>''',
    "03_preprocessing_gear": f'''<path d="M64 18v10m0 72v10M18 64h10m72 0h10M31 31l7 7m52 52 7 7m0-66-7 7M38 90l-7 7" stroke="{N}" stroke-width="8" stroke-linecap="round"/><circle cx="64" cy="64" r="36" fill="#fff" stroke="{N}" stroke-width="7"/><circle cx="64" cy="64" r="14" fill="#E7F0F6" stroke="{B}" stroke-width="5"/>''',
    "04_evaluation_database": f'''<ellipse cx="64" cy="28" rx="38" ry="14" fill="#EAF3F8" stroke="{N}" stroke-width="3"/><path d="M26 28v65c0 8 17 14 38 14s38-6 38-14V28" fill="#fff" stroke="{N}" stroke-width="3"/><path d="M26 49c0 8 17 14 38 14s38-6 38-14M26 71c0 8 17 14 38 14s38-6 38-14" fill="none" stroke="{B}" stroke-width="3"/><ellipse cx="64" cy="28" rx="38" ry="14" fill="none" stroke="{N}" stroke-width="3"/>''',
    "05_sequence_sliders": f'''<g stroke="{N}" stroke-width="4" stroke-linecap="round"><path d="M20 35h88M20 64h88M20 93h88"/></g><g fill="#fff" stroke-width="4"><circle cx="48" cy="35" r="8" stroke="{B}"/><circle cx="81" cy="64" r="8" stroke="{G}"/><circle cx="59" cy="93" r="8" stroke="{O}"/></g>''',
    "06_anatomical_graph": f'''<g stroke="{N}" stroke-width="3" stroke-linecap="round"><path d="M64 24v20m0 0-19 14m19-14 19 14m-38 0-11 20m11-20 14 20m24-20 11 20m-11-20L64 78m0-20v20m0 0-12 25m12-25 13 25"/></g><g fill="#fff" stroke="{B}" stroke-width="3"><circle cx="64" cy="22" r="7"/><circle cx="64" cy="44" r="5"/><circle cx="45" cy="58" r="5"/><circle cx="83" cy="58" r="5"/><circle cx="34" cy="78" r="5"/><circle cx="59" cy="78" r="5"/><circle cx="89" cy="78" r="5"/><circle cx="64" cy="58" r="5"/><circle cx="52" cy="103" r="5"/><circle cx="77" cy="103" r="5"/></g>''',
    "07_fixed_bilateral_graph": f'''<g stroke="{N}" stroke-width="3" stroke-linecap="round"><path d="M43 27v26l-13 22m13-22 11 22m0 0-8 24m8-24 8 24M85 27v26L74 75m11-22 13 22m-24 0-8 24m8-24 9 24"/></g><g stroke="{B}" stroke-width="3" stroke-dasharray="5 5"><path d="M50 53h28M38 76h44M50 99h28"/></g><g fill="#fff" stroke="{N}" stroke-width="3"><circle cx="43" cy="27" r="6"/><circle cx="85" cy="27" r="6"/><circle cx="43" cy="53" r="5"/><circle cx="85" cy="53" r="5"/><circle cx="30" cy="75" r="5"/><circle cx="54" cy="75" r="5"/><circle cx="74" cy="75" r="5"/><circle cx="98" cy="75" r="5"/><circle cx="46" cy="99" r="5"/><circle cx="60" cy="99" r="5"/><circle cx="68" cy="99" r="5"/><circle cx="82" cy="99" r="5"/></g>''',
    "08_dynamic_bilateral_graph": f'''<g stroke="{N}" stroke-width="3" stroke-linecap="round"><path d="M43 27v26l-13 22m13-22 11 22m0 0-8 24m8-24 8 24M85 27v26L74 75m11-22 13 22m-24 0-8 24m8-24 9 24"/></g><g stroke="{O}" stroke-width="4"><path d="M50 53h28M38 76h44M50 99h28"/><path d="m54 48-5 5 5 5m20-10 5 5-5 5m-33 18-7-3 1 8m48-8 7-3-1 8"/></g><g fill="#fff" stroke="{B}" stroke-width="3"><circle cx="43" cy="27" r="6"/><circle cx="85" cy="27" r="6"/><circle cx="43" cy="53" r="5"/><circle cx="85" cy="53" r="5"/><circle cx="30" cy="75" r="5"/><circle cx="54" cy="75" r="5"/><circle cx="74" cy="75" r="5"/><circle cx="98" cy="75" r="5"/><circle cx="46" cy="99" r="5"/><circle cx="60" cy="99" r="5"/><circle cx="68" cy="99" r="5"/><circle cx="82" cy="99" r="5"/></g>''',
    "09_node_features": f'''<rect x="26" y="19" width="76" height="90" rx="8" fill="#fff" stroke="{N}" stroke-width="3"/><rect x="35" y="30" width="58" height="23" rx="5" fill="#E5F1F8" stroke="{B}" stroke-width="2"/><path d="M43 38h8m-8 7h28" stroke="{B}" stroke-width="2"/><rect x="35" y="60" width="58" height="20" rx="5" fill="#F0EAF8" stroke="{P}" stroke-width="2"/><circle cx="46" cy="70" r="4" fill="{P}"/><path d="M57 70h26" stroke="{P}" stroke-width="2"/><rect x="35" y="87" width="58" height="13" rx="4" fill="#EEF2F5" stroke="{L}" stroke-width="2"/>''',
    "10_spatial_graph_attention": f'''<g stroke="{L}" stroke-width="3"><path d="M64 63 34 34m30 29 30-29M64 63l-34 6m34-6 34 6M64 63 43 98m21-35 21 35m-21-35V22"/></g><g fill="#fff" stroke="{B}" stroke-width="3"><circle cx="34" cy="34" r="7"/><circle cx="94" cy="34" r="7"/><circle cx="30" cy="69" r="7"/><circle cx="98" cy="69" r="7"/><circle cx="43" cy="98" r="7"/><circle cx="85" cy="98" r="7"/></g><circle cx="64" cy="63" r="11" fill="#EAF3F8" stroke="{N}" stroke-width="4"/><path d="M64 44V31m-5 6 5-7 5 7" stroke="{O}" stroke-width="3"/>''',
    "11_temporal_transformer": f'''<rect x="14" y="31" width="28" height="60" rx="5" fill="#EEF4F8" stroke="{N}" stroke-width="3"/><rect x="50" y="22" width="28" height="78" rx="5" fill="#F1EBF8" stroke="{P}" stroke-width="3"/><rect x="86" y="31" width="28" height="60" rx="5" fill="#EEF4F8" stroke="{N}" stroke-width="3"/><path d="M42 61h8m28 0h8" stroke="{B}" stroke-width="3"/><path d="m46 56 5 5-5 5m31-10 5 5-5 5" fill="none" stroke="{B}" stroke-width="3"/><path d="m60 46 15 9-15 9z" fill="#fff" stroke="{P}" stroke-width="2"/>''',
    "12_residual_addition": f'''<circle cx="64" cy="64" r="37" fill="#fff" stroke="{N}" stroke-width="4"/><path d="M64 43v42M43 64h42" stroke="{B}" stroke-width="6" stroke-linecap="round"/>''',
    "13_classifier_head": f'''<g stroke="{L}" stroke-width="2.5"><path d="M25 38 56 31m-31 7 31 30M25 64l31-33m-31 33 31 0m-31 0 31 33M25 90l31-26m31-33 18 8m-18 0 18 14m-18 11 18-1m-18 0 18 14m-18 13 18-27"/></g><g fill="#fff" stroke="{N}" stroke-width="3"><circle cx="25" cy="38" r="5"/><circle cx="25" cy="64" r="5"/><circle cx="25" cy="90" r="5"/><circle cx="56" cy="31" r="5"/><circle cx="56" cy="64" r="5"/><circle cx="56" cy="97" r="5"/><circle cx="87" cy="31" r="5"/><circle cx="87" cy="52" r="5"/><circle cx="87" cy="75" r="5"/><circle cx="87" cy="97" r="5"/></g><path d="M106 36v58" stroke="{G}" stroke-width="4"/>''',
    "14_activity_prediction": f'''<circle cx="37" cy="26" r="8" fill="#fff" stroke="{N}" stroke-width="3"/><g stroke="{B}" stroke-width="4" stroke-linecap="round"><path d="M37 35v27m0-17-17 14m17-14 17-14m-17 31-15 28m15-28 22 23"/></g><rect x="72" y="28" width="40" height="68" rx="6" fill="#fff" stroke="{N}" stroke-width="3"/><path d="M80 43h24m-24 12h24m-24 12h18m-18 12h24" stroke="{T}" stroke-width="3"/><circle cx="101" cy="79" r="3" fill="{O}"/>''',
    "15_skeleton_frame_stack": f'''<g fill="#fff" stroke-width="2.5"><rect x="20" y="28" width="38" height="70" rx="4" stroke="{L}"/><rect x="38" y="21" width="42" height="78" rx="4" stroke="{B}"/><rect x="59" y="28" width="45" height="70" rx="4" stroke="{L}"/></g><g stroke="{G}" stroke-width="2.5" stroke-linecap="round"><path d="M59 43v15m0-9-9 8m9-8 9 8m-9 1-7 14m7-14 7 14m-7 0-5 14m5-14 5 14"/></g><g fill="{R}" stroke="none"><circle cx="59" cy="39" r="3"/><circle cx="50" cy="57" r="3"/><circle cx="68" cy="57" r="3"/><circle cx="53" cy="71" r="3"/><circle cx="65" cy="71" r="3"/></g>''',
}

def svg(name, body):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128" role="img" aria-label="{name.replace('_', ' ')} symbol"><g fill="none" stroke-linecap="round" stroke-linejoin="round">{body}</g></svg>\n'''

for name, body in icons.items():
    (OUT / f"{name}.svg").write_text(svg(name, body), encoding="utf-8")

cols, cw, ch = 5, 160, 156
rows = (len(icons) + cols - 1) // cols
sheet = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{cols*cw}" height="{rows*ch}" viewBox="0 0 {cols*cw} {rows*ch}">', '<rect width="100%" height="100%" fill="#ffffff"/>']
for idx, (name, body) in enumerate(icons.items()):
    x, y = (idx % cols) * cw, (idx // cols) * ch
    sheet.append(f'<g transform="translate({x+16},{y+5})">{body}</g>')
    label = name.split('_', 1)[1].replace('_', ' ').title()
    sheet.append(f'<text x="{x+cw/2}" y="{y+142}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="{N}">{label}</text>')
sheet.append('</svg>')
(OUT.parent / "reference_symbols_contact_sheet.svg").write_text("\n".join(sheet), encoding="utf-8")

(OUT / "README.md").write_text(
    "# Symbols for the reference architecture\n\n"
    "These are original, clean SVG redraws of the small symbols shown in the supplied reference diagram. "
    "They are not cropped from the screenshot or copied from Google. Each has a transparent background and can be resized in Lucid.\n\n"
    "Use the numbered files to match the corresponding symbol: dataset files, pose, gear, database, sliders, anatomical graph, fixed and dynamic bilateral graphs, node features, spatial attention, temporal transformer, residual plus, classifier, activity prediction, and skeleton-frame stack.\n",
    encoding="utf-8",
)
