from pathlib import Path

OUT = Path(__file__).resolve().parent / "module_icons"
OUT.mkdir(parents=True, exist_ok=True)

N = "#24445C"
B = "#397DA3"
T = "#4E9C92"
G = "#93A7B6"
Y = "#D49A3A"

icons = {
    "dataset": f'''<rect x="29" y="26" width="62" height="72" rx="7" stroke="{N}"/><path d="M38 43h43M38 56h43M38 69h30M38 82h37" stroke="{B}"/><circle cx="78" cy="82" r="4" fill="{T}" stroke="none"/><path d="M37 20h50" stroke="{G}"/>''',
    "pose_sequence": f'''<rect x="17" y="27" width="37" height="68" rx="5" stroke="{G}"/><rect x="40" y="20" width="44" height="78" rx="6" stroke="{N}"/><rect x="69" y="28" width="42" height="68" rx="5" stroke="{G}"/><g stroke="{B}"><circle cx="62" cy="35" r="4" fill="{Y}"/><path d="M62 39v19m0-12-12 10m12-10 12 10m-12 2-9 18m9-18 9 18"/><circle cx="50" cy="57" r="3" fill="{T}"/><circle cx="74" cy="57" r="3" fill="{T}"/><circle cx="53" cy="76" r="3" fill="{T}"/><circle cx="71" cy="76" r="3" fill="{T}"/></g>''',
    "data_preparation": f'''<path d="M22 30h84L78 62v25l-20 11V62L22 30Z" stroke="{N}"/><path d="M38 42h50M47 52h32" stroke="{B}"/><circle cx="53" cy="42" r="4" fill="{T}"/><circle cx="76" cy="52" r="4" fill="{Y}"/>''',
    "anatomical_graph": f'''<g stroke="{B}"><path d="M64 26v20m0 0-19 13m19-13 19 13m-38 0-10 20m10-20 14 19m24-19 10 20m-10-20L64 85m-20-26 20 7 20-7M64 65v20m0 0-12 20m12-20 13 20"/></g><g fill="white" stroke="{N}"><circle cx="64" cy="24" r="6"/><circle cx="64" cy="46" r="5"/><circle cx="45" cy="59" r="5"/><circle cx="83" cy="59" r="5"/><circle cx="35" cy="80" r="5"/><circle cx="59" cy="83" r="5"/><circle cx="89" cy="80" r="5"/><circle cx="64" cy="65" r="5"/><circle cx="52" cy="105" r="5"/><circle cx="77" cy="105" r="5"/></g><circle cx="64" cy="24" r="2.5" fill="{Y}" stroke="none"/>''',
    "bilateral_connections": f'''<g stroke="{B}"><path d="M38 31v25l-11 20m11-20 12 20m40-45v25l-12 20m12-20 12 20M50 76l-7 25m7-25 8 25m20-25-8 25m8-25 8 25"/></g><g fill="white" stroke="{N}"><circle cx="38" cy="30" r="6"/><circle cx="38" cy="56" r="5"/><circle cx="27" cy="76" r="5"/><circle cx="50" cy="76" r="5"/><circle cx="43" cy="101" r="5"/><circle cx="58" cy="101" r="5"/><circle cx="90" cy="30" r="6"/><circle cx="90" cy="56" r="5"/><circle cx="79" cy="76" r="5"/><circle cx="102" cy="76" r="5"/><circle cx="72" cy="101" r="5"/><circle cx="87" cy="101" r="5"/></g><path d="M47 48h34" stroke="{Y}" stroke-width="3"/><path d="m52 43-6 5 6 5m24-10 6 5-6 5" stroke="{Y}"/>''',
    "joint_reliability": f'''<path d="M64 20a43 43 0 1 0 43 43" stroke="{G}" stroke-width="8"/><path d="M64 20a43 43 0 0 1 40 27" stroke="{T}" stroke-width="8"/><path d="M64 63l24-20" stroke="{N}" stroke-width="4"/><circle cx="64" cy="63" r="6" fill="{Y}" stroke="{N}"/><circle cx="64" cy="20" r="4" fill="{N}"/><circle cx="104" cy="47" r="4" fill="{N}"/><text x="64" y="91" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="{N}" stroke="none">r(t)</text>''',
    "temporal_attention": f'''<rect x="15" y="32" width="26" height="54" rx="4" stroke="{G}"/><rect x="51" y="24" width="27" height="70" rx="4" stroke="{N}"/><rect x="88" y="32" width="26" height="54" rx="4" stroke="{G}"/><path d="M40 59h10m28 0h10" stroke="{T}" stroke-width="3"/><path d="m45 54 5 5-5 5m32-10 5 5-5 5" stroke="{T}"/><g fill="{B}" stroke="none"><circle cx="28" cy="48" r="3"/><circle cx="28" cy="60" r="3"/><circle cx="28" cy="72" r="3"/><circle cx="64" cy="43" r="3"/><circle cx="64" cy="57" r="3"/><circle cx="64" cy="71" r="3"/><circle cx="64" cy="85" r="3"/><circle cx="101" cy="48" r="3"/><circle cx="101" cy="60" r="3"/><circle cx="101" cy="72" r="3"/></g>''',
    "classifier": f'''<g stroke="{G}"><path d="M24 64h20M44 64l18-26m-18 26 18 0m-18 0 18 26m0-52 20 14m-20-14 20 38m-20-12 20 12m-20-38 20-12"/></g><g fill="white" stroke="{N}"><circle cx="24" cy="64" r="6"/><circle cx="44" cy="64" r="6"/><circle cx="62" cy="38" r="5"/><circle cx="62" cy="64" r="5"/><circle cx="62" cy="90" r="5"/><circle cx="82" cy="26" r="5"/><circle cx="82" cy="52" r="5"/><circle cx="82" cy="76" r="5"/><circle cx="82" cy="102" r="5"/></g><path d="M96 36h15m-15 12h15m-15 12h15m-15 12h15m-15 12h15" stroke="{T}" stroke-width="4"/>''',
    "activity_output": f'''<circle cx="43" cy="29" r="8" fill="white" stroke="{N}"/><g stroke="{B}" stroke-width="4"><path d="M43 38v28m0-18L25 58m18-10 18-13m-18 31L26 91m17-25 19 24"/></g><path d="M77 39h35v51H77z" stroke="{N}"/><path d="M84 51h20m-20 10h20m-20 10h14m-14 10h18" stroke="{T}" stroke-width="3"/><circle cx="103" cy="80" r="3" fill="{Y}" stroke="none"/>''',
}

def doc(name, body):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128" role="img" aria-label="{name.replace('_', ' ')} icon"><g fill="none" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round">{body}</g></svg>\n'''

for name, body in icons.items():
    (OUT / f"{name}.svg").write_text(doc(name, body), encoding="utf-8")

# A labeled SVG contact sheet for choosing and copying icons into Lucid.
cols, cell_w, cell_h = 3, 210, 185
rows = (len(icons) + cols - 1) // cols
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{cols*cell_w}" height="{rows*cell_h}" viewBox="0 0 {cols*cell_w} {rows*cell_h}">',
         '<rect width="100%" height="100%" fill="#ffffff"/>']
for i, (name, body) in enumerate(icons.items()):
    x = (i % cols) * cell_w
    y = (i // cols) * cell_h
    parts.append(f'<g transform="translate({x+41},{y+12})">{body}</g>')
    label = name.replace('_', ' ').title()
    parts.append(f'<text x="{x+cell_w/2}" y="{y+165}" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" fill="{N}">{label}</text>')
parts.append('</svg>')
(OUT.parent / "module_icons_contact_sheet.svg").write_text("\n".join(parts), encoding="utf-8")
