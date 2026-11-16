from base64 import b64encode
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = Path(r"D:\Downloads\system_architecture.png")
SYMBOLS = OUT / "reference_symbols"
TARGET = OUT / "system_architecture_with_reference_symbols.svg"

if not SOURCE.exists():
    raise FileNotFoundError(f"Could not find the supplied diagram: {SOURCE}")

png_data = b64encode(SOURCE.read_bytes()).decode("ascii")

def icon(name, x, y, w, h):
    svg = (SYMBOLS / f"{name}.svg").read_text(encoding="utf-8")
    inner = svg.split("<g fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\">", 1)[1].split("</g></svg>", 1)[0]
    return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="0 0 128 128" overflow="visible">{inner}</svg>'

layers = [
    # Top-row modules with clear empty areas in the supplied manual diagram.
    icon("02_skeleton_pose", 620, 145, 105, 112),
    icon("03_preprocessing_gear", 1071, 141, 21, 21),
    icon("04_evaluation_database", 1071, 189, 21, 21),
    icon("05_sequence_sliders", 1071, 235, 21, 21),
    icon("06_anatomical_graph", 1648, 141, 100, 110),
    icon("14_activity_prediction", 2010, 148, 78, 96),
    # Temporal-transformer symbol in its open module area.
    icon("11_temporal_transformer", 1703, 507, 110, 105),
]

TARGET.write_text(
    '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
    'width="2500" height="840" viewBox="0 0 2500 840">'
    f'<image x="0" y="0" width="2500" height="840" href="data:image/png;base64,{png_data}" '
    f'xlink:href="data:image/png;base64,{png_data}"/>'
    + "".join(layers)
    + "</svg>\n",
    encoding="utf-8",
)
print(TARGET)
