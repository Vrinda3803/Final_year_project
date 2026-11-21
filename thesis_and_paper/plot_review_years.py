"""Create a title-free pie chart of publication years in review_papers.bib.

Uses only the Python standard library and writes a vector SVG plus a CSV
summary beside this script.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
BIB_PATH = HERE / "review_papers.bib"
FIGURE_DIR = HERE / "figures"
SVG_PATH = FIGURE_DIR / "reviewed_papers_by_year.svg"
PDF_DIR = HERE.parent / "output" / "pdf"
PDF_PATH = PDF_DIR / "reviewed_papers_by_year.pdf"
CSV_PATH = FIGURE_DIR / "reviewed_papers_by_year.csv"

ENTRY_RE = re.compile(r"(?im)^\s*@([a-z]+)\s*\{\s*([^,\s]+)\s*,")
YEAR_RE = re.compile(
    r'(?im)^\s*year\s*=\s*(?:\{\s*(\d{4})\s*\}|"\s*(\d{4})\s*"|(\d{4}))'
)
PALETTE = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
    "#AA3377", "#BBBBBB", "#44AA99", "#DDCC77", "#882255",
]


def read_years(bib_path: Path) -> tuple[Counter[int], int]:
    text = bib_path.read_text(encoding="utf-8-sig")
    entries = list(ENTRY_RE.finditer(text))
    counts: Counter[int] = Counter()
    missing: list[str] = []

    for index, entry in enumerate(entries):
        end = entries[index + 1].start() if index + 1 < len(entries) else len(text)
        body = text[entry.end() : end]
        match = YEAR_RE.search(body)
        if not match:
            missing.append(entry.group(2))
            continue
        year = next(value for value in match.groups() if value is not None)
        counts[int(year)] += 1

    if missing:
        raise ValueError("Missing publication year for: " + ", ".join(missing))
    if not counts:
        raise ValueError(f"No BibTeX entries with years found in {bib_path}")
    return counts, len(entries)


def make_svg(counts: Counter[int], entry_total: int) -> str:
    years = sorted(counts)
    total = sum(counts.values())
    width = 1200
    row_height = 46
    height = max(500, 120 + len(years) * row_height)
    cx, cy = 330, height / 2
    radius = min(235, height * 0.39)

    pieces: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g stroke="#ffffff" stroke-width="3">',
    ]

    angle = -90.0
    for index, year in enumerate(years):
        fraction = counts[year] / total
        end_angle = angle + fraction * 360.0
        start_x = cx + radius * math.cos(math.radians(angle))
        start_y = cy + radius * math.sin(math.radians(angle))
        end_x = cx + radius * math.cos(math.radians(end_angle))
        end_y = cy + radius * math.sin(math.radians(end_angle))
        large_arc = 1 if end_angle - angle > 180 else 0
        color = PALETTE[index % len(PALETTE)]
        pieces.append(
            f'<path d="M {cx:.2f} {cy:.2f} L {start_x:.2f} {start_y:.2f} '
            f'A {radius:.2f} {radius:.2f} 0 {large_arc} 1 {end_x:.2f} {end_y:.2f} Z" '
            f'fill="{color}"/>'
        )
        angle = end_angle
    pieces.append("</g>")

    legend_x = 690
    legend_y = (height - len(years) * row_height) / 2 + 18
    for index, year in enumerate(years):
        y = legend_y + index * row_height
        color = PALETTE[index % len(PALETTE)]
        count = counts[year]
        share = 100 * count / total
        noun = "paper" if count == 1 else "papers"
        pieces.append(
            f'<rect x="{legend_x}" y="{y - 17:.1f}" width="22" height="22" '
            f'rx="4" fill="{color}"/>'
        )
        pieces.append(
            f'<text x="{legend_x + 38}" y="{y:.1f}" '
            'font-family="Arial, Helvetica, sans-serif" font-size="25" fill="#202833">'
            f'{year} — {count} {noun} ({share:.1f}%)</text>'
        )

    # Keep the entry count as chart metadata, not as a visible heading.
    pieces.append(f'<metadata>Counted {total} dated entries from {entry_total} BibTeX records.</metadata>')
    pieces.append("</svg>")
    return "\n".join(pieces)


def make_pdf(counts: Counter[int]) -> bytes:
    """Create a simple vector PDF without requiring plotting packages."""
    years = sorted(counts)
    total = sum(counts.values())
    page_width, page_height = 864, 504
    cx, cy, radius = 245.0, 252.0, 185.0
    commands: list[str] = ["1 J 1 j", "2 w"]

    for index, year in enumerate(years):
        fraction = counts[year] / total
        start_angle = -90.0 + 360.0 * sum(counts[y] for y in years[:index]) / total
        end_angle = start_angle + 360.0 * fraction
        color = PALETTE[index % len(PALETTE)]
        red, green, blue = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
        commands.append(f"{red:.4f} {green:.4f} {blue:.4f} rg 1 1 1 RG")
        sx = cx + radius * math.cos(math.radians(start_angle))
        sy = cy + radius * math.sin(math.radians(start_angle))
        commands.append(f"{cx:.3f} {cy:.3f} m {sx:.3f} {sy:.3f} l")

        # Approximate each circular arc with cubic Bezier segments of at most 90 degrees.
        segments = max(1, math.ceil((end_angle - start_angle) / 90.0))
        step = (end_angle - start_angle) / segments
        for segment in range(segments):
            a0 = math.radians(start_angle + segment * step)
            a1 = math.radians(start_angle + (segment + 1) * step)
            factor = (4.0 / 3.0) * math.tan((a1 - a0) / 4.0)
            x0, y0 = cx + radius * math.cos(a0), cy + radius * math.sin(a0)
            x3, y3 = cx + radius * math.cos(a1), cy + radius * math.sin(a1)
            x1 = x0 - factor * radius * math.sin(a0)
            y1 = y0 + factor * radius * math.cos(a0)
            x2 = x3 + factor * radius * math.sin(a1)
            y2 = y3 - factor * radius * math.cos(a1)
            commands.append(f"{x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f} {x3:.3f} {y3:.3f} c")
        commands.append("h B")

    legend_x = 490
    row_height = min(43.0, 330.0 / max(1, len(years)))
    start_y = page_height - (page_height - len(years) * row_height) / 2 - 22
    for index, year in enumerate(years):
        y = start_y - index * row_height
        color = PALETTE[index % len(PALETTE)]
        red, green, blue = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
        count = counts[year]
        share = 100 * count / total
        noun = "paper" if count == 1 else "papers"
        label = f"{year} - {count} {noun} ({share:.1f}%)"
        escaped = label.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        commands.extend(
            [
                f"{red:.4f} {green:.4f} {blue:.4f} rg",
                f"{legend_x} {y - 5:.2f} 16 16 re f",
                "0.12 0.16 0.20 rg",
                f"BT /F1 17 Tf {legend_x + 30} {y - 1:.2f} Td ({escaped}) Tj ET",
            ]
        )

    stream = "\n".join(commands).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def main() -> None:
    counts, entries = read_years(BIB_PATH)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(make_svg(counts, entries), encoding="utf-8")
    PDF_PATH.write_bytes(make_pdf(counts))
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", "paper_count", "share_percent"])
        total = sum(counts.values())
        for year, count in sorted(counts.items()):
            writer.writerow([year, count, f"{100 * count / total:.1f}"])

    print(f"BibTeX entries: {entries}")
    print(f"Papers counted: {sum(counts.values())}")
    print("Year counts:")
    for year, count in sorted(counts.items()):
        print(f"  {year}: {count}")
    print(f"Chart: {SVG_PATH}")
    print(f"PDF: {PDF_PATH}")
    print(f"Counts: {CSV_PATH}")


if __name__ == "__main__":
    main()
