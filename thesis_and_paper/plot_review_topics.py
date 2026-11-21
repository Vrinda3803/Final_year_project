"""Create a title-free topic pie chart from the review BibTeX file.

Each reference is assigned one primary topic using the title as the coding
unit. The explicit mapping is included below so borderline assignments can
be reviewed and changed. Outputs use only the Python standard library.
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
PDF_DIR = HERE.parent / "output" / "pdf"
SVG_PATH = FIGURE_DIR / "reviewed_papers_by_topic.svg"
PDF_PATH = PDF_DIR / "reviewed_papers_by_topic.pdf"
COUNTS_PATH = FIGURE_DIR / "reviewed_papers_by_topic.csv"
ASSIGNMENTS_PATH = FIGURE_DIR / "review_paper_topic_assignments.csv"

ENTRY_RE = re.compile(r"(?im)^\s*@\w+\s*\{\s*([^,\s]+)\s*,")
PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]

# Broad, mutually exclusive primary-topic groups. Assignments are based on
# the paper titles in the bibliography, not an abstract/full-text review.
TOPIC_GROUPS = {
    "Eye-tracking and gaze analysis": [
        4, 8, 16, 19, 20, 21, 22, 23, 24, 25, 30, 31, 32, 34, 37, 40, 42,
        43, 45, 46, 47, 48, 49, 52, 66, 76, 77, 78, 79, 80, 81, 86, 87, 88,
        89, 98, 99, 100,
    ],
    "Pose, skeleton, and movement": [
        3, 5, 9, 13, 18, 56, 62, 67, 68, 73, 74, 75, 90, 91, 94,
    ],
    "Multimodal behavioral analysis": [
        6, 11, 15, 26, 27, 28, 35, 51, 54, 55, 57, 58, 61, 63, 64, 65,
        71, 72, 83, 84, 85, 92,
    ],
    "Facial expression and emotion": [
        7, 10, 12, 33, 50, 69, 70, 82, 96, 97,
    ],
    "Social interaction and communication": [
        1, 2, 14, 17, 29, 44, 53, 60, 93, 95,
    ],
    "Other screening approaches": [36, 38, 39, 41, 59],
}


def read_braced_value(text: str, start: int) -> tuple[str, int]:
    """Read a BibTeX braced value, allowing nested braces."""
    depth = 0
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise ValueError("Unclosed braced value in BibTeX entry")


def get_field(body: str, field_name: str) -> str | None:
    match = re.search(rf"(?im)^\s*{re.escape(field_name)}\s*=\s*", body)
    if not match:
        return None
    index = match.end()
    while index < len(body) and body[index].isspace():
        index += 1
    if index >= len(body):
        return None
    if body[index] == "{":
        value, _ = read_braced_value(body, index)
        return value
    if body[index] == '"':
        end = index + 1
        escaped = False
        while end < len(body):
            if escaped:
                escaped = False
            elif body[end] == "\\":
                escaped = True
            elif body[end] == '"':
                return body[index + 1 : end]
            end += 1
        raise ValueError(f"Unclosed quoted {field_name} value")
    end = body.find(",", index)
    return body[index:] if end < 0 else body[index:end]


def clean_title(title: str) -> str:
    title = re.sub(r"\\([&%_$#])", r"\1", title)
    title = re.sub(r"\\[a-zA-Z]+\s*", "", title)
    title = title.replace("{", "").replace("}", "").replace("~", " ")
    return re.sub(r"\s+", " ", title).strip()


def read_entries() -> dict[str, str]:
    text = BIB_PATH.read_text(encoding="utf-8-sig")
    matches = list(ENTRY_RE.finditer(text))
    titles: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = get_field(text[match.end() : end], "title")
        if title is None:
            raise ValueError(f"Missing title for BibTeX key {match.group(1)}")
        titles[match.group(1)] = clean_title(title)
    return titles


def topic_for_key(key: str) -> str:
    number = int(key.removeprefix("paper"))
    for topic, numbers in TOPIC_GROUPS.items():
        if number in numbers:
            return topic
    raise ValueError(f"No topic assigned to {key}")


def make_svg(counts: Counter[str], total: int) -> str:
    ordered = sorted(counts, key=lambda topic: (-counts[topic], topic))
    width, height = 1280, 610
    cx, cy, radius = 300.0, height / 2, 245.0
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g stroke="#ffffff" stroke-width="4">',
    ]
    angle = -90.0
    for index, topic in enumerate(ordered):
        end_angle = angle + 360 * counts[topic] / total
        sx, sy = cx + radius * math.cos(math.radians(angle)), cy + radius * math.sin(math.radians(angle))
        ex, ey = cx + radius * math.cos(math.radians(end_angle)), cy + radius * math.sin(math.radians(end_angle))
        large_arc = int(end_angle - angle > 180)
        pieces.append(
            f'<path d="M {cx:.2f} {cy:.2f} L {sx:.2f} {sy:.2f} '
            f'A {radius:.2f} {radius:.2f} 0 {large_arc} 1 {ex:.2f} {ey:.2f} Z" '
            f'fill="{PALETTE[index]}"/>'
        )
        angle = end_angle
    pieces.append("</g>")

    x, row_height = 650, 57
    start_y = (height - len(ordered) * row_height) / 2 + 24
    for index, topic in enumerate(ordered):
        y = start_y + index * row_height
        count = counts[topic]
        percent = 100 * count / total
        pieces.append(f'<rect x="{x}" y="{y - 19:.1f}" width="24" height="24" rx="4" fill="{PALETTE[index]}"/>')
        pieces.append(
            f'<text x="{x + 42}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="23" fill="#202833">{topic} - {count} papers ({percent:.1f}%)</text>'
        )
    pieces.append("</svg>")
    return "\n".join(pieces)


def make_pdf(counts: Counter[str], total: int) -> bytes:
    ordered = sorted(counts, key=lambda topic: (-counts[topic], topic))
    page_width, page_height = 960.0, 540.0
    cx, cy, radius = 230.0, 270.0, 205.0
    commands = ["1 J 1 j", "2.5 w"]
    angle = -90.0
    for index, topic in enumerate(ordered):
        end_angle = angle + 360 * counts[topic] / total
        color = PALETTE[index]
        rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        commands.append(f"{rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f} rg 1 1 1 RG")
        sx, sy = cx + radius * math.cos(math.radians(angle)), cy + radius * math.sin(math.radians(angle))
        commands.append(f"{cx:.3f} {cy:.3f} m {sx:.3f} {sy:.3f} l")
        segments = max(1, math.ceil((end_angle - angle) / 90.0))
        step = (end_angle - angle) / segments
        for segment in range(segments):
            a0 = math.radians(angle + segment * step)
            a1 = math.radians(angle + (segment + 1) * step)
            factor = (4.0 / 3.0) * math.tan((a1 - a0) / 4.0)
            x0, y0 = cx + radius * math.cos(a0), cy + radius * math.sin(a0)
            x3, y3 = cx + radius * math.cos(a1), cy + radius * math.sin(a1)
            x1, y1 = x0 - factor * radius * math.sin(a0), y0 + factor * radius * math.cos(a0)
            x2, y2 = x3 + factor * radius * math.sin(a1), y3 - factor * radius * math.cos(a1)
            commands.append(f"{x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f} {x3:.3f} {y3:.3f} c")
        commands.append("h B")
        angle = end_angle

    legend_x, row_height = 470, 50.0
    start_y = page_height - (page_height - len(ordered) * row_height) / 2 - 21
    for index, topic in enumerate(ordered):
        y = start_y - index * row_height
        color = PALETTE[index]
        rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        count = counts[topic]
        percent = 100 * count / total
        label = f"{topic} - {count} papers ({percent:.1f}%)"
        escaped = label.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        commands.extend(
            [
                f"{rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f} rg",
                f"{legend_x} {y - 5:.2f} 17 17 re f",
                "0.12 0.16 0.20 rg",
                f"BT /F1 14 Tf {legend_x + 29} {y - 1:.2f} Td ({escaped}) Tj ET",
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
        pdf.extend(body + b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    return bytes(pdf)


def main() -> None:
    titles = read_entries()
    expected_keys = {f"paper{i:02d}" if i < 100 else "paper100" for i in range(1, 101)}
    if set(titles) != expected_keys:
        raise ValueError(f"Expected paper01-paper100; found {len(titles)} entries")
    assignments = {key: topic_for_key(key) for key in titles}
    topic_counts = Counter(assignments.values())
    if len(assignments) != 100 or sum(topic_counts.values()) != 100:
        raise ValueError("Every reference must have exactly one topic assignment")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(make_svg(topic_counts, len(assignments)), encoding="utf-8")
    PDF_PATH.write_bytes(make_pdf(topic_counts, len(assignments)))

    with COUNTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["primary_topic", "paper_count", "share_percent"])
        for topic, count in sorted(topic_counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([topic, count, f"{100 * count / len(assignments):.1f}"])

    with ASSIGNMENTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bib_key", "primary_topic", "title"])
        for key in sorted(titles, key=lambda item: int(item.removeprefix("paper"))):
            writer.writerow([key, assignments[key], titles[key]])

    print(f"BibTeX entries classified: {len(assignments)}")
    for topic, count in sorted(topic_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{topic}: {count} ({100 * count / len(assignments):.1f}%)")
    print(f"PDF: {PDF_PATH}")
    print(f"SVG: {SVG_PATH}")
    print(f"Counts: {COUNTS_PATH}")
    print(f"Paper assignments: {ASSIGNMENTS_PATH}")


if __name__ == "__main__":
    main()
