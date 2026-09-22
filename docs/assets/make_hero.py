"""Render the side-by-side comparison as an SVG for the README.

SVG rather than a screenshot: it is text, so it diffs in git and can be regenerated from a
real run rather than re-cropped. The content here came out of examples/chat.py --compare,
not out of this file -- hero.json is what that run wrote.

It carries its own dark background so it reads the same on either GitHub theme, and the
divergent half is coloured so the point lands before the text is read.

Every run of text carries `textLength`, and that is not decoration. A viewer picks the
first monospace font it has, whose advance width is not the CHAR_W assumed here, so any
x computed as `column_start + len(prefix) * CHAR_W` lands in the middle of a glyph and the
two runs overlap -- which is exactly what happened to the first version of this file.
`textLength` with `lengthAdjust="spacingAndGlyphs"` makes each run occupy the width it was
laid out for, whatever font draws it, so the columns line up by construction instead of by
a guess about the reader's machine.
"""

import html
import json
import pathlib
import textwrap

CHAR_W = 8.0        # 14px monospace advance
LINE_H = 21
PAD = 22
COLUMN = 46         # characters per column

BG = "#0d1117"
FRAME = "#30363d"
DIM = "#8b949e"
TEXT = "#e6edf3"
SAME = "#7ee787"
DIFF = "#ffa657"


def common_prefix_words(a, b):
    """How many leading words the two share -- what gets drawn as agreeing."""
    wa, wb = a.split(), b.split()
    n = 0
    while n < min(len(wa), len(wb)) and wa[n] == wb[n]:
        n += 1
    return n


def lay_out(text, shared_words):
    """Wrapped lines, each split into (agreeing part, diverging part)."""
    words = text.split()
    agree = " ".join(words[:shared_words])
    lines = textwrap.wrap(text, COLUMN) or [""]
    out, used = [], 0
    for line in lines:
        remaining = max(0, len(agree) - used)
        head = line[:remaining]
        out.append((head, line[len(head):]))
        used += len(line) + 1
    return out


def run(content, fill, weight="normal"):
    """One <tspan> that is forced to the width its character count was laid out for."""
    return (f'<tspan fill="{fill}" font-weight="{weight}" '
            f'textLength="{len(content) * CHAR_W:.1f}" lengthAdjust="spacingAndGlyphs">'
            f'{html.escape(content)}</tspan>')


def line_at(x, y, runs):
    """A <text> anchored once, with its runs flowing after it.

    Only the first run is positioned. The rest follow, and because each is pinned to its
    own textLength the one after it starts where the previous actually ended.
    """
    runs = [r for r in runs if r[0]]
    if not runs:
        return ""
    spans = "".join(run(*r) for r in runs)
    return f'<text x="{x:.1f}" y="{y}" xml:space="preserve">{spans}</text>'


def main():
    here = pathlib.Path(__file__).parent
    data = json.loads((here / "hero.json").read_text(encoding="utf-8"))
    shared = common_prefix_words(data["reference"], data["candidate"])
    left = lay_out(data["reference"], shared)
    right = lay_out(data["candidate"], shared)
    rows = max(len(left), len(right))
    left += [("", "")] * (rows - len(left))
    right += [("", "")] * (rows - len(right))

    width = int(PAD * 2 + COLUMN * 2 * CHAR_W + 3 * CHAR_W)
    header_rows = 3
    height = int(PAD * 2 + (rows + header_rows + 2) * LINE_H)
    mid = PAD + COLUMN * CHAR_W + CHAR_W
    right_x = mid + CHAR_W * 2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-monospace,SFMono-Regular,'
        f'Menlo,Consolas,monospace" font-size="14">',
        f'<rect width="{width}" height="{height}" rx="10" fill="{BG}" stroke="{FRAME}"/>',
    ]

    y = PAD + LINE_H
    command = f'$ chat.py --load-packed {data["config"]}.bin --compare'
    parts.append(line_at(PAD, y, [(command, DIM)]))
    y += LINE_H
    parts.append(line_at(PAD, y, [(f'you> {data["prompt"]}', TEXT, "bold")]))
    y += LINE_H + 6

    parts.append(line_at(PAD, y, [("bf16", DIM, "bold")]))
    parts.append(line_at(right_x, y, [("quantized  (W4)", DIM, "bold")]))
    y += 8
    parts.append(f'<line x1="{PAD}" y1="{y}" x2="{width - PAD}" y2="{y}" '
                 f'stroke="{FRAME}"/>')
    y += LINE_H

    for (la, ld), (ra, rd) in zip(left, right, strict=True):
        parts.append(line_at(PAD, y, [(la, SAME), (ld, TEXT)]))
        parts.append(line_at(right_x, y, [(ra, SAME), (rd, DIFF)]))
        y += LINE_H

    parts.append(f'<line x1="{mid}" y1="{PAD + LINE_H * 3 + 2}" x2="{mid}" '
                 f'y2="{y - LINE_H + 6}" stroke="{FRAME}"/>')
    y += 4
    parts.append(f'<line x1="{PAD}" y1="{y - LINE_H + 2}" x2="{width - PAD}" '
                 f'y2="{y - LINE_H + 2}" stroke="{FRAME}"/>')
    parts.append(line_at(PAD, y, [(data["verdict"], DIFF)]))
    parts.append("</svg>")

    out = pathlib.Path("docs/assets")
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.svg").write_text("\n".join(p for p in parts if p), encoding="utf-8")
    print(f"wrote docs/assets/compare.svg  ({width}x{height}, {rows} rows, "
          f"{shared} shared leading words)")


if __name__ == "__main__":
    main()
